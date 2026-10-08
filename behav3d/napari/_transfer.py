"""GUI for packing, verifying and importing project transfer bundles.

The work itself lives in :mod:`behav3d.io.transfer`; this module only adds
the dialogs, a QThread wrapper with progress/cancel, and the menu that
``DataPreparationTab`` puts next to the output-directory field.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

from qtpy.QtCore import Qt, Signal, QThread
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from behav3d.napari._background_runner import begin_busy, end_busy, warn_if_busy


def _fmt_bytes(n):
    from behav3d.io.transfer import _fmt_bytes as fmt
    return fmt(n)


class _TransferWorker(QThread):
    """Runs one ``behav3d.io.transfer`` call with progress, log and cancel."""

    progress = Signal(object, object, str)  # done, total (bytes; may exceed int32)
    log = Signal(str)
    done = Signal(object)  # {"ok": bool, "result": ..., "error": str, "kind": str}

    def __init__(self, fn, kwargs, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._kwargs = kwargs
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        from behav3d.io.transfer import InterruptedRunError, TransferCancelled, TransferError

        try:
            result = self._fn(
                progress=lambda d, t, m: self.progress.emit(d, t, m),
                cancel=self._cancel.is_set,
                log=self.log.emit,
                **self._kwargs,
            )
            self.done.emit({"ok": True, "result": result})
        except TransferCancelled:
            self.done.emit({"ok": False, "kind": "cancelled", "error": "Cancelled."})
        except InterruptedRunError as e:
            self.done.emit({"ok": False, "kind": "interrupted", "error": str(e)})
        except TransferError as e:
            self.done.emit({"ok": False, "kind": "transfer", "error": str(e)})
        except Exception as e:  # unexpected: report, don't kill the GUI
            self.done.emit({"ok": False, "kind": "error", "error": f"{type(e).__name__}: {e}"})


class PackDialog(QDialog):
    """Pick destination, samples and verification for a pack run."""

    def __init__(self, output_dir, samples, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pack project for transfer")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Writes one archive per sample plus one for the rest of the project, "
            "with checksums, straight to the destination (e.g. a USB drive or NAS "
            "share). Copy the resulting <i>…_behav3d_bundle</i> folder as is, then "
            "use <b>Import packed project…</b> on the other computer.<br><br>"
            f"Project: <b>{Path(output_dir).name}</b>"
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        row = QHBoxLayout()
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Destination folder…")
        self.dest_edit.textChanged.connect(self._update_free_space)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.dest_edit, stretch=1)
        row.addWidget(browse)
        lay.addLayout(row)
        self.free_label = QLabel("")
        lay.addWidget(self.free_label)

        self.sample_list = QListWidget()
        for s in samples:
            item = QListWidgetItem(str(s))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.sample_list.addItem(item)
        if samples:
            lay.addWidget(QLabel("Samples (project-level files are always included):"))
            lay.addWidget(self.sample_list)
        else:
            self.sample_list.hide()

        self.verify_check = QCheckBox("Re-read and check archives after writing (recommended)")
        self.verify_check.setChecked(True)
        lay.addWidget(self.verify_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Pack")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Destination for the bundle")
        if d:
            self.dest_edit.setText(d)

    def _update_free_space(self, text):
        try:
            free = shutil.disk_usage(text).free if text and Path(text).exists() else None
        except OSError:
            free = None
        self.free_label.setText(f"Free space at destination: {_fmt_bytes(free)}" if free is not None else "")

    def _accept(self):
        if not self.dest_edit.text().strip():
            QMessageBox.warning(self, "Pack", "Choose a destination folder.")
            return
        if self.sample_list.count() and not self.selected_samples():
            QMessageBox.warning(self, "Pack", "Select at least one sample.")
            return
        self.accept()

    def destination(self):
        return self.dest_edit.text().strip()

    def selected_samples(self):
        if not self.sample_list.count():
            return None
        return [
            self.sample_list.item(i).text()
            for i in range(self.sample_list.count())
            if self.sample_list.item(i).checkState() == Qt.Checked
        ]

    def all_selected(self):
        sel = self.selected_samples()
        return sel is None or len(sel) == self.sample_list.count()


class TransferController:
    """Owns the Transfer menu of a ``DataPreparationTab`` and runs its actions."""

    def __init__(self, tab):
        self.tab = tab
        self._worker = None
        self._busy = None
        self._progress_dialog = None

    # -- widget -----------------------------------------------------------
    def make_button(self):
        btn = QToolButton(self.tab)
        btn.setText("Transfer")
        btn.setToolTip("Pack this project into a few verified archives for moving it "
                       "to another drive or computer, check a copied bundle, or import one.")
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        menu.addAction("Pack project for transfer…", self.pack)
        menu.addAction("Verify a copied bundle…", self.verify)
        menu.addAction("Import packed project…", self.import_bundle)
        btn.setMenu(menu)
        return btn

    # -- helpers ------------------------------------------------------------
    def _log(self, msg):
        self.tab._log(msg)

    def _running(self):
        return self._worker is not None and self._worker.isRunning()

    def _start(self, title, fn, kwargs, on_done):
        self._progress_dialog = dlg = QProgressDialog(title, "Cancel", 0, 1000, self.tab)
        dlg.setWindowTitle("BEHAV3D transfer")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        self._worker = worker = _TransferWorker(fn, kwargs, parent=self.tab)
        worker.log.connect(self._log, Qt.QueuedConnection)
        worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        worker.done.connect(lambda res: self._finish(res, on_done), Qt.QueuedConnection)
        dlg.canceled.connect(worker.cancel)
        worker.start()
        self._busy = begin_busy(title, alive=worker.isRunning)
        dlg.show()

    def _on_progress(self, done, total, message):
        dlg = self._progress_dialog
        if dlg is None:
            return
        frac = (done / total) if total else 0.0
        dlg.setValue(min(1000, int(frac * 1000)))
        label = f"{_fmt_bytes(done)} of {_fmt_bytes(total)}"
        if message:
            label += f"\n{message if len(message) < 90 else '…' + message[-88:]}"
        dlg.setLabelText(label)

    def _finish(self, res, on_done):
        end_busy(self._busy)
        self._busy = None
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        on_done(res)

    def _guard(self, what):
        if self._running():
            QMessageBox.information(self.tab, "Transfer", "A transfer is already running.")
            return True
        return warn_if_busy(self.tab, what)

    # -- pack -----------------------------------------------------------------
    def _sample_names(self, output_dir):
        md = getattr(self.tab, "metadata", None)
        if md is None:
            csv = Path(output_dir) / "metadata.csv"
            if csv.exists():
                try:
                    from behav3d.core.metadata import load_behav3d_metadata
                    md = load_behav3d_metadata(csv)
                except Exception:
                    md = None
        if md is None or "sample_name" not in md.columns:
            return []
        return [str(s) for s in md["sample_name"].dropna()]

    def _metadata_csv(self):
        csv = getattr(self.tab, "_loaded_csv_path", "") or ""
        if not csv and hasattr(self.tab, "csv_path_edit"):
            csv = self.tab.csv_path_edit.text().strip()
        return csv if csv and Path(csv).exists() else None

    def pack(self, force=False, _repeat=None):
        if self._guard("packing the project"):
            return
        from behav3d.io.transfer import pack_project

        if _repeat is None:
            output_dir = self.tab.output_dir_edit.text().strip()
            if not output_dir or not Path(output_dir).is_dir():
                QMessageBox.warning(self.tab, "Pack", "Set an existing output directory first.")
                return
            dlg = PackDialog(output_dir, self._sample_names(output_dir), parent=self.tab)
            if not dlg.exec_():
                return
            kwargs = dict(
                output_dir=output_dir,
                destination=dlg.destination(),
                metadata_csv=self._metadata_csv(),
                samples=None if dlg.all_selected() else dlg.selected_samples(),
                verify=dlg.verify_check.isChecked(),
            )
        else:
            kwargs = dict(_repeat)
        kwargs["force"] = force

        def on_done(res):
            if res["ok"]:
                bundle = res["result"]
                QMessageBox.information(
                    self.tab, "Pack complete",
                    f"Bundle ready:\n{bundle}\n\nCopy the whole folder. On the other computer use "
                    "Transfer → Verify (optional) and Import packed project.",
                )
            elif res["kind"] == "interrupted":
                answer = QMessageBox.question(
                    self.tab, "Interrupted segmentation",
                    res["error"] + "\n\nPack anyway (the interrupted state is packed as is)?",
                )
                if answer == QMessageBox.Yes:
                    self.pack(force=True, _repeat={k: v for k, v in kwargs.items() if k != "force"})
            elif res["kind"] == "cancelled":
                self._log("Packing cancelled; completed archives are kept and reused on the next run.")
            else:
                self._log(f"❌ Packing failed: {res['error']}")
                QMessageBox.critical(self.tab, "Pack failed", res["error"])

        self._start("Packing project…", pack_project, kwargs, on_done)

    # -- verify ---------------------------------------------------------------
    def verify(self):
        if self._guard("verifying a bundle"):
            return
        from behav3d.io.transfer import BUNDLE_JSON, verify_bundle

        bundle = QFileDialog.getExistingDirectory(self.tab, "Select the …_behav3d_bundle folder")
        if not bundle:
            return
        if not (Path(bundle) / BUNDLE_JSON).exists():
            QMessageBox.warning(self.tab, "Verify", f"No {BUNDLE_JSON} in that folder.")
            return
        box = QMessageBox(self.tab)
        box.setWindowTitle("Verify bundle")
        box.setText("Quick check compares each archive with its checksum (finds any copy error).\n"
                    "Full check also re-reads every file inside the archives.")
        quick = box.addButton("Quick check", QMessageBox.AcceptRole)
        full = box.addButton("Full check", QMessageBox.YesRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        if box.clickedButton() not in (quick, full):
            return

        def on_done(res):
            if not res["ok"]:
                if res["kind"] != "cancelled":
                    QMessageBox.critical(self.tab, "Verify failed", res["error"])
                return
            report = res["result"]
            if report["ok"]:
                QMessageBox.information(self.tab, "Verify", f"All {report['archives']} archive(s) are intact.")
            else:
                QMessageBox.critical(
                    self.tab, "Verify",
                    "Problems found - copy these archives again:\n\n" + "\n".join(report["problems"][:15]),
                )

        self._start("Verifying bundle…", verify_bundle,
                    {"bundle_dir": bundle, "deep": box.clickedButton() is full}, on_done)

    # -- import ---------------------------------------------------------------
    def import_bundle(self):
        if self._guard("importing a project"):
            return
        from behav3d.io.transfer import BUNDLE_JSON, _load_bundle_json, unpack_bundle

        bundle = QFileDialog.getExistingDirectory(self.tab, "Select the …_behav3d_bundle folder to import")
        if not bundle:
            return
        if not (Path(bundle) / BUNDLE_JSON).exists():
            QMessageBox.warning(self.tab, "Import", f"No {BUNDLE_JSON} in that folder.")
            return
        info = _load_bundle_json(bundle)
        parent_dir = QFileDialog.getExistingDirectory(self.tab, "Import into which folder?")
        if not parent_dir:
            return
        target = Path(parent_dir)
        if any(target.iterdir()):
            target = target / info.get("project_name", "behav3d_project")
        answer = QMessageBox.question(
            self.tab, "Import packed project",
            f"Import '{info.get('project_name')}' ({len(info['archives'])} archive(s)) into:\n\n{target}\n\n"
            "Every file is checked against its checksum before it is moved into place.",
        )
        if answer != QMessageBox.Yes:
            return

        def on_done(res):
            if not res["ok"]:
                if res["kind"] != "cancelled":
                    self._log(f"❌ Import failed: {res['error']}")
                    QMessageBox.critical(self.tab, "Import failed", res["error"])
                return
            result = res["result"]
            md_csv = result.get("metadata_csv")
            load = QMessageBox.question(
                self.tab, "Import complete",
                f"Project imported into:\n{result['output_dir']}\n\nLoad it now?",
            )
            if load == QMessageBox.Yes and md_csv is not None:
                self.tab.output_dir_edit.setText(str(result["output_dir"]))
                self.tab.output_dir = str(result["output_dir"])
                self.tab.csv_path_edit.setText(str(md_csv))
                self.tab._on_load_metadata()

        self._start("Importing project…", unpack_bundle,
                    {"bundle_dir": bundle, "output_dir": str(target)}, on_done)
