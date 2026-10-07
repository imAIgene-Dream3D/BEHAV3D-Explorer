"""
BEHAV3D napari plugin – shared overwrite-prompt helper.

Provides a single ``prompt_overwrite`` function used by every tab so all
Overwrite / Skip / Cancel dialogs have the same body, default button and
return values.

Return values
-------------
The function returns one of the string tokens:

- ``"overwrite"`` — user wants to overwrite the existing data
- ``"skip"``      — user wants to skip the existing data (batch only)
- ``"cancel"``    — user dismissed the dialog
- a custom token  — when the caller passes ``extra_buttons`` and the user
  picked one of them (the token is the second element of the tuple).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from behav3d.core.run_plan import ALL_ORGANOIDS, Action, PlanItem, RunPlan, Status


def prompt_overwrite(
    parent: Optional[QWidget],
    title: str,
    items: Sequence[str],
    *,
    body_prefix: str = "The following data already exists:",
    body_suffix: str = "What do you want to do?",
    overwrite_label: str = "Overwrite",
    skip_label: str = "Skip",
    allow_skip: bool = True,
    extra_buttons: Optional[Iterable[Tuple[str, str, QMessageBox.ButtonRole]]] = None,
) -> str:
    """Show a uniform overwrite-prompt dialog.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget for modal anchoring.
    title : str
        Window title (e.g. "Overwrite Existing Features?").
    items : sequence of str
        Bullet-listed names of the existing artefacts that would be
        overwritten. Shown as ``"  • item"`` lines between
        ``body_prefix`` and ``body_suffix``.
    body_prefix, body_suffix : str
        Surrounding text around the bullet list.
    overwrite_label, skip_label : str
        Labels for the destructive / accept buttons. Use
        ``"Overwrite All"`` / ``"Skip Existing"`` for batch dialogs.
    allow_skip : bool
        When False, the Skip button is hidden (single-cell flows often
        treat Skip the same as Cancel; some callers prefer to remove
        the redundancy explicitly).
    extra_buttons : iterable of (label, token, role), optional
        Additional buttons inserted between Skip and Cancel. ``token``
        is what ``prompt_overwrite`` returns when that button is
        picked. ``role`` is a ``QMessageBox.ButtonRole`` controlling
        Qt's button ordering / default styling.

    Returns
    -------
    str
        One of ``"overwrite"``, ``"skip"``, ``"cancel"`` or one of the
        custom tokens supplied via ``extra_buttons``.
    """
    details = "\n".join(f"  • {it}" for it in items) if items else ""
    text_parts = [body_prefix]
    if details:
        text_parts.append("")
        text_parts.append(details)
    text_parts.append("")
    text_parts.append(body_suffix)

    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Warning)
    box.setText("\n".join(text_parts))

    btn_overwrite = box.addButton(overwrite_label, QMessageBox.DestructiveRole)

    extra_mapped: list[tuple[object, str]] = []
    if extra_buttons:
        for label, token, role in extra_buttons:
            btn = box.addButton(label, role)
            extra_mapped.append((btn, token))

    btn_skip = box.addButton(skip_label, QMessageBox.AcceptRole) if allow_skip else None
    btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(btn_cancel)

    box.exec_()
    clicked = box.clickedButton()

    if clicked is btn_overwrite:
        return "overwrite"
    if btn_skip is not None and clicked is btn_skip:
        return "skip"
    for btn, token in extra_mapped:
        if clicked is btn:
            return token
    return "cancel"


def prompt_overwrite_single(
    parent: Optional[QWidget],
    title: str,
    items: Sequence[str],
    *,
    skip_label: str = "Skip",
    body_suffix: str = "What do you want to do?",
    extra_buttons: Optional[Iterable[Tuple[str, str, QMessageBox.ButtonRole]]] = None,
) -> str:
    """Convenience wrapper for single-cell-type runs.

    Uses ``"Overwrite"`` and ``"Skip"`` labels. Callers that can tell some of the
    listed outputs are incomplete pass ``skip_label="Skip & Resume"``, so the button
    says what it will actually do.
    """
    return prompt_overwrite(
        parent,
        title,
        items,
        overwrite_label="Overwrite",
        skip_label=skip_label,
        body_suffix=body_suffix,
        extra_buttons=extra_buttons,
    )


def prompt_overwrite_batch(
    parent: Optional[QWidget],
    title: str,
    items: Sequence[str],
    *,
    skip_label: str = "Skip Existing",
    body_prefix: str = "The following data already exists:",
    body_suffix: str = "What do you want to do?",
    extra_buttons: Optional[Iterable[Tuple[str, str, QMessageBox.ButtonRole]]] = None,
) -> str:
    """Convenience wrapper for batch (multi-cell-type) runs.

    Uses ``"Overwrite All"`` and ``"Skip Existing"`` labels. Callers that can tell
    some of the listed outputs are incomplete pass ``skip_label="Skip & Resume"``,
    so the button says what it will actually do.
    """
    return prompt_overwrite(
        parent,
        title,
        items,
        overwrite_label="Overwrite All",
        skip_label=skip_label,
        body_prefix=body_prefix,
        body_suffix=body_suffix,
        extra_buttons=extra_buttons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-item run plan dialog (sample × cell type)
# ─────────────────────────────────────────────────────────────────────────────
_STATUS_ICON = {Status.COMPLETE: "✔", Status.PARTIAL: "◐", Status.MISSING: "○"}
_STATUS_TEXT = {
    Status.COMPLETE: "Complete",
    Status.PARTIAL: "Incomplete",
    Status.MISSING: "Missing",
}
_MAX_BODY_ROWS = 24


def _cell_type_label(cell_type: str) -> str:
    return "all organoids" if cell_type == ALL_ORGANOIDS else cell_type


def plan_needs_prompt(items: Sequence[PlanItem]) -> bool:
    """True when at least one item already has (some) output on disk."""
    return any(it.status is not Status.MISSING for it in items)


def _describe_items(items: Sequence[PlanItem]) -> List[str]:
    existing = [it for it in items if it.status is not Status.MISSING]
    lines = [
        f"  {_STATUS_ICON[it.status]} {it.sample} / {_cell_type_label(it.cell_type)}"
        f" — {it.detail or _STATUS_TEXT[it.status]}"
        for it in existing[:_MAX_BODY_ROWS]
    ]
    if len(existing) > _MAX_BODY_ROWS:
        lines.append(f"  … and {len(existing) - _MAX_BODY_ROWS} more")
    n_missing = sum(1 for it in items if it.status is Status.MISSING)
    if n_missing:
        lines.append(f"  ○ {n_missing} item(s) have no data yet and will always be run")
    return lines


class RunPlanDialog(QDialog):
    """Per-item checklist: pick Skip / Run / Overwrite for each sample × cell type.

    Rows that are incomplete or missing are locked to running (they can only be
    upgraded to Overwrite), so a run always ends with everything complete.  Only
    complete rows can be skipped.
    """

    _SKIP = "Skip (keep existing)"
    _RUN_MISSING = "Run missing"
    _RUN = "Run"
    _OVERWRITE = "Overwrite"

    def __init__(self, parent: Optional[QWidget], title: str, items: Sequence[PlanItem]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 460)
        self._items = list(items)
        self._combos: List[QComboBox] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Choose what to do for each item.\n"
            "Incomplete and missing items are always run so later steps never see gaps; "
            "set Overwrite on a complete item to redo it as well."
        ))

        self._with_step = any(it.step for it in self._items)
        headers = ["Sample", "Cell type", "Status", "Details", "Action"]
        if self._with_step:
            headers.insert(0, "Step")
        self._action_col = len(headers) - 1
        self.table = QTableWidget(len(self._items), len(headers), self)
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, it in enumerate(self._items):
            texts = [
                it.sample,
                _cell_type_label(it.cell_type),
                f"{_STATUS_ICON[it.status]} {_STATUS_TEXT[it.status]}",
                it.detail,
            ]
            if self._with_step:
                texts.insert(0, it.step)
            for col, text in enumerate(texts):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row, col, cell)
            combo = QComboBox()
            if it.status is Status.COMPLETE:
                combo.addItems([self._SKIP, self._OVERWRITE])
            elif it.status is Status.PARTIAL:
                combo.addItems([self._RUN_MISSING, self._OVERWRITE])
            else:
                combo.addItems([self._RUN])
                combo.setEnabled(False)
            self._combos.append(combo)
            self.table.setCellWidget(row, self._action_col, combo)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table, 1)

        bulk = QHBoxLayout()
        for label, slot in (
            ("Overwrite all", lambda: self._set_all(overwrite=True)),
            ("Skip complete (default)", lambda: self._set_all(overwrite=False)),
            ("Overwrite selected", lambda: self._set_selected(overwrite=True)),
            ("Reset selected", lambda: self._set_selected(overwrite=False)),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            bulk.addWidget(btn)
        bulk.addStretch(1)
        layout.addLayout(bulk)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self.reject)
        btn_run = QPushButton("Run")
        btn_run.setDefault(True)
        btn_run.clicked.connect(self.accept)
        buttons.addWidget(btn_back)
        buttons.addWidget(btn_run)
        layout.addLayout(buttons)

    def _set_row(self, row: int, overwrite: bool) -> None:
        combo = self._combos[row]
        if combo.isEnabled():
            combo.setCurrentIndex(1 if overwrite else 0)

    def _set_all(self, overwrite: bool) -> None:
        for row in range(len(self._items)):
            self._set_row(row, overwrite)

    def _set_selected(self, overwrite: bool) -> None:
        for idx in self.table.selectionModel().selectedRows():
            self._set_row(idx.row(), overwrite)

    def _action_of(self, combo: QComboBox) -> Action:
        text = combo.currentText()
        if text == self._OVERWRITE:
            return Action.OVERWRITE
        if text == self._SKIP:
            return Action.SKIP
        return Action.RUN

    def plan(self) -> RunPlan:
        choices: Dict[Tuple[str, str], Action] = {}
        for it, combo in zip(self._items, self._combos):
            choices[it.key] = self._action_of(combo)
        return RunPlan.custom(self._items, choices)

    def plans_by_step(self) -> Dict[str, RunPlan]:
        """One plan per queue step (items are keyed by sample and cell type
        only, so they must not be mixed across steps)."""
        grouped: Dict[str, list] = {}
        for it, combo in zip(self._items, self._combos):
            grouped.setdefault(it.step, []).append((it, self._action_of(combo)))
        return {
            step: RunPlan.custom([it for it, _a in rows], {it.key: a for it, a in rows})
            for step, rows in grouped.items()
        }


def prompt_run_plan(
    parent: Optional[QWidget],
    title: str,
    items: Sequence[PlanItem],
    *,
    batch: bool = True,
    body_prefix: str = "The following data already exists:",
    extra_buttons: Optional[Iterable[Tuple[str, str, QMessageBox.ButtonRole]]] = None,
) -> Tuple[str, Optional[RunPlan]]:
    """Ask what to do about existing outputs, per (sample × cell type) item.

    Returns ``(choice, plan)``:

    - ``("run", plan)``    — go ahead with ``plan`` (Overwrite / Skip / Customize)
    - ``("cancel", None)`` — the user dismissed the dialog
    - ``(token, None)``    — one of the ``extra_buttons`` tokens was picked

    "Skip Existing" never cancels: it keeps complete items, resumes incomplete
    ones and runs the missing ones.
    """
    items = list(items)
    text = "\n".join([body_prefix, "", *_describe_items(items), "", "What do you want to do?"])

    def _customize():
        dlg = RunPlanDialog(parent, title, items)
        return dlg.plan() if dlg.exec_() else None

    choice, custom = _prompt_loop(
        parent, title, text, items, batch=batch, extra_buttons=extra_buttons,
        customize=_customize,
    )
    if choice == "overwrite":
        return "run", RunPlan.overwrite_all(items)
    if choice == "skip":
        return "run", RunPlan.skip_existing(items)
    if choice == "custom":
        return "run", custom
    return choice, None


def _prompt_loop(parent, title, text, items, *, batch, extra_buttons, customize):
    """Show the Overwrite / Skip / Customize… / Cancel box until the user decides.

    Returns ``(choice, custom_result)`` with ``choice`` one of ``"overwrite"``,
    ``"skip"``, ``"custom"``, ``"cancel"`` or an extra-button token.
    ``customize()`` returns the custom result, or ``None`` for "Back" (the box is
    then shown again).
    """
    has_partial = any(it.status is Status.PARTIAL for it in items)
    if has_partial:
        skip_label = "Skip & Resume"
    else:
        skip_label = "Skip Existing" if batch else "Skip"

    while True:
        box = QMessageBox(parent)
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Warning)
        box.setText(text)
        btn_overwrite = box.addButton(
            "Overwrite All" if batch else "Overwrite", QMessageBox.DestructiveRole
        )
        btn_skip = box.addButton(skip_label, QMessageBox.AcceptRole)
        btn_custom = box.addButton("Customize…", QMessageBox.ActionRole)
        extra_mapped = []
        for label, token, role in (extra_buttons or ()):
            extra_mapped.append((box.addButton(label, role), token))
        btn_cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(btn_cancel)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is btn_overwrite:
            return "overwrite", None
        if clicked is btn_skip:
            return "skip", None
        if clicked is btn_custom:
            result = customize()
            if result is not None:
                return "custom", result
            continue  # "Back" -> show the first dialog again
        for btn, token in extra_mapped:
            if clicked is btn:
                return token, None
        return "cancel", None


def prompt_queue_plans(
    parent: Optional[QWidget],
    title: str,
    steps: Sequence[Tuple[object, str, Sequence[PlanItem]]],
    *,
    extra_text: str = "",
) -> Tuple[str, Dict[object, RunPlan]]:
    """Queue-wide version of :func:`prompt_run_plan`.

    ``steps`` is ``[(step_key, step_label, items), ...]`` (labels must be unique).
    Returns ``(choice, plans)`` where ``choice`` is ``"overwrite"``, ``"skip"``,
    ``"custom"`` or ``"cancel"`` and ``plans`` maps each ``step_key`` to its plan.
    """
    flat: List[PlanItem] = []
    for _key, label, items in steps:
        for it in items:
            flat.append(PlanItem(it.sample, it.cell_type, it.status, it.detail, step=label))

    def _plans(builder) -> Dict[object, RunPlan]:
        return {
            key: builder([it for it in flat if it.step == label])
            for key, label, _items in steps
        }

    body = ["The following data already exists:", ""]
    for _key, label, items in steps:
        lines = _describe_items(items)
        if lines:
            body.append(label)
            body.extend(lines)
    if extra_text:
        body.extend(["", extra_text])
    body.extend(["", "What do you want to do?"])

    def _customize():
        dlg = RunPlanDialog(parent, title, flat)
        if not dlg.exec_():
            return None
        by_step = dlg.plans_by_step()
        return {key: by_step[label] for key, label, _items in steps if label in by_step}

    choice, custom = _prompt_loop(
        parent, title, "\n".join(body), flat, batch=True, extra_buttons=None,
        customize=_customize,
    )
    if choice == "overwrite":
        return choice, _plans(RunPlan.overwrite_all)
    if choice == "skip":
        return choice, _plans(RunPlan.skip_existing)
    if choice == "custom":
        return choice, custom
    return "cancel", {}
