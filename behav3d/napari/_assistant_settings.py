"""
BEHAV3D assistant — choosing a model provider.

The key lives in the operating system's keychain (see ``_assistant_providers``) and
the rest in ``QSettings``, both per user. Nothing here may go in
``napari/assistant_config.json``: that file is tracked in a public repository.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import QObject, QSettings, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)

from behav3d.napari._assistant_providers import (
    PROVIDERS, ApiKey, ProviderConfig, ProviderError, check_connection, delete_api_key,
    keychain_available, list_models, load_api_key, provider_by_id, save_api_key,
)

_SHARED_SERVICE_LABEL = "BEHAV3D service (shared, no key needed)"
_DIALOG_WIDTH = 600
_MODEL_NAME_LENGTH = 26
_SUMMARY = (
    "your questions and a text summary of the plugin (sample names, image paths, "
    "experiment notes and recent logs, but never the images)"
)
_running_tasks: set = set()   # a QThread must outlive the dialog that started it


def _settings() -> QSettings:
    return QSettings("BEHAV3D", "Assistant")


def load_active_provider() -> Optional[ProviderConfig]:
    """Return the chosen provider, or ``None`` for the shared service.

    A chosen provider with no key is still returned: falling back to the shared
    service here would send a researcher's data somewhere they did not choose.
    """
    settings = _settings()
    spec = provider_by_id(str(settings.value("provider/id", "") or ""))
    if spec is None:
        return None
    model = str(settings.value(f"provider/{spec.id}/model", "") or spec.default_model)
    return ProviderConfig(spec, model, load_api_key(spec.id))


def save_active_provider(provider_id: str, model: str = "") -> None:
    settings = _settings()
    settings.setValue("provider/id", provider_id)
    if provider_id:
        settings.setValue(f"provider/{provider_id}/model", model)


def data_notice(spec=None) -> str:
    """State plainly where a researcher's data goes with this choice."""
    if spec is None:
        return (
            f"QueenB sends {_SUMMARY} to the BEHAV3D service, which sends them to "
            "DeepSeek for the answer."
        )
    notice = (
        f"Your key stays on this computer. QueenB sends {_SUMMARY} to the BEHAV3D "
        f"service for document search, then to {spec.label} for the answer. Each "
        f"message sends about 20,000 tokens or more, which {spec.label} bills to your "
        "account."
    )
    if spec.adapter == "anthropic":
        notice += (
            " If Claude declines a request, Anthropic may retry it on another Claude "
            "model, also billed to your account."
        )
    return notice


class _ProviderTask(QObject):
    """Runs one provider call off the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, work):
        super().__init__()
        self._work = work

    def run(self):
        try:
            self.succeeded.emit(self._work())
        except ProviderError as failure:
            self.failed.emit(failure.message)
        finally:
            self.finished.emit()


class ProviderSettingsDialog(QDialog):
    """Choose the shared service, or a provider with the researcher's own key."""

    def __init__(self, parent=None, *, service_supported: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Assistant model")
        self.setMinimumWidth(_DIALOG_WIDTH)
        self._service_supported = service_supported

        layout = QVBoxLayout(self)
        form = QGridLayout()
        form.setColumnStretch(1, 1)
        form.setVerticalSpacing(10)
        self.provider = QComboBox()
        self.provider.addItem(_SHARED_SERVICE_LABEL, "")
        for spec in PROVIDERS:
            self.provider.addItem(spec.label, spec.id)
        form.addWidget(QLabel("Provider"), 0, 0)
        form.addWidget(self.provider, 0, 1, 1, 2)

        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.btn_remove_key = QPushButton("Remove key")
        self.btn_remove_key.clicked.connect(self._remove_key)
        form.addWidget(QLabel("API key"), 1, 0)
        form.addWidget(self.key, 1, 1)
        form.addWidget(self.btn_remove_key, 1, 2)

        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumContentsLength(_MODEL_NAME_LENGTH)
        self.model.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.btn_load_models = QPushButton("Load models")
        self.btn_load_models.clicked.connect(self._load_models)
        form.addWidget(QLabel("Model"), 2, 0)
        form.addWidget(self.model, 2, 1)
        form.addWidget(self.btn_load_models, 2, 2)
        layout.addLayout(form)

        test_row = QHBoxLayout()
        self.btn_test = QPushButton("Test connection")
        self.btn_test.clicked.connect(self._test_connection)
        test_row.addWidget(self.btn_test)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        test_row.addWidget(self.message, stretch=1)
        layout.addLayout(test_row)

        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.provider.currentIndexChanged.connect(self._on_provider_changed)
        saved = str(_settings().value("provider/id", "") or "")
        self.provider.setCurrentIndex(max(self.provider.findData(saved), 0))
        self._on_provider_changed()

    def _spec(self):
        return provider_by_id(str(self.provider.currentData() or ""))

    def _typed_or_saved_key(self, spec) -> ApiKey:
        typed = ApiKey(self.key.text())
        return typed if typed else load_api_key(spec.id)

    def _config(self) -> Optional[ProviderConfig]:
        spec = self._spec()
        if spec is None:
            return None
        return ProviderConfig(
            spec, self.model.currentText().strip(), self._typed_or_saved_key(spec)
        )

    def _on_provider_changed(self, *_args):
        spec = self._spec()
        own = spec is not None
        for widget in (
            self.key, self.model, self.btn_load_models, self.btn_test, self.btn_remove_key,
        ):
            widget.setEnabled(own)
        self.key.clear()
        self.model.clear()
        self.message.clear()
        notice = data_notice(spec)
        if own:
            saved = bool(load_api_key(spec.id))
            self.key.setPlaceholderText(
                "Saved on this computer" if saved else f"Your {spec.label} API key"
            )
            self.btn_remove_key.setEnabled(saved)
            model = str(_settings().value(f"provider/{spec.id}/model", "") or "")
            self.model.setCurrentText(model or spec.default_model)
            if not keychain_available():
                notice += (
                    " No secure keychain was found, so your key will only last until "
                    "BEHAV3D closes."
                )
            if not self._service_supported:
                notice += " The BEHAV3D service needs updating before this can be used."
        else:
            self.key.setPlaceholderText("")
        self.notice.setText(notice)
        self._fit_notice()

    def _fit_notice(self):
        """A wrapped label has no useful size hint, and this text must never be cut:
        it is how a researcher learns where their data goes and what is billed."""
        margins = self.layout().contentsMargins()
        width = max(self.width(), _DIALOG_WIDTH) - margins.left() - margins.right()
        self.notice.setMinimumHeight(self.notice.heightForWidth(width))
        self.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.notice.setMinimumHeight(self.notice.heightForWidth(self.notice.width()))

    def _start(self, work, on_success):
        task = _ProviderTask(work)
        thread = QThread()
        task.moveToThread(thread)
        thread.started.connect(task.run)
        task.succeeded.connect(on_success)
        task.failed.connect(self._show_failure)
        task.finished.connect(thread.quit)
        pair = (thread, task)
        _running_tasks.add(pair)
        thread.finished.connect(lambda: _running_tasks.discard(pair))
        self.btn_test.setEnabled(False)
        self.btn_load_models.setEnabled(False)
        task.finished.connect(self._task_finished)
        thread.start()

    def _task_finished(self):
        self.btn_test.setEnabled(True)
        self.btn_load_models.setEnabled(True)

    def _show_failure(self, message: str):
        self.message.setText(message)

    def _needs_key(self, config: ProviderConfig) -> bool:
        if config.key:
            return False
        self.message.setText(f"Enter your {config.spec.label} API key first.")
        return True

    def _load_models(self):
        config = self._config()
        if config is None or self._needs_key(config):
            return
        self.message.setText("Loading models...")
        self._start(lambda: list_models(config), self._show_models)

    def _show_models(self, models: list):
        current = self.model.currentText()
        self.model.clear()
        self.model.addItems(models)
        self.model.setCurrentText(current)
        self.message.setText(f"{len(models)} models available.")

    def _test_connection(self):
        config = self._config()
        if config is None or self._needs_key(config):
            return
        if not config.model:
            self.message.setText("Choose a model first.")
            return
        self.message.setText("Testing...")
        self._start(
            lambda: check_connection(config),
            lambda _result: self.message.setText("Connected. The key and model work."),
        )

    def _remove_key(self):
        spec = self._spec()
        if spec is not None:
            delete_api_key(spec.id)
            self._on_provider_changed()

    def _save(self):
        config = self._config()
        if config is None:
            save_active_provider("")
            self.accept()
            return
        if self._needs_key(config):
            return
        if not config.model:
            self.message.setText("Choose a model first.")
            return
        typed = ApiKey(self.key.text())
        if typed:
            save_api_key(config.spec.id, typed)
        save_active_provider(config.spec.id, config.model)
        self.accept()
