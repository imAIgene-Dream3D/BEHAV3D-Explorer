"""
BEHAV3D napari plugin – a message box whose text scrolls.

``QMessageBox`` grows with its text, so a long list of existing outputs (many
samples × cell types) pushes the buttons off the screen.  :class:`ScrollableChoiceBox`
keeps the same "text + row of buttons" shape, but the dialog is never taller than
the BEHAV3D window (or the screen) and only the text scrolls, so the buttons stay
visible.
"""
from __future__ import annotations

from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

# Room for the window title bar and frame, which the layout does not account for.
_WINDOW_CHROME_PX = 48
# Never shrink the scrolling text below this, even in a tiny window.
_MIN_TEXT_HEIGHT_PX = 80
_ICON_PX = 32


def available_height(parent: Optional[QWidget]) -> int:
    """Height a dialog may use: the parent's window, capped by the screen."""
    screen = None
    if parent is not None and parent.window() is not None:
        screen = parent.window().screen() if hasattr(parent.window(), "screen") else None
    screen = screen or QGuiApplication.primaryScreen()
    limit = screen.availableGeometry().height() if screen is not None else 800
    if parent is not None:
        win = parent.window()
        if win is not None and win.isVisible() and win.height() > 0:
            limit = min(limit, win.frameGeometry().height())
    return limit


class ScrollableChoiceBox(QDialog):
    """Icon + scrollable text + buttons; mirrors the bits of ``QMessageBox`` we use.

    ``clickedButton()`` returns the button the user pressed.  Closing the dialog
    (Esc / window ×) counts as pressing the first button added with
    ``QMessageBox.RejectRole``, so callers never see ``None``.
    """

    def __init__(self, parent: Optional[QWidget], title: str, text: str,
                 icon: QStyle.StandardPixmap = QStyle.SP_MessageBoxWarning):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._clicked: Optional[QPushButton] = None
        self._escape: Optional[QPushButton] = None
        self._buttons: List[QPushButton] = []

        icon_label = QLabel()
        icon_label.setPixmap(self.style().standardIcon(icon).pixmap(_ICON_PX, _ICON_PX))
        icon_label.setAlignment(Qt.AlignTop)

        self._text = QLabel(text)
        self._text.setTextFormat(Qt.PlainText)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._text)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        body = QHBoxLayout()
        body.addWidget(icon_label, 0, Qt.AlignTop)
        body.addWidget(self._scroll, 1)

        self._button_row = QHBoxLayout()
        self._button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(self._button_row, 0)

    def addButton(self, label: str, role: QMessageBox.ButtonRole) -> QPushButton:  # noqa: N802
        btn = QPushButton(label)
        btn.setAutoDefault(False)
        btn.clicked.connect(lambda _checked=False, b=btn: self._on_clicked(b))
        self._button_row.addWidget(btn)
        self._buttons.append(btn)
        if role == QMessageBox.RejectRole and self._escape is None:
            self._escape = btn
        return btn

    def setDefaultButton(self, btn: QPushButton) -> None:  # noqa: N802
        for b in self._buttons:
            b.setDefault(b is btn)
        btn.setFocus()

    def clickedButton(self) -> Optional[QPushButton]:  # noqa: N802
        return self._clicked if self._clicked is not None else self._escape

    def _on_clicked(self, btn: QPushButton) -> None:
        self._clicked = btn
        self.accept()

    def _fit_to_window(self) -> None:
        """Size the text area to its content, but keep the dialog within the window."""
        max_h = available_height(self.parentWidget())
        text_h = self._text.sizeHint().height() + 2 * self._scroll.frameWidth()
        # Height of everything except the scrolling text.
        self._scroll.setMinimumHeight(0)
        self._scroll.setMaximumHeight(0)
        others_h = self.layout().sizeHint().height()
        self._scroll.setMaximumHeight(16777215)
        budget = max(_MIN_TEXT_HEIGHT_PX, max_h - others_h - _WINDOW_CHROME_PX)
        fitted = min(text_h, budget)
        if text_h > budget:
            # A vertical scrollbar appears; widen so it does not cover the text.
            sb_w = self._scroll.verticalScrollBar().sizeHint().width()
            self._scroll.setMinimumWidth(self._text.sizeHint().width() + sb_w)
        self._scroll.setMinimumHeight(fitted)
        self.setMaximumHeight(max(others_h + _MIN_TEXT_HEIGHT_PX, max_h - _WINDOW_CHROME_PX))
        self.adjustSize()

    def exec_(self) -> int:
        self._fit_to_window()
        return super().exec_()
