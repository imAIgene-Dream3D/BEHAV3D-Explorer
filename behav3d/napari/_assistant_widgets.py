"""
BEHAV3D assistant — chat bubbles and the pinned plan panel.

Each message is its own small ``QTextBrowser`` so it can have rounded corners and
shrink to its text, and so a streamed reply re-renders one bubble rather than the
whole conversation. ``QTextCursor.insertMarkdown`` does not exist in every Qt
binding, hence one ``setMarkdown`` document per bubble.
"""
from __future__ import annotations

import math

from qtpy.QtCore import Qt, QTimer, QUrl, Signal
from qtpy.QtGui import (
    QBrush, QColor, QDesktopServices, QTextBlockFormat, QTextCharFormat, QTextCursor,
    QTextTable,
)
from qtpy.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QTextBrowser, QTextEdit,
    QToolButton, QVBoxLayout, QWidget,
)

from behav3d.napari._assistant_plan import PlanState

_CHAT_ROLE_COLORS = {
    "assistant": "#30363b",
    "user": "#173847",
}
_ROLE_CAPTIONS = {"assistant": "BEHAV3D Assistant", "user": "You"}
_ROLE_WIDTH_FRACTION = {"assistant": 0.94, "user": 0.85}
_LINK_COLOR = "#9fc6e0"
_BUBBLE_PADDING_X = 11
_BUBBLE_PADDING_Y = 7
_LIST_INDENT = 20
_MIN_TEXT_WIDTH = 80
_STICK_TOLERANCE = 24
_REFIT_DELAY_MS = 30
_PLAN_LIST_MAX_HEIGHT = 132
_PLAN_LIST_SLACK = 4   # rows that exactly fit still get a scrollbar without this
_PLAN_ROW_MARKS = {"done": "✓", "current": "▸", "pending": "·"}

CHAT_STYLESHEET = f"""
    QScrollArea#chatTranscript {{
        background-color: #232629;
        border: 1px solid #3a3f44;
        border-radius: 4px;
    }}
    QWidget#chatMessages {{ background-color: #232629; }}
    QLabel#chatCaption {{ color: #8f9aa3; font-size: 10px; font-weight: 600; }}
    QTextBrowser#chatBubbleAssistant, QTextBrowser#chatBubbleUser {{
        color: #e6e6e6;
        border: none;
        border-radius: 12px;
        padding: {_BUBBLE_PADDING_Y}px {_BUBBLE_PADDING_X}px;
        font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
        font-size: 13px;
        selection-background-color: #3d6b8a;
    }}
    QTextBrowser#chatBubbleAssistant {{
        background-color: {_CHAT_ROLE_COLORS['assistant']};
    }}
    QTextBrowser#chatBubbleUser {{ background-color: {_CHAT_ROLE_COLORS['user']}; }}
    QTextBrowser#chatBubbleAssistant > QWidget,
    QTextBrowser#chatBubbleUser > QWidget {{ background: transparent; }}
    QFrame#planPanel {{
        background-color: #20303a;
        border: 1px solid #3a5a6a;
        border-radius: 6px;
    }}
    QScrollArea#planSteps, QWidget#planStepList {{
        background: transparent;
        border: none;
    }}
    QLabel#planHeader {{ color: #b8c6ce; font-size: 10px; font-weight: 600; }}
    QLabel[planStatus="done"] {{ color: #82939d; font-size: 11px; }}
    QLabel[planStatus="pending"] {{ color: #c5d0d6; font-size: 11px; }}
    QLabel[planStatus="current"] {{
        color: #e0f0ff; font-size: 11px; font-weight: 600;
    }}
    QPushButton#planAdvance {{ padding: 4px 9px; font-size: 11px; }}
    QPushButton#planAdvance[emphasis="true"] {{
        background-color: #2f6f8f; border: 1px solid #5285a6;
    }}
    QPushButton#planOffer {{ padding: 3px 7px; font-size: 10px; }}
    QToolButton#planDismiss {{
        background: transparent; border: none; color: #8f9aa3; font-size: 11px;
    }}
    QToolButton#planDismiss:hover {{ color: #e6e6e6; }}
"""


def style_transcript_document(document):
    """setMarkdown renders paragraphs/lists very tightly. Walk the document and add
    line spacing + paragraph spacing so responses are readable."""
    # Qt 6 takes the line-height type as an int, which PyQt6 enums are not.
    proportional = getattr(QTextBlockFormat.ProportionalHeight, "value",
                           QTextBlockFormat.ProportionalHeight)
    block = document.begin()
    while block.isValid():
        cursor = QTextCursor(block)
        block_format = cursor.blockFormat()
        block_format.setLineHeight(140, proportional)  # 1.4× line spacing
        block_format.setTopMargin(0.0 if block == document.begin() else 3.0)
        block_format.setBottomMargin(3.0)
        cursor.setBlockFormat(block_format)
        _recolor_links(block)
        block = block.next()


def _recolor_links(block):
    """The Markdown importer stamps the application's link colour onto the text,
    which is dark blue under napari and hard to read on a bubble."""
    link_format = QTextCharFormat()
    link_format.setForeground(QBrush(QColor(_LINK_COLOR)))
    fragments = block.begin()
    while not fragments.atEnd():
        fragment = fragments.fragment()
        if fragment.charFormat().isAnchor():
            cursor = QTextCursor(block.document())
            cursor.setPosition(fragment.position())
            cursor.setPosition(
                fragment.position() + fragment.length(), QTextCursor.KeepAnchor
            )
            cursor.mergeCharFormat(link_format)
        fragments += 1


class _MessageBubble(QTextBrowser):
    """One chat message, sized to its text by :meth:`fit`."""

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.markdown = None
        self.setObjectName(
            "chatBubbleUser" if role == "user" else "chatBubbleAssistant"
        )
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.ClickFocus)
        # In the default wrap mode QTextEdit resets the document to its viewport
        # width, which is still the default size on a bubble that was just created.
        self.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._open_link)
        self.document().setDocumentMargin(0)
        self.document().setIndentWidth(_LIST_INDENT)

    @staticmethod
    def _open_link(url: QUrl):
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def wheelEvent(self, event):
        # A bubble never scrolls itself, and Ctrl+wheel would zoom only this message.
        event.ignore()

    def set_markdown(self, markdown: str):
        self.markdown = markdown
        # The stylesheet font applies at polish, and fit() must measure with it.
        self.ensurePolished()
        try:
            self.setMarkdown(markdown)
        except Exception:
            # Fallback for any Qt build lacking setMarkdown.
            self.setPlainText(markdown)
            return
        try:
            style_transcript_document(self.document())
        except Exception:
            pass  # spacing only; keep the rendered markdown

    def _has_table(self) -> bool:
        return any(
            isinstance(frame, QTextTable)
            for frame in self.document().rootFrame().childFrames()
        )

    def fit(self, available_width: int):
        """Shrink to the text, up to this role's share of ``available_width``."""
        limit = max(
            int(available_width * _ROLE_WIDTH_FRACTION[self.role])
            - 2 * _BUBBLE_PADDING_X,
            _MIN_TEXT_WIDTH,
        )
        document = self.document()
        self.setLineWrapColumnOrWidth(limit)
        width = limit
        if not self._has_table():
            width = min(limit, math.ceil(document.idealWidth()) + 1)
            self.setLineWrapColumnOrWidth(width)
        self.setFixedSize(
            width + 2 * _BUBBLE_PADDING_X,
            math.ceil(document.size().height()) + 2 * _BUBBLE_PADDING_Y,
        )


class _ChatTranscript(QScrollArea):
    """Scrolling column of message bubbles, kept at the end while it is being read
    there. Scrolling up stops the view following a streamed reply."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatTranscript")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # An on-demand bar changes the width, which refits and can toggle the bar.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._column = QWidget()
        self._column.setObjectName("chatMessages")
        self._layout = QVBoxLayout(self._column)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self.setWidget(self._column)
        self._rows: list[tuple[_MessageBubble, QLabel | None]] = []
        self._stick_to_end = True
        self._fitted_width = 0
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(_REFIT_DELAY_MS)
        self._refit_timer.timeout.connect(self._refit)
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(self._on_range_changed)

    def bubbles(self) -> list[_MessageBubble]:
        return [bubble for bubble, _caption in self._rows]

    def toPlainText(self) -> str:
        return "\n\n".join(bubble.toPlainText() for bubble in self.bubbles())

    def scroll_to_end(self):
        self._stick_to_end = True
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_scrolled(self, value: int):
        self._stick_to_end = (
            value >= self.verticalScrollBar().maximum() - _STICK_TOLERANCE
        )

    def _on_range_changed(self, _minimum: int, maximum: int):
        # The range settles several event-loop passes after a bubble grows, so
        # setting the value straight after adding text would land short of the end.
        if self._stick_to_end:
            self.verticalScrollBar().setValue(maximum)

    def _available_width(self) -> int:
        margins = self._layout.contentsMargins()
        return self.viewport().width() - margins.left() - margins.right()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._available_width() != self._fitted_width:
            self._refit_timer.start()

    def _refit(self):
        self._fitted_width = self._available_width()
        for bubble in self.bubbles():
            bubble.fit(self._fitted_width)

    def _add_row(self, role: str) -> _MessageBubble:
        caption = None
        if not self._rows or self._rows[-1][0].role != role:
            caption = QLabel(_ROLE_CAPTIONS[role])
            caption.setObjectName("chatCaption")
        alignment = Qt.AlignRight if role == "user" else Qt.AlignLeft
        before_stretch = self._layout.count() - 1
        if caption is not None:
            self._layout.insertWidget(before_stretch, caption, 0, alignment)
            before_stretch += 1
        bubble = _MessageBubble(role, self._column)
        self._layout.insertWidget(before_stretch, bubble, 0, alignment)
        self._rows.append((bubble, caption))
        return bubble

    def _remove_rows_from(self, index: int):
        for bubble, caption in self._rows[index:]:
            for widget in (bubble, caption):
                if widget is not None:
                    self._layout.removeWidget(widget)
                    widget.setParent(None)
                    widget.deleteLater()
        del self._rows[index:]

    def set_blocks(self, blocks: list[tuple[str, str]]):
        """Show ``(role, markdown)`` blocks, touching only the bubbles that differ,
        so a selection in an older message survives a streamed reply."""
        existing = self.bubbles()
        keep = 0
        while (
            keep < min(len(existing), len(blocks))
            and existing[keep].role == blocks[keep][0]
        ):
            keep += 1
        self._remove_rows_from(keep)
        width = self._available_width()
        for index, (role, markdown) in enumerate(blocks):
            bubble = existing[index] if index < keep else self._add_row(role)
            if bubble.markdown != markdown:
                bubble.set_markdown(markdown)
                bubble.fit(width)


class _PlanRow(QLabel):
    """One step of the plan. Clicking it marks the step done or not done."""

    clicked = Signal()

    def __init__(self, text: str, status: str, tooltip: str, parent=None):
        super().__init__(text, parent)
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setProperty("planStatus", status)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit()
        event.accept()


class _PlanPanel(QFrame):
    """The checklist pinned above the message box."""

    advanceRequested = Signal()
    dismissRequested = Signal()
    offerAccepted = Signal()
    stepToggled = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("planPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 6)
        layout.setSpacing(3)

        header_row = QHBoxLayout()
        self.header = QLabel("Plan")
        self.header.setObjectName("planHeader")
        header_row.addWidget(self.header)
        header_row.addStretch(1)
        self.btn_dismiss = QToolButton()
        self.btn_dismiss.setObjectName("planDismiss")
        self.btn_dismiss.setText("✕")
        self.btn_dismiss.setToolTip("Dismiss this plan")
        self.btn_dismiss.clicked.connect(self.dismissRequested)
        header_row.addWidget(self.btn_dismiss)
        layout.addLayout(header_row)

        self._steps = QScrollArea()
        self._steps.setObjectName("planSteps")
        self._steps.setWidgetResizable(True)
        self._steps.setFrameShape(QFrame.NoFrame)
        self._steps.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._steps.setMaximumHeight(_PLAN_LIST_MAX_HEIGHT)
        step_list = QWidget()
        step_list.setObjectName("planStepList")
        self._rows_layout = QVBoxLayout(step_list)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._steps.setWidget(step_list)
        layout.addWidget(self._steps)

        button_row = QHBoxLayout()
        self.btn_offer = QPushButton()
        self.btn_offer.setObjectName("planOffer")
        self.btn_offer.setToolTip("Replace this plan with the steps in the latest reply")
        self.btn_offer.clicked.connect(self.offerAccepted)
        button_row.addWidget(self.btn_offer)
        button_row.addStretch(1)
        self.btn_advance = QPushButton()
        self.btn_advance.setObjectName("planAdvance")
        self.btn_advance.clicked.connect(self.advanceRequested)
        button_row.addWidget(self.btn_advance)
        layout.addLayout(button_row)

    def row_labels(self) -> list[_PlanRow]:
        return [
            self._rows_layout.itemAt(index).widget()
            for index in range(self._rows_layout.count())
        ]

    def show_plan(self, plan: PlanState, *, enabled: bool, emphasis: bool):
        self.setVisible(plan.active)
        if not plan.active:
            return
        while self._rows_layout.count():
            widget = self._rows_layout.takeAt(0).widget()
            widget.setParent(None)
            widget.deleteLater()
        for index, step in enumerate(plan.steps):
            status = plan.status(index)
            row = _PlanRow(
                f"{_PLAN_ROW_MARKS[status]}  {index + 1}  {step.title}",
                status,
                f"{step.detail}\n\nClick to mark this step "
                f"{'not done' if step.done else 'done'}.",
            )
            row.clicked.connect(lambda i=index: self.stepToggled.emit(i))
            self._rows_layout.addWidget(row)
            # A layout shows a new child on a later event, and until then counts it
            # as empty, which would collapse the list to nothing in _fit_steps.
            row.show()
        self.header.setText(f"Plan · {plan.done_count} of {len(plan.steps)} done")
        self.btn_offer.setVisible(bool(plan.offer))
        if plan.offer:
            self.btn_offer.setText(f"Use these {len(plan.offer)} steps")
        self.btn_advance.setText(plan.action_label())
        self.btn_advance.setEnabled(enabled)
        self.btn_advance.setProperty("emphasis", "true" if emphasis else "false")
        self.btn_advance.style().unpolish(self.btn_advance)
        self.btn_advance.style().polish(self.btn_advance)
        self._fit_steps()

    def _fit_steps(self):
        """Size the list to its rows until the cap, after which it scrolls. Wrapped
        rows have no useful size hint, so ask the layout for the current width."""
        for row in self.row_labels():
            row.ensurePolished()
        height = self._rows_layout.totalHeightForWidth(self._steps.viewport().width())
        self._steps.setFixedHeight(min(height + _PLAN_LIST_SLACK, _PLAN_LIST_MAX_HEIGHT))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_steps()
