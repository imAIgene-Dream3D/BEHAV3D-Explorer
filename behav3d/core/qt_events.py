"""Event pumping that keeps the UI repainting without re-entering click handlers.

Long synchronous GUI-thread operations (loading training data, training log
sinks) call ``processEvents()`` so progress bars and logs repaint. A plain
``processEvents()`` also delivers *user input*, so the user can click another
button or switch tab from inside the half-finished handler -- re-entrancy that
the busy guard cannot see because the pumping handler is not a background
operation. ``pump_events`` excludes user-input events (mouse / keyboard), while
still delivering paints, timers and queued signals.
"""
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QApplication


def pump_events() -> None:
    app = QApplication.instance()
    if app is None:
        return
    try:
        app.processEvents(QEventLoop.ExcludeUserInputEvents)
    except Exception:  # unexpected enum / binding difference: fall back safely
        app.processEvents()
