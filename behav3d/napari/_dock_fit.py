"""Keep napari's dock title bar from covering the top of a docked panel.

napari's ``QtCustomTitleBar`` (the close / hide / float strip) reports a hard-coded
``sizeHint`` height of 20 px, and that is all the space the dock reserves for it. Its
*minimum* height, however, is derived from the font metrics (``font_height + 10``) and
its stylesheet adds 3 px of padding top and bottom, so on a scaled display (26 px at
150 % on Windows) the widget grows past the reserved strip and paints over the top of
the panel - the tab bar loses its first rows.

``fit_dock_title_bar`` slims the bar down to the height the dock reserves for it.
"""
from __future__ import annotations

_SLIM_QSS = "#QtCustomTitleBar { padding-top: 0px; padding-bottom: 0px; }"


def fit_dock_title_bar(dock) -> None:
    """Make ``dock`` (a napari ``QtViewerDockWidget``) title bar fit its reserved strip.

    Safe to call repeatedly. napari rebuilds the title bar whenever the dock is
    floated or re-docked, so the fix is re-applied on those signals (connected once).
    """
    if dock is None or not hasattr(dock, "title"):
        return

    def _apply(*_):
        title = dock.title
        if title.vertical:  # left-edge strip used for top/bottom docks
            return
        title.setStyleSheet(_SLIM_QSS)
        title.setFixedHeight(max(title.sizeHint().height(), title.layout().minimumSize().height()))
        dock.layout().invalidate()

    _apply()
    if not getattr(dock, "_behav3d_title_fit", False):
        dock._behav3d_title_fit = True
        dock.topLevelChanged.connect(_apply)
        dock.dockLocationChanged.connect(_apply)
