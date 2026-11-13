"""Optional trace of what moves on screen, for platform-specific glitches.

Set ``SONICLAB_UI_TRACE=1`` and SonicLab writes every widget move, resize,
show and hide, every scroll and splitter change, and the window geometry to
``~/.cache/soniclab/ui-trace.log``, with markers for imports and stem
transfers. On a Mac this shows which widget slid or which bar travelled,
without guessing. Off by default; then ``mark`` does nothing.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QSplitter, QWidget

_LOG = logging.getLogger("soniclab.ui_trace")
_ENABLED = os.environ.get("SONICLAB_UI_TRACE", "") not in ("", "0")
_KINDS = {
    QEvent.Type.Move: "move",
    QEvent.Type.Resize: "resize",
    QEvent.Type.Show: "show",
    QEvent.Type.Hide: "hide",
}


def enabled() -> bool:
    return _ENABLED


def _name(widget) -> str:
    parts = []
    while widget is not None and len(parts) < 4:
        parts.append(widget.objectName() or type(widget).__name__)
        widget = widget.parentWidget() if isinstance(widget, QWidget) else None
    return "/".join(reversed(parts))


class _Tracer(QObject):
    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._hooked: set[int] = set()

    def eventFilter(self, watched, event):
        kind = _KINDS.get(event.type())
        if kind is not None and isinstance(watched, QWidget):
            if kind == "move":
                detail = f"{event.oldPos().toTuple()} -> {event.pos().toTuple()}"
            elif kind == "resize":
                detail = f"{event.oldSize().toTuple()} -> {event.size().toTuple()}"
            else:
                detail = ""
            _LOG.info("%s %s %s", kind, _name(watched), detail)
            if kind == "show":
                self.hook(watched)
        return False

    def hook(self, root) -> None:
        """Log scroll bars and splitters under ``root`` (once each)."""
        widgets = [root, *root.findChildren(QWidget)] if isinstance(root, QWidget) else []
        for widget in widgets:
            if id(widget) in self._hooked:
                continue
            if isinstance(widget, QAbstractScrollArea):
                self._hooked.add(id(widget))
                name = _name(widget)
                widget.horizontalScrollBar().valueChanged.connect(
                    lambda value, name=name: _LOG.info("scroll-x %s %s", name, value)
                )
                widget.verticalScrollBar().valueChanged.connect(
                    lambda value, name=name: _LOG.info("scroll-y %s %s", name, value)
                )
            elif isinstance(widget, QSplitter):
                self._hooked.add(id(widget))
                name = _name(widget)
                widget.splitterMoved.connect(
                    lambda pos, index, name=name: _LOG.info("splitter %s handle %s at %s", name, index, pos)
                )


def install(window) -> None:
    if not _ENABLED:
        return
    folder = Path.home() / ".cache" / "soniclab"
    folder.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(folder / "ui-trace.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(relativeCreated)9.1f ms  %(message)s"))
    _LOG.addHandler(handler)
    _LOG.setLevel(logging.INFO)
    tracer = _Tracer(window)
    tracer.hook(window)
    QApplication.instance().installEventFilter(tracer)
    _LOG.info("trace on: platform %s, window %s, dpr %s", QApplication.platformName(), window.geometry().getRect(), window.devicePixelRatioF())


def mark(label: str) -> None:
    """A line in the trace before something happens (import, transfer)."""
    if _ENABLED:
        _LOG.info("---- %s (%.3f)", label, time.monotonic())
