import bisect
from collections import OrderedDict
import logging
import sys
import math
import time

import numpy as np
from shiboken6 import isValid

from PySide6.QtCore import QLineF, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPen, QPainter, QPolygonF

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.core.bar_map import (
    bar_index,
    bar_span,
    beat_lines,
    clean,
    extend,
    format_clock,
    grid_bars,
    move_line,
)
from soniclab.core.beats import analyze_beats
from soniclab.core.waveform import cached_envelope, display_envelope, pixel_columns
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGraphicsLineItem,
    QStyle,
    QToolTip,
)

# Musical time starts at x = 0. Track controls live in a widget column beside
# the view, so the scene no longer reserves a header gutter.
TRACK_HEIGHT = 76
# Held while dragging to take one bar out of a track. Qt reports the Mac's
# Command key as Control.
BAR_DRAG_KEY = "⌘" if sys.platform == "darwin" else "Ctrl"


DRAG_LOG = logging.getLogger("soniclab.drag")


def track_canvas(track) -> int:
    """Which canvas a track is on: 0 is the main one, 1 the second."""
    try:
        return int(track.get("canvas", 0) or 0)
    except (TypeError, ValueError):
        return 0
HEADER_WIDTH = 0
PIXELS_PER_SECOND = 90.0
RULER_HEIGHT = 32
# Waveforms are cached as images this many pixels wide, a few per clip.
WAVE_TILE = 1024
WAVE_TILES_KEPT = 8
# How the canvas follows the playhead during playback.
FOLLOW_OFF = "off"  # the canvas stays put; the playhead moves across it
FOLLOW_PAGE = "follow"  # glide only when the playhead reaches the right edge
FOLLOW_CENTER = "center"  # keep the playhead in the middle
FOLLOW_MODES = (FOLLOW_OFF, FOLLOW_PAGE, FOLLOW_CENTER)
FOLLOW_EASE_SECONDS = 0.09  # time constant of the glide after a jump
USER_SCROLL_HOLD = 1.5  # seconds a manual scroll pauses following
PLAYHEAD_COLOR = QColor("#F4F6F8")
PLAYHEAD_GLOW = QColor(244, 246, 248, 46)
PLAYHEAD_EDGE = QColor("#0C0D10")
# A stem block playing on its own (its ▶ turned to ⏸): tint, edge, cursor.
PREVIEW_TINT = QColor(120, 200, 150, 40)
PREVIEW_PAUSED_TINT = QColor(200, 200, 200, 22)
PREVIEW_EDGE = QColor("#8FE0A8")
PREVIEW_CURSOR = QColor("#E9FFF0")
CLIP_HEIGHT = 54
CLIP_TOP = 10
CLIP_BODY = QColor("#1C382C")
CLIP_EDGE = QColor("#3E6A52")
WAVE_FILL = QColor("#8FCB9B")
WAVE_CENTER = QColor(46, 78, 62, 160)
BEAT_MARK = QColor(214, 230, 218, 90)
BAR_MARK = QColor(232, 242, 234, 200)
DETECTED_DOWNBEAT = QColor(244, 214, 150, 235)
ONSET_EARLY = QColor(126, 168, 214, 230)
ONSET_ON_TIME = QColor(214, 230, 218, 170)
ONSET_LATE = QColor(196, 112, 96, 230)
ONSET_UNMATCHED = QColor(180, 186, 194, 150)
ONSET_COLORS = {
    "early": ONSET_EARLY,
    "on_time": ONSET_ON_TIME,
    "late": ONSET_LATE,
    "unmatched": ONSET_UNMATCHED,
}

PALETTE = [
    "#6E93B0",
    "#6F9A86",
    "#8E86A8",
    "#C4A46A",
    "#A67B80",
    "#7E92A3",
]


def parse_time_signature(value: object) -> tuple[int, int]:
    """Accept '4/4', (4, 4), [4, 4], or a numerator integer."""
    try:
        if isinstance(value, str):
            parts = value.strip().split("/")
            numerator = int(parts[0])
            denominator = int(parts[1]) if len(parts) > 1 else 4

        elif isinstance(value, (tuple, list)):
            numerator = int(value[0])
            denominator = int(value[1]) if len(value) > 1 else 4

        elif isinstance(value, (int, float)):
            numerator = int(value)
            denominator = 4

        else:
            return 4, 4

        if numerator <= 0 or denominator <= 0:
            return 4, 4

        return numerator, denominator

    except (ValueError, TypeError, IndexError):
        return 4, 4


def track_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def clip_local_beats(events, offset: float, duration: float) -> list[tuple[float, bool]]:
    """Map source-audio beats into a clip.

    ``local_seconds = time_seconds - offset``. A mark is included when
    ``0 <= local_seconds < duration``. Beat position 1 is a downbeat.
    ``events`` are ``(time_seconds, beat_position)`` pairs.
    """
    origin = float(offset)
    length = float(duration)
    marks = []
    for time_seconds, position in events:
        local = float(time_seconds) - origin
        if local < 0.0 or local >= length:
            continue
        marks.append((local, int(position) == 1))
    return marks


class ClipItem(QGraphicsRectItem):
    def __init__(
        self,
        clip_index,
        name,
        audio,
        duration,
        color,
        moved_callback,
        snap_pixels,
        pixels_per_second=PIXELS_PER_SECOND,
        clip_height=CLIP_HEIGHT,
        grid_snap=True,
        trim_callback=None,
        release_callback=None,
        fade_in=0.0,
        fade_out=0.0,
    ):
        self.pixels_per_second = float(pixels_per_second)
        self._clip_height = float(clip_height)
        super().__init__(
            0,
            0,
            max(30, duration * self.pixels_per_second),
            self._clip_height,
        )

        self.clip_index = clip_index
        self.name = name
        self.moved_callback = moved_callback
        self.snap_pixels = max(1.0, float(snap_pixels))
        self._snap_enabled = False
        self._report_moves = False
        self._grid_snap = bool(grid_snap)
        self.trim_callback = trim_callback
        self.release_callback = release_callback
        self.downbeat_callback = None
        self.rename_callback = None
        self._fade_in = float(fade_in)
        self._fade_out = float(fade_out)
        self._edge = None
        # Scene x of every beat when the bars are uneven; else snap_pixels.
        self.snap_points: list[float] = []
        self._trim_right = 0.0
        self._y_locked = False
        self._locked_y = 0.0
        self.setAcceptHoverEvents(True)

        self._accent = QColor(color)
        self.setBrush(QBrush(CLIP_BODY))
        self.setPen(QPen(CLIP_EDGE, 1))

        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        # Paint gets the exposed strip, so a moving playhead redraws a few
        # pixels of the clip instead of all of it.
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

        self.setZValue(3)
        self.setToolTip(
            f"{name} · {duration:.2f}s · drag to place · double-click the name to rename"
        )

        footer = 14.0 if self._clip_height >= 36 else 0.0
        top = 4.0
        bottom = max(top + 4.0, self._clip_height - footer - 2.0)
        self._wave_mid = (top + bottom) / 2.0
        self._wave_gain = max(1.0, (bottom - top) / 2.0)
        self._footer = footer
        self._source = cached_envelope(audio)
        self._fit_width = -1
        self._mins, self._maxs = self._source
        self._upper = self._mins
        self._lower = self._maxs
        self._tiles: OrderedDict = OrderedDict()  # tile index -> (bytes, QImage)
        self._tiles_key = None
        self._fit_envelope()
        self._beats: tuple[float, ...] = ()
        self._beat_origin = 0.0
        self._bar_every = 4
        self._show_beats = True
        self._detected: tuple[tuple[float, int], ...] = ()
        self._detected_offset = 0.0
        self._show_detected = False
        self._onsets: tuple[tuple[float, str], ...] = ()
        self._show_onsets = False
        self._bar_cuts: tuple[float, ...] = ()
        # (bar, local_x0, local_x1, local_t0, local_t1) for each painted block.
        self._block_spans: tuple[tuple, ...] = ()
        self._block_press = None
        self._block_dragging = False
        self._hidden_span = None
        # Preview voices on this clip's blocks: bar -> (playing, local x).
        self._preview: dict[int, tuple[bool, float]] = {}
        self.block_drag_callback = None
        self.block_drop_callback = None
        # Called on release with (clip, scene position, scene position of the
        # press); returns True when it took the drop (the other canvas).
        self.canvas_drop_callback = None
        self._press_scene = None
        self._press_play = None
        self._dragging_whole = False  # a one-block clip past the drag distance
        self._press_item_pos = QPointF()
        # ``outside_drop(clip, scene pos)``: True when released on neither canvas.
        self.outside_drop = None
        self._press_downbeat = None
        # While the clip is dragged it may change rows: ``row_snap(clip, y)``
        # returns the y of the row under the clip, and ``row_drop(clip)`` on
        # release returns True when it moved the clip to another row.
        self.row_snap = None
        self.row_drop = None
        self._row_drag = False
        # Bars picked with the selection box (bar numbers, span[0]). A plain
        # drag on a picked bar moves every picked bar (``group_drag`` /
        # ``group_drop``); ``drag_move`` reports whole-clip drags so the other
        # canvas can show where they would land.
        self.picked: set[int] = set()
        self.group_drag = None
        self.group_drop = None
        self.drag_move = None
        self._group = False
        self._duration = float(duration)
        self._draw_title(name)

    @property
    def envelope(self) -> tuple[np.ndarray, np.ndarray]:
        return self._mins, self._maxs

    def set_beats(self, beats, origin: float, bar_every: int, visible: bool) -> None:
        self._beats = tuple(float(beat) for beat in beats)
        self._beat_origin = float(origin)
        self._bar_every = max(1, int(bar_every))
        self._show_beats = bool(visible)
        self.update()

    def beat_markers(self) -> list[tuple[float, bool]]:
        """Clip-local beat times and whether each one starts a bar.

        Times are the detected timestamps, not waveform buckets.
        """
        if not self._show_beats:
            return []
        marks = []
        for index, beat in enumerate(self._beats):
            local = beat - self._beat_origin
            if local < -1e-6 or local > self._duration + 1e-6:
                continue
            marks.append((local, index % self._bar_every == 0))
        return marks

    def set_detected_beats(self, events, offset: float, visible: bool) -> None:
        """Store Madmom source times. Positions are recalculated when painting."""
        self._detected = tuple((float(time), int(position)) for time, position in events)
        self._detected_offset = float(offset)
        self._show_detected = bool(visible)
        self.update()

    def detected_markers(self) -> list[tuple[float, bool]]:
        """Clip-local Madmom beats. The second value is true for a downbeat."""
        if not self._show_detected:
            return []
        return clip_local_beats(self._detected, self._detected_offset, self._duration)

    def set_onsets(self, marks, visible: bool) -> None:
        """Store clip-local onset times. Kinds are early, on_time, late, or unmatched."""
        self._onsets = tuple((float(local), str(kind)) for local, kind in marks)
        self._show_onsets = bool(visible)
        self.update()

    def onset_markers(self) -> list[tuple[float, str]]:
        if not self._show_onsets:
            return []
        return list(self._onsets)

    def _fit_envelope(self) -> None:
        width = max(1, int(self.rect().width()))
        if width == self._fit_width:
            return
        self._mins, self._maxs = display_envelope(self._source[0], self._source[1], width)
        self._upper, self._lower = pixel_columns(self._mins, self._maxs, width)
        self._fit_width = width
        self._tiles.clear()

    def _draw_title(self, name):
        shown = name if len(name) <= 32 else name[:31] + "…"
        title = QGraphicsSimpleTextItem(shown, self)
        title.setBrush(QBrush(QColor("#E7F2EA")))
        title.setFont(QFont("DejaVu Sans", 8))
        title.setPos(8, max(2.0, self._clip_height - self._footer))
        title.setZValue(6)
        title.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        title.setToolTip("Double-click the name to rename")
        self._title = title

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        exposed = option.exposedRect
        if exposed.isNull() or exposed.width() <= 0:
            exposed = self.rect()
        painter.save()
        self._paint_wave(painter, exposed.left(), exposed.right())
        self._paint_fades(painter)
        self._paint_beats(painter, exposed.left(), exposed.right())
        self._paint_detected(painter, exposed.left(), exposed.right())
        self._paint_onsets(painter, exposed.left(), exposed.right())
        if self._footer > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#14261C"))
            painter.drawRect(
                0,
                self.rect().height() - self._footer,
                self.rect().width(),
                self._footer,
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._accent)
        painter.drawRect(0, 0, 3, self.rect().height())
        if option.state & QStyle.StateFlag.State_Selected:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#D7B36A"), 1))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        self._paint_block_gaps(painter)
        if self.picked:
            for span in self._block_spans:
                if span[0] in self.picked:
                    block = QRectF(float(span[1]) + 1, 1, max(0.0, float(span[2]) - float(span[1]) - 2), self.rect().height() - 2)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(215, 179, 106, 46))
                    painter.drawRect(block)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor("#D7B36A"), 1.5))
                    painter.drawRect(block)
        if self._preview:
            self._paint_preview(painter)
        if self._hidden_span is not None:
            left, right = self._hidden_span
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#0C0D10"))
            painter.drawRect(QRectF(left, 0, max(0.0, right - left), self.rect().height()))
        painter.restore()

    def set_bar_cuts(self, cuts) -> None:
        """Clip-local x positions where this track splits into bar blocks."""
        self._bar_cuts = tuple(float(x) for x in cuts)
        self.update()

    def set_block_spans(self, cuts, spans) -> None:
        """Remember each bar block.

        A plain drag moves the whole track, bar blocks and all. With more
        than one block, Ctrl-drag (⌘-drag on macOS) takes one bar out of it.
        """
        self.set_bar_cuts(cuts)
        self._block_spans = tuple(spans)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        if len(self._block_spans) > 1:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setToolTip(
                f"{self.name} · drag a bar block to move just that block"
                " · double-click the name to rename"
            )

    def _block_rect(self, bar: int) -> QRectF | None:
        for span in self._block_spans or ((0, 0.0, float(self.rect().width()), 0.0, 0.0),):
            if span[0] == bar:
                return QRectF(float(span[1]), 0, max(0.0, float(span[2]) - float(span[1])), self.rect().height())
        return None

    def set_preview(self, marks: dict) -> None:
        """Show which blocks play on their own: ``bar -> (playing, local x)``.

        Repaints only what changed: a whole block when it starts, pauses or
        stops, otherwise the two thin strips its cursor moved between.
        """
        old = self._preview
        if marks == old:
            return
        self._preview = dict(marks)
        for bar in set(old) | set(marks):
            before, after = old.get(bar), marks.get(bar)
            if before == after:
                continue
            block = self._block_rect(bar)
            if block is None:
                continue
            if before is None or after is None or before[0] != after[0]:
                self.update(block.adjusted(-2, 0, 2, 0))
                continue
            for x in (before[1], after[1]):
                self.update(QRectF(x - 3, 0, 6, self.rect().height()))

    def _paint_preview(self, painter) -> None:
        height = self.rect().height()
        for bar, (playing, x) in self._preview.items():
            block = self._block_rect(bar)
            if block is None:
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PREVIEW_TINT if playing else PREVIEW_PAUSED_TINT)
            painter.drawRect(block)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(PREVIEW_EDGE if playing else CLIP_EDGE, 1.5))
            painter.drawRect(block.adjusted(1, 1, -1, -1))
            painter.fillRect(QRectF(x - 1.0, 1, 2.0, height - 2), PREVIEW_CURSOR)

    def _span_at(self, x: float):
        for span in self._block_spans:
            if span[1] - 1 <= x <= span[2] + 1:
                return span
        return None

    def _paint_block_gaps(self, painter) -> None:
        if not self._bar_cuts:
            return
        height = self.rect().height()
        width = self.rect().width()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0C0D10"))
        for x in self._bar_cuts:
            painter.drawRect(QRectF(x - 3, 0, 6, height))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(CLIP_EDGE, 1))
        edges = (0.0, *self._bar_cuts, width)
        for left, right in zip(edges, edges[1:]):
            if right - left < 4:
                continue
            painter.drawRect(QRectF(left + 1, 1, right - left - 5, height - 2))

    def _paint_wave(self, painter, left: float, right: float):
        self._fit_envelope()
        width = int(self._upper.size)
        if width == 0:
            return
        left = max(0, int(left) - 1)
        right = min(width, int(math.ceil(right)) + 2)
        for tile in range(left // WAVE_TILE, (right - 1) // WAVE_TILE + 1):
            image = self._wave_tile(tile, width)
            origin = tile * WAVE_TILE
            start = max(origin, left)
            stop = min(origin + image.width(), right)
            if stop > start:
                painter.drawImage(start, 0, image, start - origin, 0, stop - start, image.height())

    def _wave_tile(self, tile: int, width: int) -> QImage:
        """One cached slice of the waveform image, drawn on first use."""
        key = (width, max(1, int(self.rect().height())), self._show_beats, self._beats)
        if key != self._tiles_key:
            self._tiles.clear()
            self._tiles_key = key
        cached = self._tiles.get(tile)
        if cached is not None:
            self._tiles.move_to_end(tile)
            return cached[1]
        start = tile * WAVE_TILE
        cached = self._rasterize(start, min(width, start + WAVE_TILE))
        self._tiles[tile] = cached
        while len(self._tiles) > WAVE_TILES_KEPT:
            self._tiles.popitem(last=False)
        return cached[1]

    def _rasterize(self, start: int, stop: int):
        span = max(1, stop - start)
        height = max(1, int(self.rect().height()))
        buffer = np.zeros((height, span, 4), dtype=np.uint8)
        if self._upper.size:
            upper = self._upper[start:stop]
            lower = self._lower[start:stop]
            mid = self._wave_mid
            gain = self._wave_gain
            top = np.clip(np.rint(mid - upper * gain), 0, height - 1).astype(np.int32)
            bot = np.clip(np.rint(mid - lower * gain), 0, height - 1).astype(np.int32)
            top = np.minimum(top, bot)
            rows = np.arange(height, dtype=np.int32)[:, None]
            mask = (rows >= top) & (rows <= bot)
            buffer[mask, 0] = 143
            buffer[mask, 1] = 203
            buffer[mask, 2] = 155
            buffer[mask, 3] = 255
            center = int(np.clip(round(mid), 0, height - 1))
            buffer[center, :, 0] = 46
            buffer[center, :, 1] = 78
            buffer[center, :, 2] = 62
            buffer[center, :, 3] = 170
            for local, bar in self.beat_markers():
                x = int(round(local * self.pixels_per_second))
                column = x - start
                if x < 4 or x > self.rect().width() - 2 or column < 0 or column >= span:
                    continue
                end = min(height, 13 if bar else 7)
                if bar:
                    buffer[2:end, column] = (232, 242, 234, 200)
                else:
                    buffer[2:end, column] = (214, 230, 218, 90)
        raw = np.ascontiguousarray(buffer).tobytes()
        image = QImage(raw, span, height, span * 4, QImage.Format.Format_RGBA8888)
        return raw, image

    def _paint_beats(self, painter, left: float, right: float):
        for local, bar in self.beat_markers():
            x = int(round(local * self.pixels_per_second))
            if x < left - 1 or x > right + 1 or x < 4 or x > self.rect().width() - 2:
                continue
            painter.setPen(QPen(BAR_MARK if bar else BEAT_MARK, 1.4 if bar else 1))
            painter.drawLine(x, 2, x, 13 if bar else 7)

    def _paint_detected(self, painter, left: float, right: float):
        for local, downbeat in self.detected_markers():
            if not downbeat:
                continue
            x = int(round(local * self.pixels_per_second))
            if x < left - 1 or x > right + 1 or x < 0 or x > self.rect().width() - 1:
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(DETECTED_DOWNBEAT)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x - 5, 2),
                        QPointF(x + 5, 2),
                        QPointF(x, 12),
                    ]
                )
            )
            painter.setPen(QPen(DETECTED_DOWNBEAT, 1.4))
            painter.drawLine(x, 12, x, 20)

    def _paint_onsets(self, painter, left: float, right: float):
        for local, kind in self.onset_markers():
            x = int(round(local * self.pixels_per_second))
            if x < left - 1 or x > right + 1 or x < 0 or x > self.rect().width() - 1:
                continue
            painter.setPen(QPen(ONSET_COLORS.get(kind, ONSET_UNMATCHED), 1.6))
            painter.drawLine(x, 24, x, 34)

    def _paint_fades(self, painter):
        width = self.rect().width()
        height = self.rect().height()
        painter.setBrush(QColor(8, 10, 14, 90))
        painter.setPen(Qt.PenStyle.NoPen)
        fade_in = min(width * 0.45, self._fade_in * self.pixels_per_second)
        fade_out = min(width * 0.45, self._fade_out * self.pixels_per_second)
        if fade_in > 1:
            painter.drawPolygon(
                QPolygonF([QPointF(0, 0), QPointF(fade_in, 0), QPointF(0, height)])
            )
        if fade_out > 1:
            painter.drawPolygon(
                QPolygonF(
                    [QPointF(width, 0), QPointF(width - fade_out, 0), QPointF(width, height)]
                )
            )

    def _edge_at(self, x):
        if x <= 6:
            return "left"
        if x >= self.rect().width() - 6:
            return "right"
        return None

    def hoverMoveEvent(self, event):
        if self._edge_at(event.pos().x()):
            cursor = Qt.CursorShape.SizeHorCursor
        elif len(self._block_spans) > 1:
            cursor = Qt.CursorShape.OpenHandCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def _downbeat_hit(self, x: float) -> float | None:
        best = None
        best_distance = 7.0
        for local, downbeat in self.detected_markers():
            if not downbeat:
                continue
            distance = abs(local * self.pixels_per_second - x)
            if distance <= best_distance:
                best = float(local)
                best_distance = distance
        return best

    def _play_button_at(self, pos: QPointF):
        for child in self.childItems():
            if isinstance(child, BarPlayButton) and child.isVisible():
                if child.mapRectToParent(child.rect()).contains(pos):
                    return child
        return None

    def mousePressEvent(self, event):
        self._press_scene = QPointF(event.scenePos())
        self._press_play = (
            self._play_button_at(event.pos()) if event.button() == Qt.MouseButton.LeftButton else None
        )
        # The downbeat marker is the top of the clip. The trim handles stay
        # on the lower part of the same edge. A click on a marker opens its
        # menu on release; pressing it and moving drags the clip, since the
        # marker sits on the corner people grab a bar by.
        self._press_downbeat = None
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.pos().y() <= 16
            and self.downbeat_callback is not None
        ):
            local = self._downbeat_hit(event.pos().x())
            if local is not None:
                self._press_downbeat = (local, event.screenPos())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.picked
            and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
            and self._edge_at(event.pos().x()) is None
            and self.group_drag is not None
        ):
            span = self._span_at(event.pos().x())
            if span is not None and span[0] in self.picked:
                # A picked bar: drag every picked bar together.
                self._block_press = (span, QPointF(event.scenePos()))
                self._block_dragging = False
                self._group = True
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and self.group_drag is not None:
            # Pressing anywhere else ends the bar selection.
            self.group_drag(None, None, None, None)
        if event.button() == Qt.MouseButton.LeftButton and self._press_play is not None:
            # ▶ / ⏸ is a button: it plays or pauses on release and never drags.
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and len(self._block_spans) > 1
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and self._edge_at(event.pos().x()) is None
        ):
            # A left-drag moves the one stem block under the mouse. The rest
            # of the track, its row and the canvas stay where they are.
            span = self._span_at(event.pos().x())
            if span is not None:
                if self.scene() is not None:
                    self.scene().clearSelection()
                self.setSelected(True)
                self._block_press = (span, QPointF(event.scenePos()))
                self._block_dragging = False
                event.accept()
                return
        edge = (
            self._edge_at(event.pos().x())
            if event.button() == Qt.MouseButton.LeftButton and self._press_downbeat is None
            else None
        )
        if edge:
            self._edge = edge
            self._report_moves = False
            self._trim_right = float(self.pos().x() + self.rect().width())
            event.accept()
            return
        self._row_drag = event.button() == Qt.MouseButton.LeftButton
        self._dragging_whole = False
        self._press_item_pos = QPointF(self.pos())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        footer = max(self._footer, 14.0)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.pos().y() >= self._clip_height - footer
            and self.rename_callback is not None
        ):
            self.rename_callback(self.clip_index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _past_drag_distance(self, scene_pos) -> bool:
        press = self._press_scene
        return press is None or (QPointF(scene_pos) - press).manhattanLength() >= QApplication.startDragDistance()

    def mouseMoveEvent(self, event):
        if self._press_play is not None:
            event.accept()  # a press on ▶ / ⏸ does not drag
            return
        if self._block_press is not None:
            span, origin = self._block_press
            moved = self._block_dragging or self._past_drag_distance(event.scenePos())
            if moved:
                self._block_dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                callback = self.group_drag if self._group else self.block_drag_callback
                if callback:
                    callback(self, span, QPointF(event.scenePos()), origin)
            event.accept()
            return
        if not self._edge:
            if not self._dragging_whole and not self._past_drag_distance(event.scenePos()):
                event.accept()  # a click, or a hand that wobbled: not a drag yet
                return
            self._dragging_whole = True
            super().mouseMoveEvent(event)
            if self.drag_move is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.drag_move(self, QPointF(event.scenePos()))
            return
        minimum = 8.0 if not self._grid_snap else max(8.0, self.snap_pixels)
        if self._edge == "right":
            right = self._arrangement_x(event.scenePos().x())
            width = max(minimum, right - float(self.pos().x()))
            self.setRect(0, 0, width, self._clip_height)
        else:
            left = self._arrangement_x(event.scenePos().x())
            left = min(left, self._trim_right - minimum)
            self.setPos(left, self._locked_y if self._y_locked else self.pos().y())
            self.setRect(0, 0, max(minimum, self._trim_right - left), self._clip_height)
        event.accept()

    def mouseReleaseEvent(self, event):
        button, self._press_play = getattr(self, "_press_play", None), None
        marker, self._press_downbeat = getattr(self, "_press_downbeat", None), None
        press = getattr(self, "_press_scene", None)
        clicked = press is not None and (QPointF(event.scenePos()) - press).manhattanLength() < 4
        if marker is not None and clicked and self.downbeat_callback is not None:
            # A click on a downbeat marker, not a drag: open its menu.
            super().mouseReleaseEvent(event)
            self._press_scene = None
            self.downbeat_callback(self, marker[0], marker[1])
            return
        if (
            button is not None
            and press is not None
            and (QPointF(event.scenePos()) - press).manhattanLength() < 4
        ):
            # A click on the bar's play button, not a drag. End the press as
            # Qt would, so no row drag or move is left half-started.
            self._block_press = None
            self._block_dragging = False
            self._group = False
            self._press_scene = None
            self._row_drag = False
            if self._y_locked and abs(float(self.pos().y()) - self._locked_y) > 0.5:
                self.setPos(self.pos().x(), self._locked_y)
            super().mouseReleaseEvent(event)
            if isValid(button):
                button.play()
            return
        if self._block_press is not None:
            span, origin = self._block_press
            dragging = self._block_dragging
            group, self._group = self._group, False
            self._block_press = None
            self._block_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            scene_pos = QPointF(event.scenePos())
            callback = self.group_drop if group else self.block_drop_callback
            if dragging and callback:
                callback(self, span, scene_pos, origin)
            event.accept()
            return
        edge = self._edge
        self._edge = None
        super().mouseReleaseEvent(event)
        if edge:
            self._report_moves = True
            if self.trim_callback:
                start = max(0.0, (float(self.pos().x()) - HEADER_WIDTH) / self.pixels_per_second)
                duration = max(0.01, float(self.rect().width()) / self.pixels_per_second)
                self.trim_callback(self.clip_index, start, duration)
            return
        press, self._press_scene = self._press_scene, None
        row_drag, self._row_drag = self._row_drag, False
        if self.drag_move is not None:
            self.drag_move(self, None)  # the drag is over: clear any preview
        DRAG_LOG.info(
            "release clip %s: press %s scene %s row_drag %s modifiers %s",
            self.clip_index,
            None if press is None else (round(press.x()), round(press.y())),
            (round(event.scenePos().x()), round(event.scenePos().y())),
            row_drag,
            int(event.modifiers().value),
        )
        outside = self.outside_drop is not None and self.outside_drop(self, QPointF(event.scenePos()))
        if outside:
            # Released on neither canvas: back where it was picked up.
            self.setPos(self._press_item_pos)
        elif (
            press is not None
            and self.canvas_drop_callback is not None
            and self.canvas_drop_callback(self, QPointF(event.scenePos()), press)
        ):
            return
        if not outside and row_drag and self.row_drop is not None and self.row_drop(self):
            return
        if self._y_locked and abs(float(self.pos().y()) - self._locked_y) > 0.5:
            self.setPos(self.pos().x(), self._locked_y)  # same row after all
        if self.release_callback:
            self.release_callback(self.clip_index)

    def _arrangement_x(self, x):
        origin = float(HEADER_WIDTH)
        if not self._snap_enabled or not self._grid_snap:
            return max(origin, float(x))
        if self.snap_points:
            position = bisect.bisect_left(self.snap_points, float(x))
            nearby = self.snap_points[max(0, position - 1) : position + 1]
            return max(origin, min(nearby, key=lambda point: abs(point - float(x))))
        relative = float(x) - origin
        snapped = math.floor(relative / self.snap_pixels + 0.5) * self.snap_pixels
        if snapped < 0:
            snapped = 0.0
        return origin + snapped

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionChange:
            x = self._arrangement_x(value.x())
            y = self._locked_y if self._y_locked else float(value.y())
            if self._row_drag and self._y_locked and self.row_snap is not None:
                y = self.row_snap(self, float(value.y()))
            return type(value)(x, y)

        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._y_locked and not self._row_drag:
                self._locked_y = float(self.pos().y())
                self._y_locked = True
            seconds = max(
                0.0,
                (float(self.pos().x()) - HEADER_WIDTH) / self.pixels_per_second,
            )
            if self._report_moves and self.moved_callback:
                self.moved_callback(self.clip_index, seconds)

        return super().itemChange(change, value)


class PlayheadItem(QGraphicsItem):
    """The transport indicator: one line across every track of a canvas.

    A hairline with a soft glow and a handle in the ruler. It moves with
    ``setPos``, so each frame repaints only the strip it left and the strip
    it entered; the clips and grid underneath are not rebuilt.
    """

    HALF_WIDTH = 6.0
    HANDLE_HEIGHT = 12.0

    def __init__(self, height: float):
        super().__init__()
        self._height = float(height)
        self.setZValue(26)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_height(self, height: float) -> None:
        if abs(height - self._height) >= 1:
            self.prepareGeometryChange()
            self._height = float(height)

    def line(self) -> QLineF:
        """The indicator in scene coordinates."""
        x = self.x()
        return QLineF(x, 0, x, self._height)

    def boundingRect(self) -> QRectF:
        # One pixel of slack each side for antialiased edges.
        return QRectF(-self.HALF_WIDTH - 1, 0, 2 * self.HALF_WIDTH + 2, self._height)

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillRect(QRectF(-1.5, RULER_HEIGHT, 3.0, self._height - RULER_HEIGHT), PLAYHEAD_GLOW)
        painter.fillRect(QRectF(-0.5, 0, 1.0, self._height), PLAYHEAD_COLOR)
        half = self.HALF_WIDTH
        handle = QPolygonF(
            [
                QPointF(-half, 1),
                QPointF(half, 1),
                QPointF(half, self.HANDLE_HEIGHT - half * 0.6),
                QPointF(0, self.HANDLE_HEIGHT),
                QPointF(-half, self.HANDLE_HEIGHT - half * 0.6),
            ]
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(PLAYHEAD_EDGE, 1))
        painter.setBrush(PLAYHEAD_COLOR)
        painter.drawPolygon(handle)
        painter.restore()


class BarPlayButton(QGraphicsRectItem):
    """Play one bar. Sits in the middle of that bar's block."""

    def __init__(self, view: "TimelineView", index: int):
        super().__init__(0, 0, 16, 16)
        self._view = view
        self._index = index
        self.setZValue(15)
        self.setBrush(QBrush(QColor("#1A201C")))
        self.setPen(QPen(QColor("#C6A56A"), 1))
        self.playing = False  # shows ⏸ while this block's voice plays
        self.setToolTip(f"Play bar {index + 1} of this stem on its own")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    @property
    def index(self) -> int:
        return self._index

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing)
        if playing == self.playing:
            return
        self.playing = playing
        verb = "Pause" if playing else "Play"
        self.setToolTip(f"{verb} bar {self._index + 1} of this stem on its own")
        self.setPen(QPen(PREVIEW_EDGE if playing else QColor("#C6A56A"), 1))
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(self.rect(), 2, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#E7F2EA")))
        if self.playing:
            painter.drawRect(QRectF(4.5, 4, 2.5, 8))
            painter.drawRect(QRectF(9.0, 4, 2.5, 8))
        else:
            painter.drawPolygon(
                QPolygonF([QPointF(5, 4), QPointF(5, 12), QPointF(12, 8)])
            )

    def play(self) -> None:
        clip = self.parentItem()
        track = clip.clip_index if isinstance(clip, ClipItem) else -1
        self._view.bar_play_requested.emit(self._index, track)

    def mousePressEvent(self, event):
        # Let the clip underneath take the press: a click still plays the bar
        # (see ClipItem.mouseReleaseEvent), but pressing here and moving drags
        # the clip, which on a short bar is mostly covered by this button.
        event.ignore()


class TimelineView(QGraphicsView):
    seek_requested = Signal(float)
    track_selected = Signal(int)
    layout_changed = Signal()
    # The whole new bar map after a bar line is dragged on the ruler.
    bars_changed = Signal(list)
    # Zero-based bar double-clicked on the ruler.
    bar_edit_requested = Signal(int)
    # Zero-based first and last bar of a Shift-click range on the ruler.
    bars_selected = Signal(int, int)
    # Zero-based bar whose play button was pressed.
    bar_play_requested = Signal(int, int)  # bar, and the track whose ▶ it was (-1: none)
    # Zero-based bar and a global position, from a click on a downbeat marker.
    downbeat_clicked = Signal(int, QPoint)
    # Track whose name was double-clicked on the clip.
    rename_requested = Signal(int)
    # Track index, canvas, new start: a whole clip dropped on the other canvas.
    canvas_move_requested = Signal(int, int, float)
    # Clips dropped on the other canvas: their indices (the selection), the
    # one that was grabbed, how far in time they moved, and the other
    # canvas's display row under the pointer.
    canvas_drop_requested = Signal(list, int, float, int)
    # Picked bars dragged and dropped: [(track index, local start, local
    # duration, bar)], the position in that list of the grabbed one, how far
    # in time they moved, the canvas they were dropped on, and its display
    # row under the pointer.
    blocks_move_requested = Signal(list, int, float, int, int)
    # Track index and display row: a whole clip dropped on another row of
    # this canvas (a row past the last one means a new row).
    row_move_requested = Signal(int, int)

    def __init__(self, parent=None, canvas: int = 0):
        super().__init__(parent)
        # Which canvas this view shows: tracks whose ``canvas`` field is this
        # number (0, the default, is the main canvas). ``peer`` is the view
        # of the other canvas, if there is one; clips and bar blocks dropped
        # on it move there.
        self.canvas = int(canvas)
        self.peer: "TimelineView | None" = None
        self.empty_text = ""
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setBackgroundBrush(QColor("#0C0D10"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(240)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # addLine/addRect/addText/addItem results are owned by Python.
        # Without these references, garbage collection deletes the timeline.
        self._retained: list = []
        self.playhead: PlayheadItem | None = None
        self.pixels_per_second = float(PIXELS_PER_SECOND)
        self.track_height = TRACK_HEIGHT
        self.snap_to_grid = True
        self.show_beats = True
        self.show_detected = True
        self.show_onsets = True
        self.on_trim = None
        self.on_release = None
        self.on_block_move = None
        self._row_heights: list[int] = []
        self._args = None
        self._rebuilding = False
        self._in_resize = False
        self._create_playhead(500)
        self._loop_region: tuple[float, float] | None = None
        self._loop_region_item = None
        self._bar_map: list[float] = []
        self._bar_lines: list[float] = []
        self._bar_fallback = 2.0
        self._snap_points: list[float] = []
        self._bar_handles: list = []
        self._bar_drag: tuple[int, QGraphicsLineItem] | None = None
        self._bar_press_x = 0.0
        self._bar_anchor = 0
        self._canvas_pan = None
        self._canvas_panned = False
        self._empty_press = None
        self._band = None  # (scene origin, rect item, add to picks) while drawing a bar-selection box
        self._picks: dict[int, set[int]] = {}
        self._group_drag = None
        self._ghosts: list = []  # outlines of dragged items shown on this canvas
        self._block_ghost = None
        self._block_drag = None
        self._scrub = False  # dragging the playhead handle in the ruler
        self._clip_cache: dict = {}  # track index -> ClipItem
        self._previewed: set[int] = set()  # clips showing a block voice
        self._user_scrolled_at = -1e9  # monotonic time of the last manual scroll
        self.horizontalScrollBar().sliderPressed.connect(self._note_user_scroll)
        self.scene_obj.selectionChanged.connect(self._on_selection)

    @property
    def bar_map(self) -> list[float]:
        """The bar lines the caller passed in, cleaned. Empty on the tempo grid."""
        return list(self._bar_map)

    @property
    def bar_lines(self) -> list[float]:
        """Every bar line drawn, including repeats past the map."""
        return list(self._bar_lines)

    @property
    def bar_fallback(self) -> float:
        """Bar length from the tempo, used where the map has no lines."""
        return self._bar_fallback

    def row_height(self, index: int) -> int:
        if 0 <= index < len(self._row_heights):
            return self._row_heights[index]
        return int(self.track_height)

    def set_row_height(self, index: int, height: int):
        height = max(TRACK_HEIGHT, min(220, int(height)))
        while len(self._row_heights) <= index:
            self._row_heights.append(int(self.track_height))
        self._row_heights[index] = height

    def zoom_horizontal(self, factor: float):
        self.pixels_per_second = float(
            min(480.0, max(24.0, self.pixels_per_second * factor))
        )
        self._rerender()

    def zoom_vertical(self, factor: float):
        self.track_height = int(
            max(TRACK_HEIGHT, min(200, round(self.track_height * factor)))
        )
        self._row_heights = []
        self._rerender()

    def reset_zoom(self):
        self.pixels_per_second = float(PIXELS_PER_SECOND)
        self.track_height = TRACK_HEIGHT
        self._row_heights = []
        self._rerender()

    def reset_track_heights(self):
        self.track_height = TRACK_HEIGHT
        self._row_heights = []
        self._rerender()

    def _rerender(self):
        if not self._args:
            return
        tracks, on_move, bpm, signature, bars = self._args
        self.rebuild(tracks, on_move, bpm, signature, bars)

    def _keep(self, item):
        self._retained.append(item)
        return item

    def clear_scene(self) -> None:
        """Drop Python wrappers, then destroy the scene's C++ items.

        Plain Qt items (lines, text) do not tell shiboken when they die. A
        wrapper still held after ``scene.clear()`` stays mapped to the freed
        address, and a later item allocated there comes back as the old type.
        """
        self.playhead = None
        self._clip_cache = {}
        self._previewed = set()
        self._loop_region_item = None
        self._bar_handles = []
        self._bar_drag = None
        self._block_ghost = None
        self._block_drag = None
        self._retained.clear()
        self.scene_obj.clear()

    def _create_playhead(self, height) -> PlayheadItem:
        playhead = self._keep(PlayheadItem(height))
        playhead.setPos(HEADER_WIDTH, 0)
        playhead.setVisible(getattr(self, "_playhead_shown", True))
        self.scene_obj.addItem(playhead)
        self.playhead = playhead
        return playhead

    def _live_playhead(self, height) -> PlayheadItem:
        current = self.playhead
        if current is not None and isValid(current):
            return current
        self.playhead = None
        return self._create_playhead(height)

    def _row_top(self, index: int) -> float:
        top = float(RULER_HEIGHT)
        for row in range(index):
            top += self.row_height(row)
        return top

    def lane_rows(self, tracks) -> tuple[list[int], list[int]]:
        """Display row of each track, and the first track index of each row.

        Tracks that share a ``lane`` stay on one row. A track without one
        keeps its own row. A track on another canvas has row -1 here.
        """
        rows: list[int] = []
        primaries: list[int] = []
        seen: dict = {}
        for index, track in enumerate(tracks):
            if track_canvas(track) != self.canvas:
                rows.append(-1)
                continue
            lane = track.get("lane")
            key = ("shared", lane) if lane is not None else ("solo", index)
            row = seen.get(key)
            if row is None:
                row = len(primaries)
                seen[key] = row
                primaries.append(index)
            rows.append(row)
        return rows, primaries

    def _display_top(self, display_row: int, primaries: list[int]) -> float:
        top = float(RULER_HEIGHT)
        for row in range(min(display_row, len(primaries))):
            top += self.row_height(primaries[row])
        return top

    def rebuild(self, tracks, on_move, bpm=120.0, time_signature="4/4", bars=None):
        # Release Python-owned items before the scene destroys their C++ objects.
        self._rebuilding = True
        scroll_x = self.horizontalScrollBar().value()
        scroll_y = self.verticalScrollBar().value()
        self._args = (tracks, on_move, bpm, time_signature, bars)
        self.clear_scene()

        pps = self.pixels_per_second
        max_duration = max(
            [t.get("duration", 10) + t.get("start", 0) for t in tracks] + [30]
        )
        width = HEADER_WIDTH + max_duration * pps + 500
        display_rows, primaries = self.lane_rows(tracks)
        content_bottom = self._display_top(len(primaries), primaries)
        viewport_height = self.viewport().height() if self.viewport() is not None else 0
        height = max(500, content_bottom + TRACK_HEIGHT, viewport_height)
        self.scene_obj.setSceneRect(0, 0, width, height)
        self._create_playhead(height)

        self._keep(
            self.scene_obj.addRect(
                0,
                0,
                width,
                RULER_HEIGHT,
                QPen(Qt.PenStyle.NoPen),
                QBrush(QColor("#101318")),
            )
        )
        self._keep(
            self.scene_obj.addLine(
                0,
                RULER_HEIGHT,
                width,
                RULER_HEIGHT,
                QPen(QColor("#2A3038"), 1),
            )
        )

        # A beat is one denominator-note. A bar is `numerator` of those beats.
        # Bar lines come from the caller's bar map when given, else the tempo.
        beats_per_bar, beat_unit = parse_time_signature(time_signature)
        bpm = max(1.0, float(bpm))
        beat_seconds = (4.0 / beat_unit) * 60.0 / bpm
        bar_duration = max(0.01, beats_per_bar * beat_seconds)
        snap_pixels = max(1.0, beat_seconds * pps)
        until = (width - HEADER_WIDTH) / pps
        if bars:
            self._bar_map = clean(bars)
        else:
            self._bar_map = []
        lines = extend(self._bar_map, until, bar_duration) if self._bar_map else grid_bars(
            bpm, beats_per_bar, beat_unit, until
        )
        self._bar_lines = lines
        self._bar_fallback = bar_duration
        marks = beat_lines(lines, beats_per_bar)
        if lines and lines[0] > 1e-6:
            # Bar 1 is the first detected downbeat, often seconds into the
            # song. The intro before it gets beats too (counting back at the
            # first bar's pace) and the song's start, so clips can be drawn on
            # a grid there and dragged all the way back to 0 s.
            first_bar = (lines[1] - lines[0]) if len(lines) > 1 else bar_duration
            beat = max(0.01, first_bar / max(1, beats_per_bar))
            pickup = []
            time = lines[0] - beat
            while time > 1e-6:
                pickup.append((round(time, 6), False))
                time -= beat
            marks = [(0.0, False)] + pickup[::-1] + marks
        self._snap_points = [HEADER_WIDTH + time * pps for time, _bar in marks]
        mapped = len(self._bar_map)
        bar_pen = QPen(QColor("#343C48"), 1)
        beat_pen = QPen(QColor("#222830"), 1)
        tick_pen = QPen(QColor("#3A424E"), 1)
        previous = None
        number = 0
        for time, is_bar in marks:
            x = HEADER_WIDTH + time * pps
            if is_bar:
                self._keep(self.scene_obj.addLine(x, 0, x, height, bar_pen))
                number += 1
                label = self._keep(self.scene_obj.addText(str(number)))
                label.document().setDocumentMargin(0)
                label.setFont(QFont("DejaVu Sans Mono", 8))
                # Lines the caller placed are gold; repeats past the map are dim.
                label.setDefaultTextColor(QColor("#C6A56A" if not mapped or number <= mapped else "#7D6C4C"))
                label.setPos(x + 6, 6)
                label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            else:
                spacing = x - previous if previous is not None else 0.0
                if spacing >= 6:
                    self._keep(self.scene_obj.addLine(x, RULER_HEIGHT, x, height, beat_pen))
                    if spacing >= 28:
                        tick = x - spacing * 0.5
                        self._keep(self.scene_obj.addLine(tick, 20, tick, RULER_HEIGHT - 1, tick_pen))
            previous = x
        self._bar_handles = []
        if mapped:
            handle_pen = QPen(QColor("#D7B36A"), 2)
            for time in self._bar_map:
                x = HEADER_WIDTH + time * pps
                self._bar_handles.append(self._keep(self.scene_obj.addLine(x, 0, x, RULER_HEIGHT, handle_pen)))

        for display, primary in enumerate(primaries):
            row_h = self.row_height(primary)
            y = self._display_top(display, primaries)
            lane = QColor("#12161B") if display % 2 == 0 else QColor("#161B22")
            self._keep(
                self.scene_obj.addRect(
                    0,
                    y,
                    width,
                    row_h,
                    QPen(Qt.PenStyle.NoPen),
                    QBrush(lane),
                )
            )
            self._keep(
                self.scene_obj.addLine(
                    0,
                    y + row_h,
                    width,
                    y + row_h,
                    QPen(QColor("#222830"), 1),
                )
            )

        if not primaries and self.empty_text:
            hint = self._keep(self.scene_obj.addText(self.empty_text))
            hint.setDefaultTextColor(QColor("#8B939F"))
            hint.setPos(HEADER_WIDTH + 16, RULER_HEIGHT + 14)
            hint.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        for row, track in enumerate(tracks):
            display = display_rows[row]
            if display < 0:
                continue  # on the other canvas
            row_h = self.row_height(primaries[display])
            y = self._display_top(display, primaries)
            audio = track.get("_audio")
            if audio is None or len(audio) == 0:
                continue
            duration = track.get("duration", 0)
            if duration <= 0:
                continue

            clip_height, clip_top = self._clip_box(row_h)
            offset_seconds = float(track.get("offset", 0.0) or 0.0)
            start_i = max(0, int(offset_seconds * OUTPUT_RATE))
            count_i = max(1, int(float(duration) * OUTPUT_RATE))
            visible = audio[start_i : start_i + count_i]
            if len(visible) == 0:
                visible = audio[:1]
            clip = ClipItem(
                row,
                track.get("name", "Audio"),
                visible,
                duration,
                track_color(row),
                on_move,
                snap_pixels,
                pps,
                clip_height,
                self.snap_to_grid,
                self.on_trim,
                self.on_release,
                float(track.get("fade_in", 0.0) or 0.0),
                float(track.get("fade_out", 0.0) or 0.0),
            )
            self.scene_obj.addItem(self._keep(clip))
            clip.setPos(
                HEADER_WIDTH + track.get("start", 0) * pps,
                y + clip_top,
            )
            clip._snap_enabled = True
            clip.snap_points = self._snap_points
            clip._report_moves = True
            clip.downbeat_callback = self._downbeat_pressed
            clip.rename_callback = self.rename_requested.emit
            clip.block_drag_callback = self._drag_block
            clip.block_drop_callback = self._drop_block
            clip.canvas_drop_callback = self._drop_clip_on_peer
            clip.outside_drop = lambda _clip, scene_pos: not self.over_canvas(scene_pos)
            clip.row_snap = self._snap_to_row
            clip.row_drop = self._drop_clip_on_row
            clip.group_drag = self._drag_group
            clip.group_drop = self._drop_group
            clip.drag_move = self._preview_on_peer
            if track.get("bar_block"):
                # A placed bar can land on top of other audio on its row. It
                # is drawn, and grabbed, above it.
                clip.setZValue(4)
            beats, tempo = self._beat_track(track, audio)
            clip.set_beats(beats, offset_seconds, beats_per_bar, self.show_beats)
            if tempo > 0:
                clip.setToolTip(f"{clip.toolTip()} · {tempo:.1f} BPM detected")
            self._place_bar_blocks(
                clip,
                float(track.get("start", 0.0) or 0.0),
                float(duration),
                whole=bool(track.get("bar_block")),
            )

        self._apply_picks()
        self._paint_loop_region()
        self.horizontalScrollBar().setValue(scroll_x)
        self.verticalScrollBar().setValue(scroll_y)
        self._rebuilding = False
        self.layout_changed.emit()

    def _place_bar_blocks(self, clip: ClipItem, start: float, duration: float, whole: bool = False) -> None:
        """Split one track into a block per bar and center a play button in each.

        ``whole`` keeps the clip as one block: a bar that was dragged out of
        its track is a single piece even where it no longer lines up with the
        bar lines, so it can be dragged again as a whole.
        """
        if not self._bar_lines or duration <= 0:
            return
        pps = self.pixels_per_second
        width = float(clip.rect().width())
        end = start + duration
        if whole:
            bar = bar_index(self._bar_lines, start + duration / 2, self._bar_fallback)
            clip.set_block_spans([], [(max(0, bar), 0.0, width, 0.0, duration)])
            if width >= 22:
                button = BarPlayButton(self, max(0, bar))
                button.setParentItem(clip)
                button.setPos(width / 2 - 8.0, (clip._clip_height - 16.0) / 2)
                self._keep(button)
            return
        first = bar_index(self._bar_lines, start, self._bar_fallback)
        index = 0 if first < 0 else first
        cuts: list[float] = []
        spans: list[tuple] = []
        for _step in range(4000):
            left_t, right_t = bar_span(self._bar_lines, index, self._bar_fallback)
            if left_t >= end - 1e-4:
                break
            if right_t > start + 1e-4:
                local_left = max(0.0, (left_t - start) * pps)
                local_right = min(width, (right_t - start) * pps)
                if local_right - local_left >= 2:
                    if local_left > 1:
                        cuts.append(local_left)
                    local_t0 = max(0.0, left_t - start)
                    local_t1 = min(duration, right_t - start)
                    spans.append((index, local_left, local_right, local_t0, local_t1))
            index += 1
        clip.set_block_spans(cuts, spans)
        size = 16.0
        for bar, left, right, _t0, _t1 in spans:
            inset_left = left + (3 if left > 1 else 0)
            inset_right = right - 3
            if inset_right - inset_left < 22:
                continue
            button = BarPlayButton(self, bar)
            button.setParentItem(clip)
            button.setPos(
                (inset_left + inset_right) / 2 - size / 2,
                (clip._clip_height - size) / 2,
            )
            self._keep(button)

    def _drag_block(self, clip: ClipItem, span, scene_pos: QPointF, origin: QPointF) -> None:
        """Slide one bar block across the canvas. The rest of the clip stays."""
        if self._block_drag is None:
            left = float(clip.pos().x()) + float(span[1])
            tracks = self._args[0] if self._args else []
            rows, primaries = self.lane_rows(tracks)
            source_row = rows[clip.clip_index] if clip.clip_index < len(rows) else 0
            source_top = self._display_top(source_row, primaries) if primaries else float(RULER_HEIGHT)
            hidden = []
            for child in clip.childItems():
                if isinstance(child, BarPlayButton) and span[1] <= child.pos().x() + 8 <= span[2]:
                    child.setVisible(False)
                    hidden.append(child)
            clip._hidden_span = (float(span[1]), float(span[2]))
            clip.update()
            ghost = QGraphicsRectItem(0, 0, max(8.0, float(span[2] - span[1])), clip.rect().height())
            ghost.setBrush(QBrush(clip.brush()))
            ghost.setPen(QPen(clip._accent, 1))
            ghost.setZValue(12)
            ghost.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene_obj.addItem(self._keep(ghost))
            self._block_ghost = ghost
            self._block_drag = {
                "clip": clip,
                "span": span,
                "grab": float(origin.x()) - left,
                "inset": float(clip.pos().y()) - source_top,
                "source_row": source_row,
                "hidden": hidden,
                "moved": False,
            }
        drag = self._block_drag
        drag["moved"] = True
        width = float(span[2] - span[1])
        ghost = self._block_ghost
        peer_pos = self._peer_scene_pos(scene_pos)
        if self.peer is not None:
            self.peer.show_drop_ghost(
                None if peer_pos is None else QPointF(peer_pos.x() - drag["grab"], peer_pos.y()),
                width,
                clip.rect().height(),
                clip,
            )
        if ghost is not None and isValid(ghost):
            ghost.setVisible(peer_pos is None)
        if peer_pos is not None:
            return
        left = clip._arrangement_x(float(scene_pos.x()) - drag["grab"])
        row, _primary = self._row_under(float(scene_pos.y()))
        top = self._row_top_of(row)
        if ghost is not None and isValid(ghost):
            ghost.setPos(left, top + drag["inset"])
            ghost.setRect(0, 0, width, clip.rect().height())

    def _row_under(self, scene_y: float) -> tuple[int, int]:
        tracks = self._args[0] if self._args else []
        _rows, primaries = self.lane_rows(tracks)
        if not primaries:
            return 0, 0
        best = 0
        best_distance = 1e9
        for display, primary in enumerate(primaries):
            top = self._display_top(display, primaries)
            height = self.row_height(primary)
            if top <= scene_y < top + height:
                return display, primary
            distance = abs(scene_y - (top + height / 2))
            if distance < best_distance:
                best = display
                best_distance = distance
        below = self._display_top(len(primaries), primaries)
        if scene_y >= below - 8:
            return len(primaries), -1
        return best, primaries[best]

    def _row_top_of(self, display_row: int) -> float:
        tracks = self._args[0] if self._args else []
        _rows, primaries = self.lane_rows(tracks)
        if display_row >= len(primaries):
            return self._display_top(len(primaries), primaries)
        return self._display_top(display_row, primaries)

    def cancel_block_drag(self) -> bool:
        """Esc during a block drag: drop nothing, put the block back."""
        drag = self._block_drag
        if drag is None:
            return False
        clip = drag["clip"]
        if self.peer is not None:
            self.peer.show_drop_ghost(None, 0, 0, clip)
        self._clear_block_drag(restore=True)
        if isValid(clip):
            clip._block_press = None
            clip._block_dragging = False
            clip._group = False
            clip.setCursor(Qt.CursorShape.OpenHandCursor)
            if clip.scene() is not None and clip.scene().mouseGrabberItem() is clip:
                clip.ungrabMouse()
        return True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.cancel_block_drag():
            event.accept()
            return
        super().keyPressEvent(event)

    def _clear_block_drag(self, restore: bool) -> None:
        drag = self._block_drag
        ghost = self._block_ghost
        self._block_drag = None
        self._block_ghost = None
        if ghost is not None and isValid(ghost):
            self.scene_obj.removeItem(ghost)
            try:
                self._retained.remove(ghost)
            except ValueError:
                pass
        if not restore or drag is None:
            return
        clip = drag["clip"]
        if isValid(clip):
            clip._hidden_span = None
            clip.update()
        for button in drag["hidden"]:
            if isValid(button):
                button.setVisible(True)

    def _drop_block(self, clip: ClipItem, span, scene_pos: QPointF, origin: QPointF) -> None:
        if not self.over_canvas(scene_pos):
            # Dropped on neither canvas: nothing moves, the block comes back.
            DRAG_LOG.info("canvas %s: block dropped outside the canvases, cancelled", self.canvas)
            if self._block_drag is not None:
                self.cancel_block_drag()
            return
        if self._block_drag is None:
            self._drag_block(clip, span, scene_pos, origin)
        drag = self._block_drag
        ghost = self._block_ghost
        peer_pos = self._peer_scene_pos(scene_pos)
        if self.peer is not None:
            self.peer.show_drop_ghost(None, 0, 0, clip)
        if drag is not None and drag.get("moved") and peer_pos is not None:
            # Dropped on the other canvas: a new row there, at the drop time.
            peer = self.peer
            left = clip._arrangement_x(float(peer_pos.x()) - drag["grab"])
            new_start = max(0.0, (left - HEADER_WIDTH) / self.pixels_per_second)
            tracks = self._args[0] if self._args else []
            row, primary = peer._row_under(float(peer_pos.y()))
            _rows, primaries = peer.lane_rows(tracks)
            join = 0 <= row < len(primaries) and primary >= 0  # onto a row there: join it
            insert_at = primaries[row] if join else len(tracks)
            local_start = float(span[3])
            local_duration = max(0.0, float(span[4]) - local_start)
            bar = int(span[0])
            self._clear_block_drag(restore=True)
            callback = self.on_block_move
            if callback is not None:
                payload = (clip.clip_index, local_start, local_duration, new_start, False, insert_at, bar, peer.canvas, join)
                QTimer.singleShot(0, lambda payload=payload, callback=callback: callback(*payload))
            return
        if drag is None or not drag.get("moved") or ghost is None or not isValid(ghost):
            self._clear_block_drag(restore=True)
            return
        left = float(ghost.pos().x())
        new_start = max(0.0, (left - HEADER_WIDTH) / self.pixels_per_second)
        row, primary = self._row_under(float(ghost.pos().y()) + float(clip.rect().height()) / 2)
        share = row == drag["source_row"]
        tracks = self._args[0] if self._args else []
        _rows, primaries = self.lane_rows(tracks)
        join = False
        if share:
            insert_at = clip.clip_index + 1
        elif primary < 0 or row >= len(primaries):
            insert_at = len(tracks)  # below the last row: a row of its own
        else:
            insert_at = primaries[row]
            join = True  # onto another track's row: it joins that row
        local_start = float(span[3])
        local_duration = max(0.0, float(span[4]) - local_start)
        bar = int(span[0])
        self._clear_block_drag(restore=True)
        callback = self.on_block_move
        if callback is None:
            return
        payload = (clip.clip_index, local_start, local_duration, new_start, share, insert_at, bar, None, join)
        QTimer.singleShot(0, lambda payload=payload, callback=callback: callback(*payload))

    def over_canvas(self, scene_pos: QPointF) -> bool:
        """True when ``scene_pos`` is level with this canvas or the other one
        (past their left or right edge still counts: the timeline goes on).
        A drop above or below both, on the toolbar, mixer or status bar, is
        cancelled."""
        point = self.viewport().mapToGlobal(self.mapFromScene(scene_pos))
        for view in (self, self.peer):
            if view is None or not view.isVisible():
                continue
            pane = view.parentWidget() or view
            top = pane.mapToGlobal(pane.rect().topLeft()).y()
            if top <= point.y() <= top + pane.height():
                return True
        return False

    def _peer_scene_pos(self, scene_pos: QPointF) -> QPointF | None:
        """``scene_pos`` (this view's scene) in the other canvas's scene, when
        the pointer is over that canvas; else None."""
        peer = self.peer
        if peer is None or not peer.isVisible():
            DRAG_LOG.info("canvas %s: no other canvas shown", self.canvas)
            return None
        point = self.viewport().mapToGlobal(self.mapFromScene(scene_pos))
        local = peer.viewport().mapFromGlobal(point)
        area = peer.viewport().rect()
        if not area.contains(local):
            # Anywhere over the other canvas counts, its track names included:
            # the drop lands at the nearest point of its timeline.
            pane = peer.parentWidget() or peer
            if not pane.rect().contains(pane.mapFromGlobal(point)):
                DRAG_LOG.info(
                    "canvas %s: pointer %s is not over canvas %s (its area %s at %s)",
                    self.canvas, point.toTuple(), peer.canvas, pane.rect().getRect(),
                    pane.mapToGlobal(QPoint(0, 0)).toTuple(),
                )
                return None
            local = QPoint(
                min(max(local.x(), area.left()), area.right()),
                min(max(local.y(), area.top()), area.bottom()),
            )
        return peer.mapToScene(local)

    def show_drop_ghost(self, left_top: QPointF | None, width: float, height: float, clip: ClipItem | None) -> None:
        """Outline where a bar block from the other canvas would land."""
        ghost = self._block_ghost
        if left_top is None:
            if ghost is not None and isValid(ghost) and self._block_drag is None:
                self.scene_obj.removeItem(ghost)
                try:
                    self._retained.remove(ghost)
                except ValueError:
                    pass
            if self._block_drag is None:
                self._block_ghost = None
            return
        if ghost is None or not isValid(ghost):
            ghost = QGraphicsRectItem(0, 0, 8, 8)
            ghost.setZValue(12)
            ghost.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            if clip is not None:
                ghost.setBrush(QBrush(clip.brush()))
                ghost.setPen(QPen(clip._accent, 1, Qt.PenStyle.DashLine))
            self.scene_obj.addItem(self._keep(ghost))
            self._block_ghost = ghost
        left = HEADER_WIDTH if self._args is None else max(float(HEADER_WIDTH), float(left_top.x()))
        if clip is not None:
            left = clip._arrangement_x(left)
        row, _primary = self._row_under(float(left_top.y()))
        ghost.setPos(left, self._row_top_of(row) + 4)
        ghost.setRect(0, 0, max(8.0, float(width)), max(8.0, float(height)))

    # -- picked bars ---------------------------------------------------

    def _clip_items(self) -> list:
        return [item for item in self.scene_obj.items() if isinstance(item, ClipItem) and isValid(item)]

    @staticmethod
    def _spans_of(clip: ClipItem) -> tuple:
        """A clip's bar blocks; a clip without bar lines is one block."""
        if clip._block_spans:
            return clip._block_spans
        return ((0, 0.0, float(clip.rect().width()), 0.0, float(clip._duration)),)

    def picks(self) -> dict[int, set[int]]:
        """Picked bars: track index -> bar numbers."""
        return {index: set(bars) for index, bars in self._picks.items() if bars}

    def set_picks(self, picks: dict) -> None:
        self._picks = {int(index): set(bars) for index, bars in picks.items() if bars}
        self._apply_picks()

    def pick_whole_clips(self, indices) -> None:
        """Pick every bar of these clips (used after a move, so the moved
        bars stay picked and can be dragged on together)."""
        picks = {}
        for clip in self._clip_items():
            if clip.clip_index in indices:
                picks[clip.clip_index] = {int(span[0]) for span in self._spans_of(clip)}
        self.set_picks(picks)

    def _apply_picks(self) -> None:
        for clip in self._clip_items():
            wanted = set(self._picks.get(clip.clip_index, set()))
            if not clip._block_spans and wanted:
                clip._block_spans = self._spans_of(clip)
            if clip.picked != wanted:
                clip.picked = wanted
                clip.update()

    def picked_blocks(self) -> list[tuple]:
        """[(track index, local start, local duration, bar, clip, span)]"""
        found = []
        for clip in self._clip_items():
            for span in self._spans_of(clip):
                if span[0] in clip.picked:
                    found.append((clip.clip_index, float(span[3]), float(span[4]) - float(span[3]), int(span[0]), clip, span))
        found.sort(key=lambda item: (item[0], item[1]))
        return found

    def show_ghosts(self, rects) -> None:
        """Outline where dragged items would land: [(QRectF, QColor)]."""
        for ghost in self._ghosts:
            if isValid(ghost):
                self.scene_obj.removeItem(ghost)
            try:
                self._retained.remove(ghost)
            except ValueError:
                pass
        self._ghosts = []
        for rect, colour in rects:
            ghost = QGraphicsRectItem(rect)
            ghost.setBrush(QBrush(QColor(colour.red(), colour.green(), colour.blue(), 70)))
            ghost.setPen(QPen(QColor("#D7B36A"), 1.2, Qt.PenStyle.DashLine))
            ghost.setZValue(25)
            ghost.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene_obj.addItem(self._keep(ghost))
            self._ghosts.append(ghost)

    def _landing(self, view, grabbed_left: float, grabbed_row_offset: int, pointer_y: float, items):
        """Where ``items`` [(left, width, height, source row)] land in ``view``
        when the grabbed one's left edge goes to ``grabbed_left`` and its row
        is the one under ``pointer_y``. Returns (target row, rects)."""
        target_row, _primary = view._row_under(pointer_y)
        tracks = view._args[0] if view._args else []
        _rows, primaries = view.lane_rows(tracks)
        rects = []
        for left, width, height, row in items:
            wanted = max(0, target_row + row - grabbed_row_offset)
            if wanted < len(primaries):
                top = view._row_top_of(wanted)
            else:  # new rows stack below the last one
                top = view._display_top(len(primaries), primaries) + (wanted - len(primaries)) * view.track_height
            rects.append(QRectF(grabbed_left + (left - items[0][0]), top + 4, width, height))
        return target_row, rects

    def _drag_group(self, clip, span, scene_pos, origin) -> None:
        """Dragging the picked bars: outlines follow the pointer, on this
        canvas or the other one. ``clip`` None ends the selection."""
        if clip is None:
            if self._picks:
                self.set_picks({})
            return
        tracks = self._args[0] if self._args else []
        rows, _ = self.lane_rows(tracks)
        if self._group_drag is None:
            blocks = self.picked_blocks()
            grabbed = next(
                (n for n, item in enumerate(blocks) if item[0] == clip.clip_index and item[3] == int(span[0])), 0
            )
            ordered = [blocks[grabbed]] + [item for n, item in enumerate(blocks) if n != grabbed]
            self._group_drag = {
                "blocks": blocks,
                "grabbed": grabbed,
                "items": [
                    (item[4].pos().x() + float(item[5][1]), float(item[5][2]) - float(item[5][1]), item[4].rect().height(),
                     rows[item[0]] if item[0] < len(rows) else 0, item[4]._accent)
                    for item in ordered
                ],
                "grab": float(origin.x()) - (clip.pos().x() + float(span[1])),
            }
        drag = self._group_drag
        items = [(left, width, height, row) for left, width, height, row, _colour in drag["items"]]
        colours = [colour for *_rest, colour in drag["items"]]
        peer_pos = self._peer_scene_pos(scene_pos)
        view, point = (self.peer, peer_pos) if peer_pos is not None else (self, scene_pos)
        left = clip._arrangement_x(float(point.x()) - drag["grab"])
        target_row, rects = self._landing(view, left, items[0][3], float(point.y()), items)
        drag["target"] = (view.canvas, target_row, (left - items[0][0]) / self.pixels_per_second)
        view.show_ghosts(list(zip(rects, colours)))
        other = self.peer if view is self else self
        if other is not None:
            other.show_ghosts([])

    def _drop_group(self, clip, span, scene_pos, origin) -> None:
        self._drag_group(clip, span, scene_pos, origin)
        drag, self._group_drag = self._group_drag, None
        self.show_ghosts([])
        if self.peer is not None:
            self.peer.show_ghosts([])
        if drag is None or "target" not in drag or not self.over_canvas(scene_pos):
            return
        canvas, target_row, shift = drag["target"]
        blocks = [(item[0], item[1], item[2], item[3]) for item in drag["blocks"]]
        DRAG_LOG.info("canvas %s: %s picked bars dropped on canvas %s row %s, shift %.3f s",
                      self.canvas, len(blocks), canvas, target_row, shift)
        grabbed = drag["grabbed"]
        QTimer.singleShot(
            0, lambda: self.blocks_move_requested.emit(blocks, grabbed, shift, canvas, target_row)
        )

    def _preview_on_peer(self, clip, scene_pos) -> None:
        """While whole clips are dragged over the other canvas, outline all
        of them there (the clips themselves are cut off at this canvas's edge)."""
        if self.peer is None:
            return
        peer_pos = self._peer_scene_pos(scene_pos) if scene_pos is not None else None
        dragged = [item for item in self._clip_items() if item.isSelected() or item is clip]
        for item in dragged:
            # Faded while over the other canvas: they are leaving this one.
            item.setOpacity(0.35 if peer_pos is not None else 1.0)
        if peer_pos is None:
            self.peer.show_ghosts([])
            return
        tracks = self._args[0] if self._args else []
        rows, _ = self.lane_rows(tracks)
        selected = [item for item in self._clip_items() if item.isSelected() or item is clip]
        selected.sort(key=lambda item: (item is not clip, item.clip_index))
        items = [(item.pos().x(), item.rect().width(), item.rect().height(),
                  rows[item.clip_index] if item.clip_index < len(rows) else 0) for item in selected]
        grab = float(scene_pos.x()) - float(clip.pos().x())
        left = clip._arrangement_x(float(peer_pos.x()) - grab)
        _row, rects = self._landing(self.peer, left, items[0][3], float(peer_pos.y()), items)
        self.peer.show_ghosts(list(zip(rects, [item._accent for item in selected])))

    def _snap_to_row(self, clip: ClipItem, y: float) -> float:
        """y for a dragged clip: the same inset in the row under its middle."""
        tracks = self._args[0] if self._args else []
        rows, primaries = self.lane_rows(tracks)
        if not primaries or clip.clip_index >= len(rows) or rows[clip.clip_index] < 0:
            return clip._locked_y
        inset = clip._locked_y - self._display_top(rows[clip.clip_index], primaries)
        row, _primary = self._row_under(float(y) + clip.rect().height() / 2)
        return self._row_top_of(row) + inset

    def _drop_clip_on_row(self, clip: ClipItem) -> bool:
        """A dragged clip released over another row moves to that row."""
        tracks = self._args[0] if self._args else []
        rows, _primaries = self.lane_rows(tracks)
        if clip.clip_index >= len(rows) or rows[clip.clip_index] < 0:
            return False
        row, _primary = self._row_under(float(clip.pos().y()) + clip.rect().height() / 2)
        if row == rows[clip.clip_index]:
            return False
        index = clip.clip_index
        QTimer.singleShot(0, lambda: self.row_move_requested.emit(index, row))
        return True

    def _drop_clip_on_peer(self, clip: ClipItem, scene_pos: QPointF, press: QPointF) -> bool:
        """Clips released over the other canvas move there: the grabbed one
        and every other selected clip on this canvas, keeping their spacing."""
        peer_pos = self._peer_scene_pos(scene_pos)
        if peer_pos is None:
            return False
        tracks = self._args[0] if self._args else []
        grab = float(scene_pos.x()) - float(clip.pos().x())
        left = clip._arrangement_x(float(peer_pos.x()) - grab)
        start = max(0.0, (left - HEADER_WIDTH) / self.pixels_per_second)
        current = float(tracks[clip.clip_index].get("start", 0.0) or 0.0) if clip.clip_index < len(tracks) else 0.0
        indices = sorted({clip.clip_index, *self.selected_indices()})
        row, _primary = self.peer._row_under(float(peer_pos.y()))
        grabbed = clip.clip_index
        DRAG_LOG.info("canvas %s: clips %s dropped on canvas %s row %s, shift %.3f s",
                      self.canvas, indices, self.peer.canvas, row, start - current)
        QTimer.singleShot(
            0, lambda: self.canvas_drop_requested.emit(indices, grabbed, start - current, row)
        )
        return True

    def set_loop_region(self, start, end) -> None:
        """Shade an arrangement span. Seconds come from the caller."""
        if start is None or end is None or float(end) <= float(start):
            self._loop_region = None
        else:
            self._loop_region = (float(start), float(end))
        self._paint_loop_region()

    def _paint_loop_region(self) -> None:
        previous = self._loop_region_item
        self._loop_region_item = None
        if previous is not None:
            try:
                if isValid(previous):
                    self.scene_obj.removeItem(previous)
            except RuntimeError:
                pass
            try:
                self._retained.remove(previous)
            except ValueError:
                pass
        region = self._loop_region
        if region is None:
            return
        start, end = region
        x = HEADER_WIDTH + start * self.pixels_per_second
        width = max(1.0, (end - start) * self.pixels_per_second)
        height = max(500, int(self.scene_obj.sceneRect().height()))
        band = self._keep(
            self.scene_obj.addRect(
                x,
                0,
                width,
                height,
                QPen(QColor(215, 179, 106, 210), 1.5),
                QBrush(QColor(215, 179, 106, 42)),
            )
        )
        band.setZValue(8)
        band.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._loop_region_item = band

    @staticmethod
    def _clip_box(row_h: int) -> tuple[int, int]:
        # The default lane keeps the original clip box so snap tests stay put.
        if row_h <= TRACK_HEIGHT:
            return CLIP_HEIGHT, CLIP_TOP
        return int(row_h - 16), 8

    def set_playhead(self, seconds, follow=False):
        """Move the indicator. The canvas stays where it is.

        ``follow=True`` also centres the view on it at once; playback
        scrolling goes through ``follow_playhead`` instead.
        """
        height = max(500, int(self.scene_obj.sceneRect().height()))
        x = HEADER_WIDTH + float(seconds) * self.pixels_per_second
        playhead = self._live_playhead(height)
        try:
            playhead.set_height(height)
            if abs(playhead.x() - x) >= 0.05:
                playhead.setPos(x, 0)
        except RuntimeError:
            self.playhead = None
            self._create_playhead(height).setPos(x, 0)
        if follow:
            self._center_playhead(x)

    def set_playhead_visible(self, visible: bool) -> None:
        """Hide the transport indicator (a stem block plays on its own) or
        show it again. Its position is not touched."""
        self._playhead_shown = bool(visible)
        playhead = self.playhead
        if playhead is not None and isValid(playhead) and playhead.isVisible() != bool(visible):
            playhead.setVisible(bool(visible))

    def _clip_for(self, index: int):
        clip = self._clip_cache.get(index)
        if clip is None or not isValid(clip):
            self._clip_cache = {item.clip_index: item for item in self._clip_items()}
            clip = self._clip_cache.get(index)
        return clip

    def block_times(self, index: int, bar: int) -> tuple[float, float] | None:
        """Arrangement start and end of one bar block of track ``index``."""
        clip = self._clip_for(index)
        if clip is None:
            return None
        start = (float(clip.pos().x()) - HEADER_WIDTH) / self.pixels_per_second
        for span in self._spans_of(clip):
            if span[0] == bar and span[4] > span[3]:
                return start + float(span[3]), start + float(span[4])
        return None

    def block_at(self, view_pos) -> tuple[int, int] | None:
        """``(track, bar)`` of the block under a viewport position."""
        clip = self.clip_at(view_pos)
        if clip is None:
            return None
        local = clip.mapFromScene(self.mapToScene(view_pos)).x()
        for span in self._spans_of(clip):
            if span[1] - 1 <= local <= span[2] + 1:
                return clip.clip_index, int(span[0])
        return None

    def set_previews(self, states: dict) -> None:
        """Show stem block voices: ``(track, bar) -> (playing, seconds)``.

        Each block's ▶ becomes ⏸ while it plays; the block is tinted and a
        cursor marks where it is. Only clips whose state changed repaint.
        """
        by_clip: dict[int, dict] = {}
        pps = self.pixels_per_second
        for (index, bar), (playing, seconds) in states.items():
            clip = self._clip_for(index)
            if clip is None:
                continue
            local = HEADER_WIDTH + float(seconds) * pps - float(clip.pos().x())
            by_clip.setdefault(index, {})[int(bar)] = (bool(playing), round(local, 1))
        for index in self._previewed | set(by_clip):
            clip = self._clip_for(index)
            if clip is None:
                continue
            marks = by_clip.get(index, {})
            clip.set_preview(marks)
            for child in clip.childItems():
                if isinstance(child, BarPlayButton):
                    mark = marks.get(child.index)
                    child.set_playing(mark is not None and mark[0])
        self._previewed = set(by_clip)

    def playhead_x(self) -> float | None:
        playhead = self.playhead
        if playhead is None or not isValid(playhead):
            return None
        return float(playhead.x())

    def _note_user_scroll(self) -> None:
        self._user_scrolled_at = time.monotonic()

    def follow_playhead(self, mode: str, dt: float) -> None:
        """Scroll for one animation frame of ``dt`` seconds, per ``mode``.

        Off: never scroll. Follow: once the playhead reaches the right edge
        the canvas glides with it, holding it there; if it lands off screen
        (a seek, a loop restart) the canvas eases over to it. Center: the
        canvas moves under a playhead held in the middle, and eases rather
        than jumps when the playhead does. A manual scroll pauses following
        for a moment so it does not fight the user.
        """
        x = self.playhead_x()
        width = self.viewport().width()
        if mode not in (FOLLOW_PAGE, FOLLOW_CENTER) or x is None or width < 2 or self._scrub:
            return
        if time.monotonic() - self._user_scrolled_at < USER_SCROLL_HOLD:
            return
        bar = self.horizontalScrollBar()
        if bar.isSliderDown():
            return
        left = self.mapToScene(0, 0).x()
        margin = min(width * 0.25, max(40.0, width * 0.12))
        if mode == FOLLOW_CENTER:
            target = x - width / 2.0
        elif x > left + width - margin:
            target = x - (width - margin)
        elif x < left:
            target = x - margin
        else:
            return
        error = target - left
        if abs(error) < 0.5:
            return
        if abs(error) > max(48.0, width * 0.1):
            # The playhead jumped: ease over instead of snapping the canvas.
            error *= 1.0 - math.exp(-max(0.0, dt) / FOLLOW_EASE_SECONDS)
        value = int(round(bar.value() + error))
        bar.setValue(max(bar.minimum(), min(bar.maximum(), value)))

    def _center_playhead(self, x: float) -> None:
        """Scroll so the indicator sits on the horizontal middle of the view."""
        width = self.viewport().width()
        if width < 2:
            return
        bar = self.horizontalScrollBar()
        target = int(round(x - width / 2.0))
        target = max(bar.minimum(), min(bar.maximum(), target))
        if abs(bar.value() - target) < 2:
            return
        bar.setValue(target)

    def _on_selection(self):
        if self._rebuilding:
            return
        try:
            selected = [
                item
                for item in self.scene_obj.selectedItems()
                if isinstance(item, ClipItem) and isValid(item)
            ]
        except RuntimeError:
            return
        self.track_selected.emit(selected[0].clip_index if selected else -1)

    def set_show_beats(self, enabled: bool) -> None:
        self.show_beats = bool(enabled)
        for item in self.scene_obj.items():
            if isinstance(item, ClipItem):
                item._show_beats = self.show_beats
                item.update()

    def set_show_detected(self, enabled: bool) -> None:
        self.show_detected = bool(enabled)
        for item in self.scene_obj.items():
            if isinstance(item, ClipItem):
                item._show_detected = self.show_detected
                item.update()

    def set_show_onsets(self, enabled: bool) -> None:
        self.show_onsets = bool(enabled)
        for item in self.scene_obj.items():
            if isinstance(item, ClipItem):
                item._show_onsets = self.show_onsets
                item.update()

    @staticmethod
    def _beat_track(track, audio) -> tuple[tuple[float, ...], float]:
        cached = track.get("_beats")
        if cached is not None:
            return tuple(cached), float(track.get("_tempo") or 0.0)
        result = analyze_beats(audio, OUTPUT_RATE)
        track["_beats"] = result.beats
        track["_tempo"] = result.tempo
        return result.beats, result.tempo

    def set_snap(self, enabled: bool) -> None:
        self.snap_to_grid = bool(enabled)
        for item in self.scene_obj.items():
            if isinstance(item, ClipItem):
                item._grid_snap = self.snap_to_grid

    def selected_indices(self) -> list[int]:
        found = []
        for item in self.scene_obj.selectedItems():
            if isinstance(item, ClipItem) and isValid(item):
                found.append(item.clip_index)
        return sorted(set(found))

    def select_index(self, index: int):
        self.scene_obj.clearSelection()
        for item in self.scene_obj.items():
            if isinstance(item, ClipItem) and item.clip_index == index:
                item.setSelected(True)
                return
        self.track_selected.emit(index)

    def clip_at(self, view_pos):
        for item in self.items(view_pos):
            current = item
            while current is not None:
                if isinstance(current, ClipItem):
                    return current
                current = current.parentItem()
        return None

    def bar_line_at(self, view_pos) -> int:
        """Index of the bar line under ``view_pos`` on the ruler, else -1."""
        pos = self.mapToScene(view_pos)
        if pos.y() > RULER_HEIGHT or not self._bar_lines:
            return -1
        seconds = (pos.x() - HEADER_WIDTH) / self.pixels_per_second
        index = min(range(len(self._bar_lines)), key=lambda i: abs(self._bar_lines[i] - seconds))
        if abs(self._bar_lines[index] - seconds) * self.pixels_per_second > 5:
            return -1
        return index

    def bar_under(self, view_pos) -> int:
        """Zero-based bar under a viewport position, or -1 before the first line."""
        if not self._bar_lines:
            return -1
        pos = self.mapToScene(view_pos)
        if pos.x() < HEADER_WIDTH:
            return -1
        seconds = max(0.0, (pos.x() - HEADER_WIDTH) / self.pixels_per_second)
        return bar_index(self._bar_lines, seconds, self._bar_fallback)

    def _downbeat_pressed(self, clip, local: float, screen) -> None:
        start = max(0.0, (float(clip.pos().x()) - HEADER_WIDTH) / self.pixels_per_second)
        index = bar_index(self._bar_lines, start + float(local), self._bar_fallback) if self._bar_lines else -1
        point = screen.toPoint() if hasattr(screen, "toPoint") else screen
        self.downbeat_clicked.emit(int(index), point)

    def bar_tooltip(self, index: int) -> str:
        start, end = bar_span(self._bar_lines, index, self._bar_fallback)
        return f"Bar {index + 1}\nStart {format_clock(start)}\nEnd   {format_clock(end)}\nLength {end - start:.3f} s"

    def _drag_base(self, index: int) -> list[float]:
        # Lines past the caller's map are repeats; only keep those up to the
        # dragged one. Without a map every tempo line becomes a real line.
        if self._bar_map:
            return list(self._bar_lines[: max(len(self._bar_map), index + 1)])
        return list(self._bar_lines)

    def on_playhead_handle(self, view_pos) -> bool:
        """True over the playhead's handle at the top of the ruler."""
        x = self.playhead_x()
        if x is None:
            return False
        pos = self.mapToScene(view_pos)
        return pos.y() <= PlayheadItem.HANDLE_HEIGHT + 2 and abs(pos.x() - x) <= PlayheadItem.HALF_WIDTH + 2

    def _scrub_to(self, view_pos) -> None:
        x = self.mapToScene(view_pos).x()
        seconds = max(0.0, (x - HEADER_WIDTH) / self.pixels_per_second)
        self.set_playhead(seconds)  # at once, not on the next frame
        self.seek_requested.emit(seconds)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and self.on_playhead_handle(event.position().toPoint())
        ):
            # The handle drags the playhead, even over a bar line.
            self._scrub = True
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and pos.y() <= RULER_HEIGHT and self._bar_lines:
            seconds = max(0.0, (pos.x() - HEADER_WIDTH) / self.pixels_per_second)
            bar = max(0, bar_index(self._bar_lines, seconds, self._bar_fallback))
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Shift-click selects whole bars from the last bar clicked.
                self.bars_selected.emit(min(self._bar_anchor, bar), max(self._bar_anchor, bar))
                event.accept()
                return
            self._bar_anchor = bar
        if event.button() == Qt.MouseButton.LeftButton:
            line = self.bar_line_at(event.position().toPoint())
            if line >= 0:
                x = HEADER_WIDTH + self._bar_lines[line] * self.pixels_per_second
                height = self.scene_obj.sceneRect().height()
                guide = self._keep(self.scene_obj.addLine(x, 0, x, height, QPen(QColor("#D7B36A"), 1)))
                guide.setZValue(20)
                self._bar_drag = (line, guide)
                self._bar_press_x = float(event.position().x())
                event.accept()
                return
        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle-drag pans the canvas.
            self._canvas_pan = (
                QPointF(event.position()),
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self._canvas_panned = False
            event.accept()
            return
        self._empty_press = None
        if event.button() == Qt.MouseButton.LeftButton:
            view_pos = event.position().toPoint()
            if self.clip_at(view_pos) is None:
                pos = self.mapToScene(view_pos)
                if pos.x() >= HEADER_WIDTH and pos.y() > RULER_HEIGHT:
                    # Left-drag on empty space draws a selection box; a click
                    # there moves the playhead. The box picks bars; with
                    # Shift it selects whole tracks (the view's rubber band);
                    # with Ctrl it adds bars to the ones already picked.
                    self._empty_press = QPointF(event.position())
                    if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        add = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                        if not add:
                            self.set_picks({})
                            self.scene_obj.clearSelection()
                        band = QGraphicsRectItem(QRectF(pos, pos))
                        band.setPen(QPen(QColor("#D7B36A"), 1, Qt.PenStyle.DashLine))
                        band.setBrush(QBrush(QColor(215, 179, 106, 30)))
                        band.setZValue(30)
                        band.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                        self.scene_obj.addItem(self._keep(band))
                        self._band = (pos, band, add)
                        event.accept()
                        return
                elif pos.x() >= HEADER_WIDTH:
                    # A press on the ruler moves the playhead; dragging scrubs.
                    self._scrub = True
                    self._scrub_to(view_pos)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _drag_seconds(self, event) -> float:
        # Bar lines follow the audio, so they do not snap to the grid.
        x = self.mapToScene(event.position().toPoint()).x()
        return max(0.0, (x - HEADER_WIDTH) / self.pixels_per_second)

    def mouseMoveEvent(self, event):
        if self._scrub:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._scrub_to(event.position().toPoint())
            event.accept()
            return
        if self._band is not None:
            origin, band, _add = self._band
            if isValid(band):
                band.setRect(QRectF(origin, self.mapToScene(event.position().toPoint())).normalized())
            event.accept()
            return
        if self._canvas_pan is not None and event.buttons() & Qt.MouseButton.MiddleButton:
            origin, scroll_x, scroll_y = self._canvas_pan
            delta = event.position() - origin
            if abs(delta.x()) + abs(delta.y()) >= 4:
                self._canvas_panned = True
                self._note_user_scroll()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            self.horizontalScrollBar().setValue(int(scroll_x - delta.x()))
            self.verticalScrollBar().setValue(int(scroll_y - delta.y()))
            event.accept()
            return
        if self._bar_drag is not None:
            index, guide = self._bar_drag
            seconds = self._drag_seconds(event)
            x = HEADER_WIDTH + seconds * self.pixels_per_second
            line = guide.line()
            guide.setLine(x, line.y1(), x, line.y2())
            start = self._bar_lines[index - 1] if index > 0 else None
            text = f"Bar {index + 1} starts {format_clock(seconds)}"
            if start is not None:
                text += f"\nBar {index} length {seconds - start:.3f} s"
            QToolTip.showText(event.globalPosition().toPoint(), text, self.viewport())
            event.accept()
            return
        if not event.buttons():
            line = self.bar_line_at(event.position().toPoint())
            if self.on_playhead_handle(event.position().toPoint()):
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            elif line >= 0:
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                QToolTip.showText(event.globalPosition().toPoint(), self.bar_tooltip(line), self.viewport())
            else:
                self.viewport().unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._scrub and event.button() == Qt.MouseButton.LeftButton:
            self._scrub = False
            event.accept()
            return
        if self._canvas_pan is not None and event.button() == Qt.MouseButton.MiddleButton:
            self._canvas_pan = None
            self._canvas_panned = False
            self.viewport().unsetCursor()
            event.accept()
            return
        if self._band is not None and event.button() == Qt.MouseButton.LeftButton:
            origin, band, add = self._band
            self._band = None
            area = band.rect() if isValid(band) else QRectF()
            if isValid(band):
                self.scene_obj.removeItem(band)
                try:
                    self._retained.remove(band)
                except ValueError:
                    pass
            empty, self._empty_press = self._empty_press, None
            if empty is not None and (event.position() - empty).manhattanLength() < 4:
                self.seek_requested.emit(max(0.0, (origin.x() - HEADER_WIDTH) / self.pixels_per_second))
                event.accept()
                return
            picks = {index: set(bars) for index, bars in self._picks.items()} if add else {}
            for clip in self._clip_items():
                for span in self._spans_of(clip):
                    block = QRectF(clip.pos().x() + float(span[1]), clip.pos().y(), float(span[2]) - float(span[1]), clip.rect().height())
                    if block.intersects(area):
                        picks.setdefault(clip.clip_index, set()).add(int(span[0]))
            self.set_picks(picks)
            if picks:
                self.track_selected.emit(min(picks))
            event.accept()
            return
        empty, self._empty_press = getattr(self, "_empty_press", None), None
        if (
            empty is not None
            and event.button() == Qt.MouseButton.LeftButton
            and (event.position() - empty).manhattanLength() < 4
        ):
            pos = self.mapToScene(empty.toPoint())
            self.seek_requested.emit(max(0.0, (pos.x() - HEADER_WIDTH) / self.pixels_per_second))
        if self._bar_drag is not None and event.button() == Qt.MouseButton.LeftButton:
            index, guide = self._bar_drag
            self._bar_drag = None
            guide.setVisible(False)
            event.accept()
            seconds = self._drag_seconds(event)
            if abs(float(event.position().x()) - self._bar_press_x) < 3:
                # A click on a bar line seeks, like a click anywhere on the ruler.
                self.seek_requested.emit(max(0.0, self._bar_lines[index]))
                return
            ripple = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.bars_changed.emit(move_line(self._drag_base(index), index, seconds, ripple))
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and pos.y() <= RULER_HEIGHT and self._bar_lines:
            seconds = (pos.x() - HEADER_WIDTH) / self.pixels_per_second
            index = bar_index(self._bar_lines, seconds, self._bar_fallback)
            if index >= 0:
                self.bar_edit_requested.emit(index)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._rebuilding or self._in_resize or not self._args:
            return
        needed = max(
            500,
            self._row_top(len(self._args[0])) + TRACK_HEIGHT,
            self.viewport().height(),
        )
        if needed <= self.scene_obj.sceneRect().height() + 4:
            return
        self._in_resize = True
        try:
            self._rerender()
        finally:
            self._in_resize = False

    def scrollContentsBy(self, dx, dy):
        # The base implementation copies the viewport pixmap, then paints only
        # the newly exposed strip. Antialiased strokes sit outside their item
        # bounds, so the copy leaves a yellow line at every scrollbar step.
        # Keep the scroll, then repaint the whole view.
        super().scrollContentsBy(dx, dy)
        self.viewport().update()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.zoom_horizontal(1.1 if delta > 0 else 1 / 1.1)
                event.accept()
                return
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            delta = event.angleDelta().y()
            if delta:
                self.zoom_vertical(1.08 if delta > 0 else 1 / 1.08)
                event.accept()
                return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._note_user_scroll()
            self._scroll_horizontal(self._wheel_pixels(event))
            event.accept()
            return
        if event.angleDelta().x() or event.pixelDelta().x():
            self._note_user_scroll()
        super().wheelEvent(event)

    def _scroll_horizontal(self, pixels: int) -> None:
        if not pixels:
            return
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - pixels)

    @staticmethod
    def _wheel_pixels(event) -> int:
        pixel = event.pixelDelta()
        if pixel.x():
            delta = pixel.x()
        elif pixel.y():
            delta = pixel.y()
        else:
            angle = event.angleDelta()
            ticks = angle.x() or angle.y()
            delta = int(round(ticks / 120.0 * 80))
        if event.inverted():
            delta = -delta
        return delta
