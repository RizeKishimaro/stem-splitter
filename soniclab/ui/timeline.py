import bisect
import math

import numpy as np
from shiboken6 import isValid

from PySide6.QtCore import QPointF, Qt, Signal
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
HEADER_WIDTH = 0
PIXELS_PER_SECOND = 90.0
RULER_HEIGHT = 32
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

        self.setZValue(3)
        self.setToolTip(f"{name} · {duration:.2f}s")

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
        self._raster = None
        self._raster_key = None
        self._raster_bytes = b""
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
        self._raster = None
        self._raster_key = None

    def _draw_title(self, name):
        shown = name if len(name) <= 32 else name[:31] + "…"
        title = QGraphicsSimpleTextItem(shown, self)
        title.setBrush(QBrush(QColor("#E7F2EA")))
        title.setFont(QFont("DejaVu Sans", 8))
        title.setPos(8, max(2.0, self._clip_height - self._footer))
        title.setZValue(6)
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
        painter.restore()

    def _paint_wave(self, painter, left: float, right: float):
        image = self._ensure_raster()
        if image is not None:
            start = max(0, int(left) - 1)
            stop = min(image.width(), int(math.ceil(right)) + 2)
            if stop > start:
                painter.drawImage(start, 0, image, start, 0, stop - start, image.height())
            return
        self._fit_envelope()
        count = int(self._upper.size)
        if count == 0:
            return
        start = max(0, int(left) - 1)
        stop = min(count, int(right) + 2)
        if stop <= start:
            return
        raw, image = self._rasterize(start, stop)
        self._raster_bytes = raw
        painter.drawImage(start, 0, image)

    def _ensure_raster(self):
        """Reuse one image for clips that fit on a few screens."""
        self._fit_envelope()
        width = int(self._upper.size)
        height = max(1, int(self.rect().height()))
        if width == 0 or width > 8192:
            return None
        key = (width, height, self._show_beats, self._beats)
        if self._raster is not None and self._raster_key == key:
            return self._raster
        self._raster_bytes, self._raster = self._rasterize(0, width)
        self._raster_key = key
        return self._raster

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
        self.setCursor(
            Qt.CursorShape.SizeHorCursor if self._edge_at(event.pos().x()) else Qt.CursorShape.ArrowCursor
        )
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        edge = self._edge_at(event.pos().x()) if event.button() == Qt.MouseButton.LeftButton else None
        if edge:
            self._edge = edge
            self._report_moves = False
            self._trim_right = float(self.pos().x() + self.rect().width())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._edge:
            super().mouseMoveEvent(event)
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
            return type(value)(x, y)

        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._y_locked:
                self._locked_y = float(self.pos().y())
                self._y_locked = True
            seconds = max(
                0.0,
                (float(self.pos().x()) - HEADER_WIDTH) / self.pixels_per_second,
            )
            if self._report_moves and self.moved_callback:
                self.moved_callback(self.clip_index, seconds)

        return super().itemChange(change, value)


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

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.playhead: QGraphicsLineItem | None = None
        self.pixels_per_second = float(PIXELS_PER_SECOND)
        self.track_height = TRACK_HEIGHT
        self.snap_to_grid = True
        self.show_beats = True
        self.show_detected = True
        self.show_onsets = True
        self.on_trim = None
        self.on_release = None
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
        self._loop_region_item = None
        self._bar_handles = []
        self._bar_drag = None
        self._retained.clear()
        self.scene_obj.clear()

    def _create_playhead(self, height) -> QGraphicsLineItem:
        playhead = self._keep(
            self.scene_obj.addLine(
                HEADER_WIDTH,
                0,
                HEADER_WIDTH,
                height,
                QPen(QColor("#D7B36A"), 1.5),
            )
        )
        playhead.setZValue(20)
        playhead.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.playhead = playhead
        return playhead

    def _live_playhead(self, height) -> QGraphicsLineItem:
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
        content_bottom = self._row_top(len(tracks))
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

        for row, track in enumerate(tracks):
            row_h = self.row_height(row)
            y = self._row_top(row)
            lane = QColor("#12161B") if row % 2 == 0 else QColor("#161B22")
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
            beats, tempo = self._beat_track(track, audio)
            clip.set_beats(beats, offset_seconds, beats_per_bar, self.show_beats)
            if tempo > 0:
                clip.setToolTip(f"{clip.toolTip()} · {tempo:.1f} BPM detected")

        self._paint_loop_region()
        self.horizontalScrollBar().setValue(scroll_x)
        self.verticalScrollBar().setValue(scroll_y)
        self._rebuilding = False
        self.layout_changed.emit()

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
        height = max(500, int(self.scene_obj.sceneRect().height()))
        x = HEADER_WIDTH + seconds * self.pixels_per_second
        playhead = self._live_playhead(height)
        try:
            line = playhead.line()
            if abs(line.x1() - x) < 0.5 and abs(line.y2() - height) < 1:
                if follow:
                    self._center_playhead(x)
                return
            playhead.setLine(x, 0, x, height)
        except RuntimeError:
            self.playhead = None
            replacement = self._create_playhead(height)
            replacement.setLine(x, 0, x, height)
        if follow:
            self._center_playhead(x)

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

    def bar_tooltip(self, index: int) -> str:
        start, end = bar_span(self._bar_lines, index, self._bar_fallback)
        return f"Bar {index + 1}\nStart {format_clock(start)}\nEnd   {format_clock(end)}\nLength {end - start:.3f} s"

    def _drag_base(self, index: int) -> list[float]:
        # Lines past the caller's map are repeats; only keep those up to the
        # dragged one. Without a map every tempo line becomes a real line.
        if self._bar_map:
            return list(self._bar_lines[: max(len(self._bar_map), index + 1)])
        return list(self._bar_lines)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
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
        if event.button() == Qt.MouseButton.LeftButton:
            view_pos = event.position().toPoint()
            if self.clip_at(view_pos) is None:
                pos = self.mapToScene(view_pos)
                if pos.x() >= HEADER_WIDTH:
                    seconds = (pos.x() - HEADER_WIDTH) / self.pixels_per_second
                    self.seek_requested.emit(max(0.0, seconds))
        super().mousePressEvent(event)

    def _drag_seconds(self, event) -> float:
        # Bar lines follow the audio, so they do not snap to the grid.
        x = self.mapToScene(event.position().toPoint()).x()
        return max(0.0, (x - HEADER_WIDTH) / self.pixels_per_second)

    def mouseMoveEvent(self, event):
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
            if line >= 0:
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                QToolTip.showText(event.globalPosition().toPoint(), self.bar_tooltip(line), self.viewport())
            else:
                self.viewport().unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
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
            self._scroll_horizontal(self._wheel_pixels(event))
            event.accept()
            return
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
