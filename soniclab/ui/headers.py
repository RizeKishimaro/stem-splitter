from PySide6.QtCore import Qt, Signal, QEvent, QSize
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from soniclab.ui.timeline import RULER_HEIGHT, TRACK_HEIGHT, track_color


class TrackHeader(QFrame):
    selected = Signal(int)
    gain_changed = Signal(int, float)
    pan_changed = Signal(int, float)
    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)
    resize_committed = Signal(int, int)
    unavailable = Signal(str)
    arm_changed = Signal(int, bool)
    edit_finished = Signal()
    rename_requested = Signal(int)

    def __init__(self, index: int, track: dict, height: int):
        super().__init__()
        self.index = index
        self._name = track.get("name", f"Track {index + 1}")
        self._drag = False
        self._drag_y = 0.0
        self._drag_h = height
        self.setObjectName("trackHeader")
        self.setFixedHeight(height)
        self.setProperty("selected", False)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        color = QWidget()
        color.setFixedWidth(3)
        color.setStyleSheet(f"background: {track_color(index)};")
        root.addWidget(color)

        inner = QVBoxLayout()
        inner.setContentsMargins(8, 4, 8, 0)
        inner.setSpacing(2)
        root.addLayout(inner, 1)

        title = QHBoxLayout()
        title.setSpacing(6)
        self.number = QLabel(f"{index + 1:02d}")
        self.number.setObjectName("trackIndex")
        self.name_label = QLabel(self._name)
        self.name_label.setToolTip("Double-click to rename")
        self.name_label.setCursor(Qt.CursorShape.IBeamCursor)
        # The full filename must not become the header's minimum width.
        # A huge size hint feeds the scroll area, the window grows, and Qt
        # crashes while measuring the text.
        self.name_label.setMinimumWidth(0)
        self.name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        title.addWidget(self.number)
        title.addWidget(self.name_label, 1)
        inner.addLayout(title)

        controls = QHBoxLayout()
        controls.setSpacing(4)
        self.mute = self._mark("M", "Mute")
        self.solo = self._mark("S", "Solo")
        self.arm = self._mark("R", "Record arm")
        self.arm.setObjectName("armBtn")
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 150)
        self.gain.setValue(100)
        self.gain.setFixedHeight(14)
        self.gain.setToolTip("Volume")
        self.pan = QSlider(Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(0)
        self.pan.setFixedWidth(54)
        self.pan.setFixedHeight(14)
        self.pan.setToolTip("Pan")
        controls.addWidget(self.mute)
        controls.addWidget(self.solo)
        controls.addWidget(self.arm)
        controls.addWidget(self.gain, 1)
        controls.addWidget(self.pan)
        inner.addLayout(controls)
        inner.addStretch(1)

        self.handle = QWidget()
        self.handle.setFixedHeight(5)
        self.handle.setCursor(Qt.CursorShape.SplitVCursor)
        self.handle.setToolTip("Drag to set track height")
        self.handle.installEventFilter(self)
        self.name_label.installEventFilter(self)
        inner.addWidget(self.handle)

        self._apply(track)
        self.mute.toggled.connect(lambda checked: self.mute_changed.emit(index, checked))
        self.solo.toggled.connect(lambda checked: self.solo_changed.emit(index, checked))
        self.gain.valueChanged.connect(
            lambda value: self.gain_changed.emit(index, value / 100.0)
        )
        self.pan.valueChanged.connect(
            lambda value: self.pan_changed.emit(index, value / 100.0)
        )
        self.arm.toggled.connect(lambda checked: self.arm_changed.emit(index, checked))
        self.gain.sliderPressed.connect(lambda: self.selected.emit(index))
        self.pan.sliderPressed.connect(lambda: self.selected.emit(index))
        self.gain.sliderReleased.connect(self.edit_finished.emit)
        self.pan.sliderReleased.connect(self.edit_finished.emit)
        self.mute.pressed.connect(lambda: self.selected.emit(index))
        self.solo.pressed.connect(lambda: self.selected.emit(index))
        self.arm.pressed.connect(lambda: self.selected.emit(index))
        if height <= 0:
            self.setFixedHeight(0)
            self.hide()

    def _mark(self, text: str, tip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("mixBtn")
        button.setText(text)
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setFixedSize(18, 16)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

    def _apply(self, track: dict):
        self.set_gain(float(track.get("gain", 1.0)))
        self.set_pan(float(track.get("pan", 0.0)))
        self.set_muted(bool(track.get("muted", False)))
        self.set_solo(bool(track.get("solo", False)))
        self.set_armed(bool(track.get("armed", False)))

    def set_selected(self, selected: bool):
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_gain(self, gain: float):
        self.gain.blockSignals(True)
        self.gain.setValue(int(round(float(gain) * 100)))
        self.gain.blockSignals(False)

    def set_pan(self, pan: float):
        self.pan.blockSignals(True)
        self.pan.setValue(int(round(float(pan) * 100)))
        self.pan.blockSignals(False)

    def set_muted(self, muted: bool):
        self.mute.blockSignals(True)
        self.mute.setChecked(bool(muted))
        self.mute.blockSignals(False)

    def set_solo(self, solo: bool):
        self.solo.blockSignals(True)
        self.solo.setChecked(bool(solo))
        self.solo.blockSignals(False)

    def set_armed(self, armed: bool):
        self.arm.blockSignals(True)
        self.arm.setChecked(bool(armed))
        self.arm.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = max(20, min(int(self.name_label.width()), 280))
        shown = QFontMetrics(self.name_label.font()).elidedText(
            self._name,
            Qt.TextElideMode.ElideRight,
            width,
        )
        if shown != self.name_label.text():
            self.name_label.setText(shown)

    def mousePressEvent(self, event):
        self.selected.emit(self.index)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "name_label", None) and event.type() == QEvent.Type.MouseButtonDblClick:
            self.rename_requested.emit(self.index)
            return True
        if obj is not getattr(self, "handle", None):
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._drag_y = event.globalPosition().y()
            self._drag_h = self.height()
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag:
            height = int(self._drag_h + event.globalPosition().y() - self._drag_y)
            self.setFixedHeight(max(TRACK_HEIGHT, min(220, height)))
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._drag:
            self._drag = False
            self.resize_committed.emit(self.index, self.height())
            return True
        return super().eventFilter(obj, event)


class _HeaderScroll(QScrollArea):
    """Scroll the headers without telling the window to become scene-tall."""

    def sizeHint(self):
        return QSize(228, 240)

    def minimumSizeHint(self):
        return QSize(208, 0)


class HeaderColumn(QWidget):
    selected = Signal(int)
    gain_changed = Signal(int, float)
    pan_changed = Signal(int, float)
    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)
    resize_committed = Signal(int, int)
    unavailable = Signal(str)
    arm_changed = Signal(int, bool)
    edit_finished = Signal()
    rename_requested = Signal(int)
    # The ▶ in the corner: play (or stop) this canvas on its own.
    play_requested = Signal()

    def __init__(self, parent=None, title: str = "Tracks"):
        super().__init__(parent)
        self.title = title
        self._playing = False
        self.play_button: QToolButton | None = None
        self.setObjectName("headerPane")
        self.setMinimumWidth(208)
        self.setMaximumWidth(340)
        self._headers: list[TrackHeader] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.scroll = _HeaderScroll()
        self.scroll.setWidgetResizable(True)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.column = QVBoxLayout(self.body)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(0)
        self.scroll.setWidget(self.body)
        layout.addWidget(self.scroll)
        self.rebuild([], [], -1)

    def rebuild(self, tracks, heights, selected: int):
        while self.column.count():
            item = self.column.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._headers = []
        corner = QFrame()
        corner.setObjectName("rulerCorner")
        corner.setFixedHeight(RULER_HEIGHT)
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(10, 0, 8, 0)
        corner_label = QLabel(self.title)
        corner_label.setObjectName("sectionLabel")
        corner_layout.addWidget(corner_label)
        corner_layout.addStretch(1)
        self.play_button = QToolButton()
        self.play_button.setObjectName("playStemBtn")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(lambda _checked=False: self.play_requested.emit())
        corner_layout.addWidget(self.play_button)
        self.set_playing(self._playing)
        self.column.addWidget(corner)
        for index, track in enumerate(tracks):
            height = heights[index] if index < len(heights) else TRACK_HEIGHT
            header = TrackHeader(index, track, height)
            header.set_selected(index == selected)
            header.selected.connect(self.selected)
            header.gain_changed.connect(self.gain_changed)
            header.pan_changed.connect(self.pan_changed)
            header.mute_changed.connect(self.mute_changed)
            header.solo_changed.connect(self.solo_changed)
            header.resize_committed.connect(self.resize_committed)
            header.unavailable.connect(self.unavailable)
            header.arm_changed.connect(self.arm_changed)
            header.edit_finished.connect(self.edit_finished)
            header.rename_requested.connect(self.rename_requested)
            self.column.addWidget(header)
            self._headers.append(header)
        self.column.addStretch(1)

    def set_playing(self, playing: bool) -> None:
        """Show whether this canvas is playing on its own."""
        self._playing = bool(playing)
        button = self.play_button
        if button is None:
            return
        button.setChecked(self._playing)
        button.setText("■" if self._playing else "▶")
        button.setToolTip(
            f"Stop {self.title.lower()}" if self._playing else f"Play only {self.title.lower()}"
        )

    def set_scene_height(self, height: int):
        self.body.setMinimumHeight(max(1, int(height)))

    def sync(self, value: int):
        self.scroll.verticalScrollBar().setValue(int(value))

    def set_selected(self, index: int):
        for row, header in enumerate(self._headers):
            header.set_selected(row == index)

    def set_gain(self, index: int, gain: float):
        if 0 <= index < len(self._headers):
            self._headers[index].set_gain(gain)

    def set_pan(self, index: int, pan: float):
        if 0 <= index < len(self._headers):
            self._headers[index].set_pan(pan)

    def set_muted(self, index: int, muted: bool):
        if 0 <= index < len(self._headers):
            self._headers[index].set_muted(muted)

    def set_solo(self, index: int, solo: bool):
        if 0 <= index < len(self._headers):
            self._headers[index].set_solo(solo)

    def set_armed(self, index: int, armed: bool):
        if 0 <= index < len(self._headers):
            self._headers[index].set_armed(armed)
