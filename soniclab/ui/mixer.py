import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from soniclab.ui.meter import PeakMeter
from soniclab.ui.timeline import track_color


def format_db(gain: float) -> str:
    if gain <= 0.0001:
        return "−∞"
    return f"{20.0 * math.log10(max(gain, 0.0001)):+.1f}"


def file_peak_db(audio) -> str:
    if audio is None or len(audio) == 0:
        return "—"
    peak = float(np.max(np.abs(np.asarray(audio))))
    if peak <= 1e-8:
        return "−∞"
    return f"{20.0 * math.log10(peak):+.1f}"


class ChannelStrip(QFrame):
    selected = Signal(int)
    gain_changed = Signal(int, float)
    pan_changed = Signal(int, float)
    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)

    def __init__(self, index: int, track: dict):
        super().__init__()
        self.index = index
        self.setObjectName("mixerStrip")
        self.setFixedWidth(96)
        self._master = index < 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        color = QWidget()
        color.setFixedHeight(3)
        swatch = "#C8CED6" if self._master else track_color(index)
        color.setStyleSheet(f"background: {swatch};")
        layout.addWidget(color)

        raw_name = track.get("name", f"Track {index + 1}")
        name = QLabel(
            QFontMetrics(self.font()).elidedText(raw_name, Qt.TextElideMode.ElideRight, 68)
        )
        name.setToolTip(raw_name)
        layout.addWidget(name)

        if self._master:
            peak = QLabel("Live")
            peak.setToolTip("Master output. The bar is the live mix peak.")
        else:
            peak = QLabel(f"File {file_peak_db(track.get('_audio'))}")
            peak.setToolTip("File peak. The bar beside the fader is the live track level.")
        peak.setObjectName("dim")
        layout.addWidget(peak)

        self.pan = QSlider(Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setToolTip("Pan")
        layout.addWidget(self.pan)

        self.gain = QSlider(Qt.Orientation.Vertical)
        self.gain.setRange(0, 150)
        self.gain.setMinimumHeight(108)
        self.gain.setToolTip("Master volume" if self._master else "Volume")
        fader = QHBoxLayout()
        fader.setSpacing(4)
        fader.addWidget(self.gain, 1)
        self.meter = PeakMeter()
        fader.addWidget(self.meter)
        layout.addLayout(fader, 1)

        self.readout = QLabel("0.0")
        self.readout.setObjectName("dim")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.readout)

        marks = QHBoxLayout()
        marks.setSpacing(4)
        self.mute = self._mark("M", "Mute")
        self.solo = self._mark("S", "Solo")
        marks.addWidget(self.mute)
        marks.addWidget(self.solo)
        marks.addStretch(1)
        layout.addLayout(marks)

        self.set_gain(float(track.get("gain", 1.0)))
        self.set_pan(float(track.get("pan", 0.0)))
        self.set_muted(bool(track.get("muted", False)))
        self.set_solo(bool(track.get("solo", False)))
        self.gain.valueChanged.connect(
            lambda value: self.gain_changed.emit(index, value / 100.0)
        )
        self.pan.valueChanged.connect(
            lambda value: self.pan_changed.emit(index, value / 100.0)
        )
        self.mute.toggled.connect(lambda checked: self.mute_changed.emit(index, checked))
        self.solo.toggled.connect(lambda checked: self.solo_changed.emit(index, checked))
        self.gain.sliderPressed.connect(lambda: self.selected.emit(index))
        self.pan.sliderPressed.connect(lambda: self.selected.emit(index))
        if self._master:
            self.solo.hide()
            self.pan.hide()

    def _mark(self, text: str, tip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("mixBtn")
        button.setText(text)
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setFixedSize(18, 16)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

    def set_selected(self, selected: bool):
        self.setProperty("selected", bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_gain(self, gain: float):
        self.gain.blockSignals(True)
        self.gain.setValue(int(round(float(gain) * 100)))
        self.readout.setText(format_db(gain))
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

    def set_level(self, level: float) -> None:
        self.meter.set_level(level)


class Mixer(QWidget):
    selected = Signal(int)
    gain_changed = Signal(int, float)
    pan_changed = Signal(int, float)
    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)
    edit_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: list[ChannelStrip] = []
        self._master: ChannelStrip | None = None
        self._master_gain = 1.0
        self._master_muted = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.row = QHBoxLayout(self.body)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(0)
        self.scroll.setWidget(self.body)
        outer.addWidget(self.scroll)
        self.rebuild([], -1)

    def rebuild(self, tracks, selected: int):
        while self.row.count():
            item = self.row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._strips = []
        self._master = None
        if not tracks:
            empty = QLabel("Import audio to add channel strips.")
            empty.setObjectName("dim")
            empty.setContentsMargins(12, 8, 12, 8)
            self.row.addWidget(empty)
        for index, track in enumerate(tracks):
            strip = self._bind_strip(ChannelStrip(index, track), selected)
            self.row.addWidget(strip)
            self._strips.append(strip)
        master = ChannelStrip(-1, {"name": "Master", "gain": self._master_gain, "muted": self._master_muted})
        master.set_muted(self._master_muted)
        self._bind_strip(master, -2)
        self._master = master
        self.row.addWidget(master)
        self.row.addStretch(1)

    def _bind_strip(self, strip: ChannelStrip, selected: int) -> ChannelStrip:
        strip.set_selected(strip.index == selected)
        strip.selected.connect(self.selected)
        strip.gain_changed.connect(self.gain_changed)
        strip.pan_changed.connect(self.pan_changed)
        strip.mute_changed.connect(self.mute_changed)
        strip.solo_changed.connect(self.solo_changed)
        strip.gain.sliderReleased.connect(self.edit_finished.emit)
        strip.pan.sliderReleased.connect(self.edit_finished.emit)
        return strip

    def set_selected(self, index: int):
        for row, strip in enumerate(self._strips):
            strip.set_selected(row == index)

    def set_gain(self, index: int, gain: float):
        if 0 <= index < len(self._strips):
            self._strips[index].set_gain(gain)

    def set_pan(self, index: int, pan: float):
        if 0 <= index < len(self._strips):
            self._strips[index].set_pan(pan)

    def set_muted(self, index: int, muted: bool):
        if 0 <= index < len(self._strips):
            self._strips[index].set_muted(muted)

    def set_solo(self, index: int, solo: bool):
        if 0 <= index < len(self._strips):
            self._strips[index].set_solo(solo)

    def set_master_gain(self, gain: float) -> None:
        self._master_gain = float(gain)
        if self._master is not None:
            self._master.set_gain(gain)

    def set_master_muted(self, muted: bool) -> None:
        self._master_muted = bool(muted)
        if self._master is not None:
            self._master.set_muted(muted)

    def set_levels(self, tracks, master: float) -> None:
        for strip, level in zip(self._strips, tracks):
            strip.set_level(level)
        if self._master is not None:
            self._master.set_level(master)
