import math
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def format_db(gain: float) -> str:
    if gain <= 0.0001:
        return "−∞"
    return f"{20.0 * math.log10(gain):+.1f} dB"


class Inspector(QWidget):
    name_edited = Signal(str)
    start_edited = Signal(float)
    gain_edited = Signal(float)
    pan_edited = Signal(float)
    mute_edited = Signal(bool)
    solo_edited = Signal(bool)
    clip_gain_edited = Signal(float)
    fade_in_edited = Signal(float)
    fade_out_edited = Signal(float)
    edit_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filling = False
        self._index = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        self.heading = QLabel("No selection")
        self.heading.setObjectName("sectionLabel")
        layout.addWidget(self.heading)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.name_edit = QLineEdit()
        self.file_label = QLabel("—")
        self.file_label.setObjectName("dim")
        self.file_label.setWordWrap(True)
        self.start = QDoubleSpinBox()
        self.start.setRange(0, 7200)
        self.start.setDecimals(3)
        self.start.setSuffix(" s")
        self.start.setSingleStep(0.25)
        self.duration = QLabel("—")
        self.duration.setObjectName("dim")
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 1.5)
        self.gain.setDecimals(2)
        self.gain.setSingleStep(0.01)
        self.gain_readout = QLabel("0.0 dB")
        self.gain_readout.setObjectName("dim")
        self.pan = QSlider(Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan_readout = QLabel("C")
        self.pan_readout.setObjectName("dim")
        self.mute = QCheckBox("Mute")
        self.solo = QCheckBox("Solo")
        form.addRow("Name", self.name_edit)
        form.addRow("File", self.file_label)
        form.addRow("Start", self.start)
        form.addRow("Duration", self.duration)
        form.addRow("Gain", self.gain)
        form.addRow("", self.gain_readout)
        form.addRow("Pan", self.pan)
        form.addRow("", self.pan_readout)
        self.offset = QLabel("0.000 s")
        self.offset.setObjectName("dim")
        self.clip_gain = QDoubleSpinBox()
        self.clip_gain.setRange(0, 2)
        self.clip_gain.setDecimals(2)
        self.clip_gain.setSingleStep(0.05)
        self.clip_gain.setValue(1)
        self.fade_in = QDoubleSpinBox()
        self.fade_out = QDoubleSpinBox()
        for box in (self.fade_in, self.fade_out):
            box.setRange(0, 30)
            box.setDecimals(3)
            box.setSuffix(" s")
            box.setSingleStep(0.05)
        form.addRow("", self.mute)
        form.addRow("", self.solo)
        form.addRow("Offset", self.offset)
        form.addRow("Clip gain", self.clip_gain)
        form.addRow("Fade in", self.fade_in)
        form.addRow("Fade out", self.fade_out)
        layout.addLayout(form)
        note = QLabel("Drag a clip edge to trim. Effects are not available yet.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

        self.name_edit.editingFinished.connect(self._emit_name)
        self.start.editingFinished.connect(self._emit_start)
        self.gain.valueChanged.connect(self._emit_gain)
        self.pan.valueChanged.connect(self._emit_pan)
        self.mute.toggled.connect(self._emit_mute)
        self.solo.toggled.connect(self._emit_solo)
        self.clip_gain.valueChanged.connect(self._emit_clip_gain)
        self.fade_in.editingFinished.connect(self._emit_fade_in)
        self.fade_out.editingFinished.connect(self._emit_fade_out)
        self.clip_gain.editingFinished.connect(self.edit_finished.emit)
        self.gain.editingFinished.connect(self.edit_finished.emit)
        self.pan.sliderReleased.connect(self.edit_finished.emit)
        self.show_track(-1, None)

    def show_track(self, index: int, track):
        self._filling = True
        self._index = index
        active = track is not None
        for widget in (
            self.name_edit,
            self.start,
            self.gain,
            self.pan,
            self.mute,
            self.solo,
            self.clip_gain,
            self.fade_in,
            self.fade_out,
        ):
            widget.setEnabled(active)
        if not active:
            self.heading.setText("No selection")
            self.name_edit.clear()
            self.file_label.setText("—")
            self.duration.setText("—")
            self.start.setValue(0)
            self.gain.setValue(1)
            self.pan.setValue(0)
            self.gain_readout.setText("0.0 dB")
            self.pan_readout.setText("C")
            self.mute.setChecked(False)
            self.solo.setChecked(False)
            self.offset.setText("0.000 s")
            self.clip_gain.setValue(1)
            self.fade_in.setValue(0)
            self.fade_out.setValue(0)
        else:
            name = track.get("name", "Track")
            self.heading.setText(f"Track {index + 1:02d}")
            self.name_edit.setText(name)
            path = track.get("path") or "No source file"
            self.file_label.setText(os.path.basename(path) if track.get("path") else path)
            self.file_label.setToolTip(path)
            self.start.setValue(float(track.get("start", 0.0)))
            self.duration.setText(f"{float(track.get('duration', 0.0)):.3f} s")
            gain = float(track.get("gain", 1.0))
            pan = float(track.get("pan", 0.0))
            self.gain.setValue(gain)
            self.pan.setValue(int(round(pan * 100)))
            self.gain_readout.setText(format_db(gain))
            self.pan_readout.setText(self._pan_text(pan))
            self.mute.setChecked(bool(track.get("muted", False)))
            self.solo.setChecked(bool(track.get("solo", False)))
            self.offset.setText(f"{float(track.get('offset', 0.0)):.3f} s")
            self.clip_gain.setValue(float(track.get("clip_gain", 1.0)))
            self.fade_in.setValue(float(track.get("fade_in", 0.0)))
            self.fade_out.setValue(float(track.get("fade_out", 0.0)))
        self._filling = False

    def set_start(self, seconds: float):
        self.start.blockSignals(True)
        self.start.setValue(float(seconds))
        self.start.blockSignals(False)

    def set_gain(self, gain: float):
        self.gain.blockSignals(True)
        self.gain.setValue(float(gain))
        self.gain_readout.setText(format_db(gain))
        self.gain.blockSignals(False)

    def set_pan(self, pan: float):
        self.pan.blockSignals(True)
        self.pan.setValue(int(round(float(pan) * 100)))
        self.pan_readout.setText(self._pan_text(pan))
        self.pan.blockSignals(False)

    def set_muted(self, muted: bool):
        self.mute.blockSignals(True)
        self.mute.setChecked(bool(muted))
        self.mute.blockSignals(False)

    def set_solo(self, solo: bool):
        self.solo.blockSignals(True)
        self.solo.setChecked(bool(solo))
        self.solo.blockSignals(False)

    def _emit_name(self):
        if self._filling or self._index < 0:
            return
        self.name_edited.emit(self.name_edit.text())

    def _emit_start(self):
        if self._filling or self._index < 0:
            return
        self.start_edited.emit(float(self.start.value()))

    def _emit_gain(self, value: float):
        self.gain_readout.setText(format_db(value))
        if self._filling or self._index < 0:
            return
        self.gain_edited.emit(float(value))

    def _emit_pan(self, value: int):
        pan = value / 100.0
        self.pan_readout.setText(self._pan_text(pan))
        if self._filling or self._index < 0:
            return
        self.pan_edited.emit(pan)

    def _emit_mute(self, checked: bool):
        if self._filling or self._index < 0:
            return
        self.mute_edited.emit(bool(checked))

    def _emit_solo(self, checked: bool):
        if self._filling or self._index < 0:
            return
        self.solo_edited.emit(bool(checked))

    def _emit_clip_gain(self, value: float):
        if self._filling or self._index < 0:
            return
        self.clip_gain_edited.emit(float(value))

    def _emit_fade_in(self):
        if self._filling or self._index < 0:
            return
        self.fade_in_edited.emit(float(self.fade_in.value()))

    def _emit_fade_out(self):
        if self._filling or self._index < 0:
            return
        self.fade_out_edited.emit(float(self.fade_out.value()))

    @staticmethod
    def _pan_text(pan: float) -> str:
        if abs(pan) < 0.01:
            return "C"
        if pan < 0:
            return f"L {abs(pan) * 100:.0f}"
        return f"R {pan * 100:.0f}"
