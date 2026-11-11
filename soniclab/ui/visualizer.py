"""Dockable spectrum, oscilloscope, and spectrogram.

The widget only draws buffers handed to it. It does not synthesize audio.
"""

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from soniclab.core.analyze import (
    FLOOR_DB,
    Spectrogram,
    rising_trigger,
    scope_window,
    spectrogram_column,
    spectrum_bars,
)
from soniclab.core.audio_engine import OUTPUT_RATE

_LUT = None


def _color_lut() -> np.ndarray:
    global _LUT
    if _LUT is not None:
        return _LUT
    stops = np.array(
        [
            [12, 14, 18],
            [16, 42, 78],
            [48, 78, 112],
            [140, 96, 58],
            [215, 179, 106],
            [244, 236, 220],
        ],
        dtype=np.float32,
    )
    lut = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        position = index / 255.0 * (len(stops) - 1)
        left = int(position)
        right = min(len(stops) - 1, left + 1)
        mix = position - left
        lut[index] = stops[left] * (1.0 - mix) + stops[right] * mix
    _LUT = lut
    return lut


def _px(value) -> int:
    value = float(value)
    if value != value or value > 100000 or value < -100000:
        return 0
    return int(value)


def _hz(value: float) -> str:
    if value >= 1000:
        shown = value / 1000.0
        return f"{shown:.0f}k" if shown >= 10 else f"{shown:.1f}k"
    return f"{value:.0f}"


class _Plot(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setMinimumHeight(140)

    def paintEvent(self, event):
        self.owner._paint_plot(event)


class Visualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader = None
        self._mode = "spectrum"
        self._frozen = False
        self._active = True
        self._source = -1
        self._stereo = True
        self._trigger = True
        self._peak_hold = True
        self._f_min = 20.0
        self._f_max = 20000.0
        self._fft = 2048
        self._bars = 48
        self._scope_seconds = 0.02
        self._amp = 1.0
        self._display = np.full(self._bars, FLOOR_DB, dtype=np.float32)
        self._peaks = np.full(self._bars, FLOOR_DB, dtype=np.float32)
        self._edges = None
        self._scope = np.zeros((0, 2), dtype=np.float32)
        self._gram = Spectrogram(96, 220)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
        root.addLayout(self._build_toolbar())
        root.addLayout(self._build_options())
        self.plot = _Plot(self)
        root.addWidget(self.plot, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_reader(self, reader) -> None:
        self._reader = reader

    def set_sources(self, names) -> None:
        current = self.source.currentData()
        self.source.blockSignals(True)
        self.source.clear()
        self.source.addItem("Master", -1)
        for index, name in enumerate(names):
            self.source.addItem(f"{index + 1:02d}  {name}", index)
        found = self.source.findData(current if current is not None else -1)
        self.source.setCurrentIndex(max(0, found))
        self.source.blockSignals(False)
        self._source = int(self.source.currentData())

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if self._active and not self._timer.isActive():
            self._timer.start()
        if not self._active:
            self._timer.stop()

    def state(self) -> dict:
        return {
            "mode": self._mode,
            "fft": self._fft,
            "range": self.range_box.currentText(),
            "stereo": self._stereo,
            "peak": self._peak_hold,
            "trigger": self._trigger,
            "scope": self._scope_seconds,
            "rate": self._timer.interval(),
        }

    def apply_state(self, state: dict) -> None:
        if not state:
            return
        mode = state.get("mode", "spectrum")
        if mode in ("spectrum", "scope", "spectrogram"):
            self._set_mode(mode)
        fft = int(state.get("fft", 2048))
        found = self.fft_box.findData(fft)
        if found >= 0:
            self.fft_box.setCurrentIndex(found)
        range_name = state.get("range")
        if range_name:
            index = self.range_box.findText(str(range_name))
            if index >= 0:
                self.range_box.setCurrentIndex(index)
        self.stereo_box.setChecked(bool(state.get("stereo", True)))
        self.peak_box.setChecked(bool(state.get("peak", True)))
        self.trigger_box.setChecked(bool(state.get("trigger", True)))

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(4)
        self._modes = QButtonGroup(self)
        self._modes.setExclusive(True)
        self._mode_buttons = {}
        for key, label in (("spectrum", "Spectrum"), ("scope", "Scope"), ("spectrogram", "Spectrogram")):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip(label)
            self._modes.addButton(button)
            self._mode_buttons[key] = button
            row.addWidget(button)
            button.clicked.connect(lambda _checked=False, name=key: self._set_mode(name))
        self._mode_buttons["spectrum"].setChecked(True)
        row.addSpacing(8)
        self.source = QComboBox()
        self.source.addItem("Master", -1)
        self.source.setMinimumWidth(140)
        self.source.currentIndexChanged.connect(self._source_changed)
        row.addWidget(self.source)
        self.freeze_btn = QToolButton()
        self.freeze_btn.setText("Freeze")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.toggled.connect(self._set_frozen)
        row.addWidget(self.freeze_btn)
        reset = QToolButton()
        reset.setText("Reset")
        reset.clicked.connect(self.reset_view)
        row.addWidget(reset)
        row.addStretch(1)
        return row

    def _build_options(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        self.range_box = QComboBox()
        self.range_box.addItem("20 Hz – 20 kHz", (20.0, 20000.0))
        self.range_box.addItem("20 Hz – 250 Hz", (20.0, 250.0))
        self.range_box.addItem("250 Hz – 2 kHz", (250.0, 2000.0))
        self.range_box.addItem("2 kHz – 20 kHz", (2000.0, 20000.0))
        self.range_box.currentIndexChanged.connect(self._range_changed)
        self.fft_box = QComboBox()
        for size in (1024, 2048, 4096, 8192):
            self.fft_box.addItem(f"FFT {size}", size)
        self.fft_box.setCurrentIndex(1)
        self.fft_box.currentIndexChanged.connect(self._fft_changed)
        self.rate_box = QComboBox()
        self.rate_box.addItem("30 fps", 33)
        self.rate_box.addItem("20 fps", 50)
        self.rate_box.addItem("10 fps", 100)
        self.rate_box.currentIndexChanged.connect(self._rate_changed)
        self.time_box = QComboBox()
        for label, seconds in (("5 ms", 0.005), ("10 ms", 0.01), ("20 ms", 0.02), ("50 ms", 0.05), ("100 ms", 0.1)):
            self.time_box.addItem(label, seconds)
        self.time_box.setCurrentIndex(2)
        self.time_box.currentIndexChanged.connect(self._scope_changed)
        self.amp_box = QComboBox()
        for label, scale in (("1×", 1.0), ("2×", 2.0), ("4×", 4.0)):
            self.amp_box.addItem(label, scale)
        self.amp_box.currentIndexChanged.connect(self._amp_changed)
        self.stereo_box = QCheckBox("Stereo")
        self.stereo_box.setChecked(True)
        self.stereo_box.toggled.connect(self._stereo_changed)
        self.peak_box = QCheckBox("Peak hold")
        self.peak_box.setChecked(True)
        self.peak_box.toggled.connect(lambda checked: setattr(self, "_peak_hold", bool(checked)))
        self.trigger_box = QCheckBox("Trigger")
        self.trigger_box.setChecked(True)
        self.trigger_box.toggled.connect(lambda checked: setattr(self, "_trigger", bool(checked)))
        for widget in (
            self.range_box,
            self.fft_box,
            self.rate_box,
            self.time_box,
            self.amp_box,
            self.stereo_box,
            self.peak_box,
            self.trigger_box,
        ):
            row.addWidget(widget)
        row.addStretch(1)
        self._sync_option_visibility()
        return row

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        for key, button in self._mode_buttons.items():
            button.setChecked(key == mode)
        self._sync_option_visibility()
        self.plot.update()

    def _sync_option_visibility(self) -> None:
        spectrum = self._mode == "spectrum"
        scope = self._mode == "scope"
        self.range_box.setVisible(not scope)
        self.fft_box.setVisible(not scope)
        self.peak_box.setVisible(spectrum)
        self.time_box.setVisible(scope)
        self.amp_box.setVisible(scope)
        self.trigger_box.setVisible(scope)
        self.stereo_box.setVisible(scope)

    def _source_changed(self) -> None:
        data = self.source.currentData()
        self._source = -1 if data is None else int(data)
        self.reset_view()

    def _set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)
        self.freeze_btn.setText("Unfreeze" if self._frozen else "Freeze")

    def reset_view(self) -> None:
        self._display.fill(FLOOR_DB)
        self._peaks.fill(FLOOR_DB)
        self._scope = np.zeros((0, 2), dtype=np.float32)
        self._gram.clear()
        self.plot.update()

    def _range_changed(self) -> None:
        low, high = self.range_box.currentData()
        self._f_min, self._f_max = float(low), float(high)
        self.reset_view()

    def _fft_changed(self) -> None:
        self._fft = int(self.fft_box.currentData())

    def _rate_changed(self) -> None:
        self._timer.setInterval(int(self.rate_box.currentData()))

    def _scope_changed(self) -> None:
        self._scope_seconds = float(self.time_box.currentData())

    def _amp_changed(self) -> None:
        self._amp = float(self.amp_box.currentData())

    def _stereo_changed(self, checked: bool) -> None:
        self._stereo = bool(checked)

    def samples_needed(self) -> int:
        scope = int(self._scope_seconds * OUTPUT_RATE) + 8
        return max(self._fft, scope, 1024)

    def _tick(self) -> None:
        if not self._active or self._frozen or self._reader is None:
            return
        audio = self._reader(self._source, self.samples_needed())
        self.analyze(audio)
        self.plot.update()

    def analyze(self, audio) -> None:
        """Update the current mode from a real buffer. None decays to silence."""
        if audio is None or len(audio) == 0:
            audio = np.zeros((self._fft, 2), dtype=np.float32)
        else:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 1:
                audio = np.column_stack((audio, audio))
        if self._mode == "spectrum":
            levels, edges = spectrum_bars(
                audio, OUTPUT_RATE, self._bars, self._f_min, self._f_max, self._fft
            )
            self._edges = edges
            self._approach(levels)
        elif self._mode == "scope":
            window = scope_window(audio, OUTPUT_RATE, self._scope_seconds)
            if self._trigger and len(window) > 8:
                start = rising_trigger(window[:, 0], max(2, len(window) // 2))
                if start:
                    window = window[start:]
            self._scope = window
        else:
            column = spectrogram_column(
                audio, OUTPUT_RATE, self._gram.rows, self._f_min, self._f_max, self._fft
            )
            self._gram.push(column)

    def _approach(self, target) -> None:
        if self._display.shape != target.shape:
            self._display = np.full(target.shape, FLOOR_DB, dtype=np.float32)
            self._peaks = self._display.copy()
        delta = target - self._display
        self._display = self._display + np.where(delta > 0.0, delta * 0.6, delta * 0.16)
        if self._peak_hold:
            self._peaks = np.maximum(self._peaks - 0.7, self._display)
        else:
            self._peaks.fill(FLOOR_DB)

    def _paint_plot(self, event):
        painter = QPainter(self.plot)
        painter.fillRect(self.plot.rect(), QColor("#0C0D10"))
        if self._mode == "spectrum":
            self._paint_spectrum(painter)
        elif self._mode == "scope":
            self._paint_scope(painter)
        else:
            self._paint_spectrogram(painter)
        painter.end()

    def _plot_rect(self):
        rect = self.plot.rect().adjusted(36, 8, -8, -18)
        if rect.width() < 10 or rect.height() < 10:
            return self.plot.rect()
        return rect

    def _paint_grid(self, painter, rect, horizontal=4):
        painter.setPen(QPen(QColor("#1E242C"), 1))
        for step in range(1, horizontal):
            y = rect.top() + rect.height() * step / horizontal
            painter.drawLine(_px(rect.left()), _px(y), _px(rect.right()), _px(y))

    def _paint_spectrum(self, painter):
        rect = self._plot_rect()
        self._paint_grid(painter, rect, 4)
        count = max(1, len(self._display))
        gap = 2
        bar_width = max(1, (rect.width() - gap * count) / count)
        for index, level in enumerate(self._display):
            span = (np.clip(level, FLOOR_DB, 0.0) - FLOOR_DB) / -FLOOR_DB
            height = rect.height() * float(span)
            x = rect.left() + index * (bar_width + gap)
            painter.fillRect(
                _px(x),
                _px(rect.bottom() - height),
                _px(bar_width),
                max(0, _px(height)),
                QColor("#9AA6B4"),
            )
            if self._peak_hold:
                peak = (np.clip(self._peaks[index], FLOOR_DB, 0.0) - FLOOR_DB) / -FLOOR_DB
                y = rect.bottom() - rect.height() * float(peak)
                painter.fillRect(_px(x), _px(y), _px(bar_width), 2, QColor("#D7B36A"))
        painter.setPen(QColor("#8B939F"))
        for db, fraction in ((0, 0.0), (-30, 0.33), (-60, 0.66), (-90, 1.0)):
            y = rect.top() + rect.height() * fraction
            painter.drawText(2, int(y + 4), f"{db}")
        if self._edges is not None and len(self._edges) > 1:
            for mark in (0, len(self._edges) // 3, 2 * len(self._edges) // 3, len(self._edges) - 1):
                x = rect.left() + rect.width() * mark / max(1, len(self._edges) - 1)
                painter.drawText(int(x), self.plot.height() - 4, _hz(float(self._edges[mark])))

    def _paint_scope(self, painter):
        rect = self._plot_rect()
        painter.setPen(QPen(QColor("#2A3038"), 1))
        if self._stereo:
            half = rect.height() / 2
            centers = (rect.top() + half / 2, rect.top() + half + half / 2)
            colors = (QColor("#9BB0C7"), QColor("#D7B36A"))
            for center in centers:
                painter.drawLine(_px(rect.left()), _px(center), _px(rect.right()), _px(center))
            series = (0, 1)
        else:
            centers = (rect.center().y(),)
            colors = (QColor("#E6E8EC"),)
            painter.drawLine(_px(rect.left()), _px(centers[0]), _px(rect.right()), _px(centers[0]))
            series = (0,)
        audio = self._scope
        if len(audio) < 2:
            return
        amp = max(0.1, self._amp)
        scale = (rect.height() / (4 if self._stereo else 2)) * amp
        for channel, center, color in zip(series, centers, colors):
            column = audio[:, channel]
            self._stroke(painter, rect, column, center, scale, color)

    def _stroke(self, painter, rect, column, center, scale, color):
        width = max(1, rect.width())
        if len(column) > width:
            usable = (len(column) // width) * width
            folded = column[-usable:].reshape(width, -1)
            low = folded.min(axis=1)
            high = folded.max(axis=1)
            painter.setPen(QPen(color, 1))
            for index in range(width):
                x = rect.left() + index
                y0 = center - float(high[index]) * scale
                y1 = center - float(low[index]) * scale
                painter.drawLine(_px(x), _px(y0), _px(x), _px(y1))
            return
        painter.setPen(QPen(color, 1.2))
        previous = None
        for index, sample in enumerate(column):
            x = _px(rect.left() + rect.width() * index / max(1, len(column) - 1))
            y = _px(center - float(sample) * scale)
            if previous is not None:
                painter.drawLine(previous[0], previous[1], x, y)
            previous = (x, y)

    def _paint_spectrogram(self, painter):
        rect = self._plot_rect()
        values = np.clip((self._gram.values - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0)
        indices = (values * 255.0).astype(np.uint8)
        colors = np.ascontiguousarray(_color_lut()[indices][::-1])
        self._gram_bytes = colors.tobytes()
        image = QImage(
            self._gram_bytes,
            int(colors.shape[1]),
            int(colors.shape[0]),
            int(colors.shape[1] * 3),
            QImage.Format.Format_RGB888,
        )
        painter.drawImage(rect, image)
        painter.setPen(QColor("#8B939F"))
        for fraction, hz in ((0.02, self._f_max), (0.5, (self._f_min * self._f_max) ** 0.5), (0.96, self._f_min)):
            y = rect.top() + rect.height() * fraction
            painter.drawText(2, int(y), _hz(hz))
        painter.drawText(rect.left(), self.plot.height() - 4, "older")
        painter.drawText(rect.right() - 36, self.plot.height() - 4, "now")
