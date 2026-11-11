"""FFT helpers used by the visualizer.

These functions only see buffers the UI has already copied out of the
audio callback. They never run on the real-time thread.
"""

import numpy as np

BANDS = (
    ("Sub", 20.0, 60.0),
    ("Bass", 60.0, 250.0),
    ("Low mid", 250.0, 500.0),
    ("Mid", 500.0, 2000.0),
    ("High mid", 2000.0, 6000.0),
    ("High", 6000.0, 20000.0),
)

FLOOR_DB = -90.0


def _mono(samples) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def _prepared(samples, fft_size: int) -> np.ndarray:
    mono = _mono(samples)
    if mono.size == 0:
        return mono
    if mono.size > fft_size:
        mono = mono[-fft_size:]
    if mono.size < 32:
        return mono[:0]
    window = np.hanning(mono.size).astype(np.float32)
    return mono * window


def magnitude_db(samples, rate: float, fft_size: int = 2048):
    """Return (db, freqs) for one window of real samples."""
    windowed = _prepared(samples, fft_size)
    if windowed.size < 32:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    spec = np.fft.rfft(windowed)
    mag = (np.abs(spec) * (2.0 / windowed.size)).astype(np.float32)
    db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / float(rate)).astype(np.float32)
    return db, freqs


def _collect(db, freqs, f_min, f_max) -> float:
    if db.size == 0:
        return FLOOR_DB
    mask = (freqs >= f_min) & (freqs < f_max)
    if not np.any(mask):
        return FLOOR_DB
    return float(np.max(db[mask]))


def spectrum_bars(samples, rate: float, bars: int = 48, f_min: float = 20.0, f_max: float = 20000.0, fft_size: int = 2048):
    """Log-spaced bar levels in dB, plus the bar edges in Hz."""
    bars = max(4, int(bars))
    f_min = max(10.0, float(f_min))
    f_max = max(f_min * 1.5, min(float(rate) * 0.5, float(f_max)))
    edges = np.logspace(np.log10(f_min), np.log10(f_max), bars + 1).astype(np.float32)
    db, freqs = magnitude_db(samples, rate, fft_size)
    levels = np.empty(bars, dtype=np.float32)
    for index in range(bars):
        levels[index] = _collect(db, freqs, edges[index], edges[index + 1])
    return levels, edges


def band_levels(samples, rate: float, fft_size: int = 2048) -> list[tuple[str, float, float, float]]:
    db, freqs = magnitude_db(samples, rate, fft_size)
    return [(name, low, high, _collect(db, freqs, low, high)) for name, low, high in BANDS]


def scope_window(samples, rate: float, seconds: float) -> np.ndarray:
    """Return the newest `seconds` of audio, unmodified."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    count = max(2, int(float(seconds) * float(rate)))
    if audio.shape[0] <= count:
        return audio
    return audio[-count:]


def rising_trigger(mono, search: int) -> int:
    """Index of a rising zero crossing, or 0 when the buffer does not cross."""
    limit = max(2, min(int(search), len(mono) - 1))
    for index in range(1, limit):
        if mono[index - 1] <= 0.0 < mono[index]:
            return index
    return 0


def spectrogram_column(samples, rate: float, rows: int = 96, f_min: float = 20.0, f_max: float = 20000.0, fft_size: int = 2048):
    levels, _edges = spectrum_bars(samples, rate, rows, f_min, f_max, fft_size)
    return levels


class Spectrogram:
    """Scrolling columns of dB values. Column 0 is the oldest."""

    def __init__(self, rows: int = 96, columns: int = 180):
        self.rows = int(rows)
        self.columns = int(columns)
        self.values = np.full((self.rows, self.columns), FLOOR_DB, dtype=np.float32)

    def push(self, column) -> None:
        column = np.asarray(column, dtype=np.float32).reshape(-1)
        if column.size != self.rows:
            fitted = np.full(self.rows, FLOOR_DB, dtype=np.float32)
            count = min(self.rows, column.size)
            fitted[:count] = column[:count]
            column = fitted
        self.values[:, :-1] = self.values[:, 1:]
        self.values[:, -1] = column

    def clear(self) -> None:
        self.values.fill(FLOOR_DB)
