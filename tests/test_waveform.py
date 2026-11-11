import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QGraphicsLineItem

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.core.beats import analyze_beats
from soniclab.core.waveform import cached_envelope, display_envelope, peak_envelope
from soniclab.ui.timeline import ClipItem, TimelineView


def _clicks(period=0.5, seconds=8.0, rate=OUTPUT_RATE):
    audio = np.zeros(int(seconds * rate), dtype=np.float32)
    click = np.hanning(300).astype(np.float32)
    for beat in np.arange(period, seconds, period):
        start = int(beat * rate)
        audio[start : start + click.size] += click
    return np.column_stack((audio, audio * 0.5))


class WaveformDataTests(unittest.TestCase):
    def test_min_max_follow_the_samples(self):
        audio = np.zeros(4000, dtype=np.float32)
        audio[:2000] = -0.25
        audio[2000:] = 0.8
        lows, highs = peak_envelope(audio, 4)
        self.assertEqual(lows.size, 4)
        self.assertAlmostEqual(float(lows.min()), -0.25, places=5)
        self.assertAlmostEqual(float(highs.max()), 0.8, places=5)
        self.assertLess(float(lows[0]), 0.0)
        self.assertGreater(float(highs[-1]), 0.5)

    def test_a_spike_survives_a_coarser_display(self):
        audio = np.zeros(8000, dtype=np.float32)
        audio[100] = 1.0
        fine_lows, fine_highs = peak_envelope(audio, 4000)
        self.assertGreater(float(fine_highs.max()), 0.99)
        _lows, coarse = display_envelope(fine_lows, fine_highs, 16)
        self.assertGreater(float(coarse.max()), 0.99)
        self.assertLess(coarse.size, fine_highs.size)

    def test_envelope_cache_reuses_the_same_buffer(self):
        audio = np.sin(np.linspace(0, 40, 5000, dtype=np.float32))
        with patch("soniclab.core.waveform.peak_envelope", wraps=peak_envelope) as wrapped:
            first = cached_envelope(audio)
            second = cached_envelope(audio)
        self.assertIs(first, second)
        self.assertEqual(wrapped.call_count, 1)

    def test_loud_constant_is_taller_than_a_quiet_one(self):
        loud, _high = peak_envelope(np.full(1000, 0.9, dtype=np.float32), 8)
        quiet, _quiet_high = peak_envelope(np.full(1000, 0.05, dtype=np.float32), 8)
        self.assertGreater(float(np.max(np.abs(loud))), float(np.max(np.abs(quiet))))


class BeatTests(unittest.TestCase):
    def test_click_train_lands_on_the_clicks(self):
        audio = _clicks()
        result = analyze_beats(audio, OUTPUT_RATE)
        self.assertGreaterEqual(len(result.beats), 4)
        self.assertGreater(result.tempo, 100)
        self.assertLess(result.tempo, 140)
        grid = np.arange(0.5, 8.0, 0.5)
        for beat in result.beats:
            self.assertLess(beat, 8.0)
            self.assertLess(float(np.min(np.abs(grid - beat))), 0.08)

    def test_silence_has_no_invented_grid(self):
        silence = np.zeros((OUTPUT_RATE, 2), dtype=np.float32)
        result = analyze_beats(silence, OUTPUT_RATE)
        self.assertEqual(result.beats, ())
        self.assertEqual(result.tempo, 0.0)

    def test_analysis_is_cached(self):
        audio = _clicks(seconds=4.0)
        first = analyze_beats(audio, OUTPUT_RATE)
        with patch("soniclab.core.beats.aubio.tempo", side_effect=AssertionError("analyzed again")):
            second = analyze_beats(audio, OUTPUT_RATE)
        self.assertEqual(first, second)


class ClipRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = TimelineView()

    def tearDown(self):
        self.view.clear_scene()
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def _show(self, track):
        self.view.rebuild([track], lambda *_args: None, 120, "4/4")
        return next(item for item in self.view.scene_obj.items() if isinstance(item, ClipItem))

    def test_waveform_is_filled_peak_data_not_line_items(self):
        audio = np.full((OUTPUT_RATE, 2), 0.7, dtype=np.float32)
        audio[::2, :] = -0.4
        clip = self._show(
            {"name": "Bass", "start": 0.0, "duration": 1.0, "_audio": audio, "_beats": ()}
        )
        lows, highs = clip.envelope
        self.assertGreater(lows.size, 50)
        self.assertLess(float(lows.min()), 0.0)
        self.assertGreater(float(highs.max()), 0.5)
        self.assertFalse(any(isinstance(item, QGraphicsLineItem) for item in clip.childItems()))
        self.view.resize(900, 320)
        self.view.show()
        self.view.grab()

    def test_markers_use_detected_times_and_can_be_hidden(self):
        audio = np.zeros((OUTPUT_RATE * 2, 2), dtype=np.float32)
        track = {
            "name": "Hits",
            "start": 0.0,
            "duration": 2.0,
            "_audio": audio,
            "_beats": (0.2, 0.55, 1.4),
            "_tempo": 90.0,
        }
        clip = self._show(track)
        marks = clip.beat_markers()
        self.assertEqual([round(time, 2) for time, _bar in marks], [0.2, 0.55, 1.4])
        self.assertTrue(marks[0][1])
        self.assertFalse(marks[1][1])
        self.assertEqual(track["_beats"], (0.2, 0.55, 1.4))
        self.view.set_show_beats(False)
        self.assertEqual(clip.beat_markers(), [])

    def test_zooming_keeps_the_peak(self):
        audio = np.zeros((OUTPUT_RATE, 2), dtype=np.float32)
        audio[100, :] = 1.0
        track = {"name": "Spike", "start": 0.0, "duration": 1.0, "_audio": audio, "_beats": ()}
        wide = self._show(track)
        self.assertGreater(float(wide.envelope[1].max()), 0.99)
        self.view.pixels_per_second = 360
        self.view.rebuild([track], lambda *_args: None, 120, "4/4")
        closer = next(item for item in self.view.scene_obj.items() if isinstance(item, ClipItem))
        self.assertGreater(closer.envelope[0].size, wide.envelope[0].size)
        self.assertGreater(float(closer.envelope[1].max()), 0.99)

    def test_playhead_strip_does_not_rebuild_the_whole_waveform(self):
        audio = np.sin(np.linspace(0, 400, OUTPUT_RATE * 20, dtype=np.float32))
        audio = np.column_stack((audio, audio))
        clip = self._show(
            {"name": "Long", "start": 0.0, "duration": 20.0, "_audio": audio, "_beats": ()}
        )
        self.assertEqual(clip._upper.size, int(clip.rect().width()))
        image = QImage(32, 80, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        started = time.perf_counter()
        for origin in range(0, 1200, 12):
            clip._paint_wave(painter, origin, origin + 8)
        painter.end()
        self.assertLess(time.perf_counter() - started, 0.15)


if __name__ == "__main__":
    unittest.main()
