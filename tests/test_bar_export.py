import os
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine, render_clips
from soniclab.core.bar_export import FORMATS_BY_KEY, bar_frames, default_name, write_audio
from soniclab.ui.main_window import MainWindow

RATE = OUTPUT_RATE


def _noise(seconds, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, (int(seconds * RATE), 2)).astype(np.float32)


def _lag(decoded, original):
    """Offset of ``decoded`` against ``original`` by cross-correlation."""
    window = slice(4000, 24000)
    return max(
        range(-2000, 2001),
        key=lambda k: abs(float(np.dot(decoded[4000 + k : 24000 + k, 0], original[window, 0]))),
    )


class RenderTests(unittest.TestCase):
    def test_offline_render_matches_playback_exactly(self):
        engine = AudioEngine()
        engine._capture_enabled = False
        for index in range(3):
            engine.add_clip(_noise(1.5, index), 0.1 * index)
        engine.set_clip_region(1, 0.2, 0.05, 0.9, 0.1, 0.2, 0.8)
        engine.set_clip_pan(0, -0.4)
        engine.set_clip_gain(2, 1.7)
        engine.set_clip_solo(0, True)
        engine.set_clip_solo(1, True)
        engine.set_output_gain(0.9)
        engine._playing = True
        engine._position = 1234
        blocks = []
        for _ in range(100):
            out = np.zeros((512, 2), np.float32)
            engine._callback(out, 512, None, None)
            blocks.append(out)
        live = np.concatenate(blocks)
        np.testing.assert_array_equal(engine.render(1234, len(live)), live)

    def test_render_outside_clips_is_silent(self):
        clip = {"audio": _noise(1.0, 0), "start": RATE, "offset": 0, "length": RATE,
                "gain": 1.0, "clip_gain": 1.0, "pan": 0.0, "muted": False, "solo": False}
        out = render_clips([clip], 0, RATE + 10)
        self.assertEqual(out.shape, (RATE + 10, 2))
        self.assertFalse(out[:RATE].any())
        np.testing.assert_array_equal(out[RATE:], clip["audio"][:10])


class FileTests(unittest.TestCase):
    def test_bar_frames_round_once(self):
        lines = [0.3, 2.5, 4.70001]
        self.assertEqual(bar_frames(lines, 0, 1, 2.0, RATE), (13230, 207270))
        self.assertEqual(bar_frames(lines, 1, 0, 2.0, RATE), (13230, 207270))
        self.assertEqual(default_name("Song", 2, 5, "Bass"), "Song Bass bars 3-6")
        self.assertEqual(default_name("A/B", 0, 0), "A_B bar 1")

    def test_every_format_keeps_length_and_timing(self):
        audio = _noise(1.37, 7)
        with tempfile.TemporaryDirectory() as folder:
            for key, fmt in FORMATS_BY_KEY.items():
                result = write_audio(Path(folder) / f"take {key}", audio, RATE, fmt)
                self.assertEqual(result.path.suffix, fmt.suffix)
                decoded, rate = sf.read(result.path, dtype="float32", always_2d=True)
                self.assertEqual(rate, RATE, key)
                self.assertEqual(len(decoded), len(audio), key)
                self.assertEqual(_lag(decoded, audio), 0, key)
                error = float(np.max(np.abs(decoded - audio)))
                limit = {"wav32": 0.0, "wav24": 1e-6, "wav16": 1e-4}.get(key, 0.5)
                self.assertLessEqual(error, limit, key)
            leftovers = [name for name in os.listdir(folder) if name.startswith(".export-")]
            self.assertEqual(leftovers, [])

    def test_full_scale_and_clipping(self):
        audio = np.ones((100, 2), np.float32) * 1.2
        with tempfile.TemporaryDirectory() as folder:
            result = write_audio(Path(folder) / "loud.wav", audio, RATE, FORMATS_BY_KEY["wav24"])
            decoded, _rate = sf.read(result.path, dtype="float32")
        self.assertEqual(result.clipped, 200)
        self.assertGreater(float(decoded.min()), 0.999)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.window = MainWindow()
        self.audio = {"Drums": _noise(10.0, 1), "Bass": _noise(10.0, 2)}
        for name, audio in self.audio.items():
            self.window.project.tracks.append(
                {"name": name, "path": None, "start": 0.0, "offset": 0.0, "duration": 10.0, "_audio": audio}
            )
            self.window.engine.add_clip(audio, 0.0)
        self.window.project.name = "Song"
        self.window.project.bars = [0.3, 2.5, 4.7, 6.9]
        self.window._refresh_timeline()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def test_mix_of_two_bars_is_sample_exact(self):
        target = os.path.join(self.folder.name, "out.wav")
        (result,) = self.window._export_bars(1, 2, "wav32", "mix", target)
        decoded, _rate = sf.read(result.path, dtype="float32")
        start, end = round(2.5 * RATE), round(6.9 * RATE)
        expected = np.clip(self.audio["Drums"][start:end] + self.audio["Bass"][start:end], -1, 1)
        np.testing.assert_array_equal(decoded, expected)
        self.assertIn("bars 2–3", self.window.statusBar().currentMessage())

    def test_each_track_skips_muted_and_uses_its_name(self):
        self.window.project.tracks[1]["muted"] = True
        self.window.engine.set_clip_muted(1, True)
        results = self.window._export_bars(0, 0, "ogg_q10", "each", self.folder.name)
        self.assertEqual([result.path.name for result in results], ["Song Drums bar 1.ogg"])
        decoded, _rate = sf.read(results[0].path, dtype="float32")
        self.assertEqual(len(decoded), round(2.5 * RATE) - round(0.3 * RATE))

    def test_shift_click_range_becomes_the_export_default(self):
        self.window._select_bars(1, 2)
        self.assertEqual(self.window._loop_bounds, (2.5, 6.9))
        self.assertEqual(self.window._selected_bars(), (1, 2))


if __name__ == "__main__":
    unittest.main()
