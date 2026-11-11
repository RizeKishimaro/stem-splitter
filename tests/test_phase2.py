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
from PySide6.QtWidgets import QApplication, QMessageBox

from soniclab.core.analyze import band_levels, scope_window, spectrogram_column, spectrum_bars
from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.history import Edit, History
from soniclab.core.recorder import Recorder, assemble_blocks
from soniclab.core.ring_buffer import StereoRing
from soniclab.ui.main_window import MainWindow
from soniclab.ui.visualizer import Visualizer


def _sine(frequency, seconds=0.1, rate=OUTPUT_RATE):
    count = int(seconds * rate)
    phase = 2.0 * np.pi * frequency * np.arange(count) / rate
    column = np.sin(phase).astype(np.float32)
    return np.column_stack((column, column * 0.5))


class RingBufferTests(unittest.TestCase):
    def test_wrap_returns_the_newest_frames_in_order(self):
        ring = StereoRing(4)
        ring.push(np.arange(6, dtype=np.float32).reshape(3, 2))
        ring.push(np.array([[8, 9], [10, 11]], dtype=np.float32))
        latest = ring.latest(4)
        self.assertTrue(
            np.array_equal(latest, np.array([[2, 3], [4, 5], [8, 9], [10, 11]], dtype=np.float32))
        )
        self.assertEqual(len(ring.latest(0)), 0)


class AnalysisTests(unittest.TestCase):
    def test_low_tone_is_louder_in_bass_than_in_highs(self):
        bands = dict((name, level) for name, _low, _high, level in band_levels(_sine(100), OUTPUT_RATE))
        self.assertGreater(bands["Bass"], bands["High"])
        self.assertGreater(bands["Bass"], -40)

    def test_one_kilohertz_peaks_in_the_matching_bar(self):
        levels, edges = spectrum_bars(_sine(1000), OUTPUT_RATE, bars=48)
        peak = int(np.argmax(levels))
        self.assertLessEqual(edges[peak], 1000)
        self.assertGreater(edges[peak + 1], 1000)
        self.assertGreater(levels[peak], -20)

    def test_scope_window_is_the_real_tail(self):
        audio = _sine(440, seconds=0.05)
        window = scope_window(audio, OUTPUT_RATE, 0.02)
        self.assertTrue(np.allclose(window, audio[-len(window) :]))

    def test_spectrogram_column_matches_the_spectrum(self):
        audio = _sine(1000)
        column = spectrogram_column(audio, OUTPUT_RATE, rows=32)
        levels, _edges = spectrum_bars(audio, OUTPUT_RATE, bars=32)
        self.assertTrue(np.allclose(column, levels))

    def test_spectrum_stays_cheap(self):
        audio = _sine(440, seconds=0.05)
        started = time.perf_counter()
        for _ in range(100):
            spectrum_bars(audio, OUTPUT_RATE)
        self.assertLess(time.perf_counter() - started, 1.0)


class EngineTapTests(unittest.TestCase):
    def _render(self, engine, frames=256):
        out = np.zeros((frames, 2), dtype=np.float32)
        engine._playing = True
        engine._callback(out, frames, None, None)
        return out

    def test_offset_reads_the_later_part_of_the_file(self):
        audio = np.zeros((OUTPUT_RATE, 2), dtype=np.float32)
        audio[int(0.25 * OUTPUT_RATE) :] = 0.5
        engine = AudioEngine()
        engine.add_clip(audio, 0.0)
        heard = self._render(engine)
        self.assertTrue(np.allclose(heard, 0.0))

        engine.seek(0)
        engine.set_clip_region(0, 0.0, 0.25, 0.5)
        heard = self._render(engine)
        self.assertTrue(np.allclose(heard, 0.5))

    def test_capture_does_not_change_the_mix_or_the_clock(self):
        engine = AudioEngine()
        engine.add_clip(np.full((OUTPUT_RATE, 2), 0.4, dtype=np.float32), 0.0)
        engine.set_capture(True)
        captured = self._render(engine)
        self.assertAlmostEqual(engine.position, 256 / OUTPUT_RATE, places=6)
        copied = engine.read_master(256)
        self.assertTrue(np.allclose(copied, captured))

        engine.seek(0)
        engine.set_capture(False)
        silent_tap = self._render(engine)
        self.assertTrue(np.allclose(silent_tap, captured))
        self.assertEqual(len(engine.read_master(256)), 0)
        self.assertAlmostEqual(engine.position, 256 / OUTPUT_RATE, places=6)

    def test_unity_master_is_unchanged_and_peaks_are_consumed(self):
        engine = AudioEngine()
        engine.add_clip(np.full((OUTPUT_RATE, 2), 0.4, dtype=np.float32), 0.0)
        unity = self._render(engine, 32)
        self.assertTrue(np.allclose(unity, 0.4))
        master, tracks = engine.take_peaks()
        self.assertGreater(master, 0.3)
        self.assertGreater(tracks[0], 0.3)
        master, tracks = engine.take_peaks()
        self.assertEqual(master, 0.0)
        self.assertEqual(tracks[0], 0.0)

        engine.seek(0)
        engine.set_output_gain(0.5)
        quieter = self._render(engine, 32)
        self.assertTrue(np.allclose(quieter, 0.2))


class HistoryTests(unittest.TestCase):
    def test_push_does_not_redo_and_undo_restores(self):
        state = {"value": 1}
        history = History()
        history.push(Edit(lambda: state.update(value=1), lambda: state.update(value=2)))
        self.assertEqual(state["value"], 1)
        self.assertTrue(history.undo())
        self.assertEqual(state["value"], 1)
        state["value"] = 2
        history.push(Edit(lambda: state.update(value=1), lambda: state.update(value=2)))
        self.assertTrue(history.undo())
        self.assertEqual(state["value"], 1)
        self.assertTrue(history.redo())
        self.assertEqual(state["value"], 2)


class RecorderTests(unittest.TestCase):
    def test_assemble_blocks_selects_channel_without_a_device(self):
        blocks = [np.array([[0.1, 0.8], [0.2, 0.7]], dtype=np.float32)]
        stereo = assemble_blocks(blocks, 0, True)
        mono = assemble_blocks(blocks, 1, False)
        self.assertTrue(np.allclose(stereo[:, 1], [0.8, 0.7]))
        self.assertTrue(np.allclose(mono[:, 0], [0.8, 0.7]))
        self.assertTrue(np.allclose(mono[:, 1], mono[:, 0]))
        self.assertEqual(len(assemble_blocks([], 0, True)), 0)

    def test_poll_returns_a_result_once(self):
        recorder = Recorder()
        recorder._publish(("error", "device missing", None))
        self.assertEqual(recorder.poll()[0], "error")
        self.assertIsNone(recorder.poll())


class VisualizerWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_modes_source_and_freeze_use_the_supplied_buffer(self):
        widget = Visualizer()
        widget.resize(640, 320)
        widget.show()
        tone = _sine(1000, seconds=0.08)
        for _ in range(6):
            widget.analyze(tone)
        self.assertGreater(float(np.max(widget._display)), -25)
        widget._set_mode("scope")
        widget._trigger = False
        widget.analyze(tone)
        expected = scope_window(tone, OUTPUT_RATE, widget._scope_seconds)
        self.assertTrue(np.allclose(widget._scope, expected))
        widget._set_mode("spectrogram")
        before = widget._gram.values[:, -1].copy()
        widget.analyze(tone)
        self.assertFalse(np.allclose(widget._gram.values[:, -1], before))
        widget.grab()
        widget._set_mode("spectrum")
        widget.grab()
        widget._set_mode("scope")
        widget.grab()

        widget.set_sources(["Tone"])
        widget.source.setCurrentIndex(1)
        self.assertEqual(widget._source, 0)
        self.assertTrue(np.allclose(widget._display, -90))

        samples = {"audio": tone}

        def reader(_source, _count):
            return samples["audio"]

        widget.set_reader(reader)
        widget._set_mode("spectrum")
        widget._frozen = True
        frozen = widget._display.copy()
        widget._tick()
        self.assertTrue(np.allclose(widget._display, frozen))
        widget._set_frozen(False)
        for _ in range(6):
            widget._tick()
        self.assertGreater(float(np.max(widget._display)), -25)
        widget.close()
        widget.deleteLater()
        self.app.processEvents()


class EditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()
        audio = np.full((OUTPUT_RATE, 2), 0.4, dtype=np.float32)
        self.window.project.tracks.append(
            {
                "name": "Tone",
                "path": None,
                "start": 0.0,
                "duration": 1.0,
                "offset": 0.0,
                "_audio": audio,
            }
        )
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.window._select_track(0)

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_trim_split_duplicate_and_undo(self):
        self.window._trim_clip(0, 0.2, 0.5)
        track = self.window.project.tracks[0]
        self.assertAlmostEqual(track["start"], 0.2, places=3)
        self.assertAlmostEqual(track["offset"], 0.2, places=3)
        self.assertAlmostEqual(track["duration"], 0.5, places=3)
        self.window.engine.seek(0.2)
        self.window.engine._playing = True
        heard = np.zeros((32, 2), dtype=np.float32)
        self.window.engine._callback(heard, 32, None, None)
        self.assertTrue(np.allclose(heard, 0.4, atol=1e-4))

        self.window._undo()
        self.assertAlmostEqual(self.window.project.tracks[0]["duration"], 1.0, places=3)
        self.window._redo()
        self.assertAlmostEqual(self.window.project.tracks[0]["duration"], 0.5, places=3)

        self.window.engine.seek(0.4)
        self.window._split_selected()
        self.assertEqual(len(self.window.project.tracks), 2)
        self.assertAlmostEqual(self.window.project.tracks[0]["duration"], 0.2, places=3)
        self.assertAlmostEqual(self.window.project.tracks[1]["start"], 0.4, places=3)
        self.window._undo()
        self.assertEqual(len(self.window.project.tracks), 1)

        self.window._duplicate_selected()
        self.assertEqual(len(self.window.project.tracks), 2)
        self.assertIn("copy", self.window.project.tracks[1]["name"])
        self.window._move_track(1)
        self.assertEqual(self.window.project.tracks[0]["name"], "Tone copy")
        self.window._undo()
        self.assertEqual(self.window.project.tracks[0]["name"], "Tone")

        for item in self.window.timeline.scene_obj.items():
            if item.__class__.__name__ == "ClipItem":
                item.setSelected(True)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._delete_track()
        self.assertEqual(self.window.project.tracks, [])
        self.window._undo()
        self.assertEqual(len(self.window.project.tracks), 2)

    def test_snap_master_meters_and_visualizer_dock(self):
        self.assertTrue(self.window.timeline.snap_to_grid)
        self.window.snap_btn.setChecked(False)
        self.assertFalse(self.window.timeline.snap_to_grid)
        self.assertEqual(self.window.visualizer_dock.objectName(), "visualizerDock")
        self.assertIn("Visualizer", self.window.visualizer_dock.toggleViewAction().text())

        self.window._set_gain(-1, 0.5)
        self.window._commit_edit()
        self.window.engine.seek(0.1)
        self.window.engine._playing = True
        mixed = np.zeros((32, 2), dtype=np.float32)
        self.window.engine._callback(mixed, 32, None, None)
        self.assertTrue(np.allclose(mixed, 0.2, atol=1e-4))
        self.window._refresh_transport()
        self.assertGreater(self.window.mixer._master.meter._display, 0.15)
        self.window._undo()
        self.assertAlmostEqual(self.window.engine._output_gain, 1.0)

        self.window.visualizer._set_mode("spectrum")
        tone = _sine(200, seconds=0.08)
        for _ in range(8):
            self.window.visualizer.analyze(tone)
        self.assertGreater(float(np.max(self.window.visualizer._display)), -30)
        self.window.visualizer._set_frozen(True)
        self.assertEqual(self.window.visualizer.freeze_btn.text(), "Unfreeze")
        held = self.window.visualizer._display.copy()
        self.window.visualizer._tick()
        self.assertTrue(np.allclose(self.window.visualizer._display, held))

    def test_recorded_audio_becomes_a_clip_and_can_be_undone(self):
        self.window.project.tracks[0]["armed"] = True
        self.window.project.tracks[0]["empty"] = True
        self.window.project.tracks[0]["duration"] = 0.0
        self.window._armed_at_record = [0]
        self.window._record_started_at = 1.25
        audio = np.full((800, 2), 0.3, dtype=np.float32)
        self.window._place_recording(os.path.join(self.window._media_dir(), "Take 01.wav"), audio)
        placed = self.window.project.tracks[0]
        self.assertFalse(placed.get("empty", False))
        self.assertAlmostEqual(placed["start"], 1.25, places=3)
        self.assertEqual(len(placed["_audio"]), 800)
        self.window._undo()
        self.assertTrue(self.window.project.tracks[0].get("empty"))

    def test_record_without_an_armed_track_does_not_open_a_device(self):
        self.window._toggle_record()
        self.assertFalse(self.window.recorder.recording)
        self.assertIn("Arm a track", self.window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
