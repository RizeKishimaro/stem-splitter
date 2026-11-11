import os
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.core.downbeat_loop import (
    DEFAULT_LOOP_SHORTCUTS,
    arrangement_downbeats,
    downbeat_loop_region,
    loop_shortcut,
)
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import HEADER_WIDTH, PIXELS_PER_SECOND, TimelineView


DOWNBEATS = (
    (0.0, 1),
    (0.5, 2),
    (1.0, 1),
    (1.5, 3),
    (2.0, 1),
    (2.5, 4),
    (3.0, 1),
    (4.0, 1),
    (5.0, 1),
)


class _Settings:
    def __init__(self, values=None):
        self.values = values or {}

    def value(self, key, default=""):
        return self.values.get(key, default)


class _Jobs:
    def __init__(self, events):
        self.events = tuple(events)
        self.calls = []

    def events_for(self, path):
        self.calls.append(path)
        return self.events

    def poll(self):
        return None

    def submit(self, _path):
        return "queued"

    def result_for(self, _path):
        return None

    def cancel(self):
        return None


class RegionTests(unittest.TestCase):
    def test_each_length_uses_following_downbeats(self):
        expected = {
            1: (1.0, 2.0),
            2: (1.0, 3.0),
            3: (1.0, 4.0),
            4: (1.0, 5.0),
        }
        source = list(DOWNBEATS)
        for intervals, region in expected.items():
            with self.subTest(intervals=intervals):
                self.assertEqual(
                    downbeat_loop_region(source, 1.1, intervals, 0.0, 0.0),
                    region,
                )
        self.assertEqual(source, list(DOWNBEATS))

    def test_clip_start_and_offset_shift_both_edges(self):
        events = ((2.0, 1), (3.5, 1), (5.0, 1))
        region = downbeat_loop_region(events, 11.6, 1, clip_start=10.0, clip_offset=2.0)
        self.assertEqual(region, (11.5, 13.0))
        self.assertEqual(
            arrangement_downbeats(events, 10.0, 2.0),
            (10.0, 11.5, 13.0),
        )

    def test_alignment_shifts_both_edges_together(self):
        events = [(1.0, 1), (2.0, 1), (3.0, 1)]
        before = list(events)
        region = downbeat_loop_region(events, 1.2, 2, 0.0, 0.0, alignment=0.25)
        self.assertEqual(region, (1.25, 3.25))
        self.assertEqual(events, before)

    def test_tie_keeps_the_earlier_downbeat(self):
        self.assertEqual(
            downbeat_loop_region(((0.0, 1), (2.0, 1)), 1.0, 1, 0.0, 0.0),
            (0.0, 2.0),
        )

    def test_missing_and_insufficient_downbeats_make_no_region(self):
        self.assertIsNone(downbeat_loop_region((), 1.0, 1, 0.0, 0.0))
        self.assertIsNone(downbeat_loop_region(((0.5, 2), (1.0, 3)), 0.5, 1, 0.0, 0.0))
        self.assertIsNone(downbeat_loop_region(((0.0, 1), (1.0, 1)), 0.1, 4, 0.0, 0.0))
        self.assertIsNone(downbeat_loop_region(DOWNBEATS, 5.0, 1, 0.0, 0.0))

    def test_shortcut_defaults_and_overrides(self):
        self.assertEqual(
            [loop_shortcut(intervals) for intervals in (1, 2, 3, 4)],
            ["Ctrl+Alt+1", "Ctrl+Alt+2", "Ctrl+Alt+3", "Ctrl+Alt+4"],
        )
        settings = _Settings({"shortcuts/downbeatLoop2": "Ctrl+Alt+8"})
        self.assertEqual(loop_shortcut(2, settings), "Ctrl+Alt+8")
        self.assertEqual(loop_shortcut(1, settings), DEFAULT_LOOP_SHORTCUTS[1])
        self.assertEqual(loop_shortcut(3, _Settings({"shortcuts/downbeatLoop3": "  "})), "Ctrl+Alt+3")


class EngineLoopTests(unittest.TestCase):
    def test_callback_wraps_on_detected_bounds(self):
        engine = AudioEngine()
        audio = np.zeros((1000, 2), dtype=np.float32)
        samples = (np.arange(1000, dtype=np.float32) - 500.0) / 1000.0
        audio[:, 0] = samples
        audio[:, 1] = samples
        engine.add_clip(audio, 0.0)
        self.assertTrue(engine.set_loop(200 / OUTPUT_RATE, 400 / OUTPUT_RATE))
        engine.seek(350 / OUTPUT_RATE)
        engine._playing = True
        out = np.zeros((100, 2), dtype=np.float32)
        engine._callback(out, 100, None, None)
        expected = np.concatenate((samples[350:400], samples[200:250]))
        self.assertTrue(np.allclose(out[:, 0], expected))
        self.assertEqual(engine._position, 250)
        engine.clear_loop()
        linear = np.zeros((20, 2), dtype=np.float32)
        engine._callback(linear, 20, None, None)
        self.assertTrue(np.allclose(linear[:, 0], samples[250:270]))
        self.assertEqual(engine._position, 270)
        engine.close()

    def test_invalid_loop_is_rejected(self):
        engine = AudioEngine()
        self.assertFalse(engine.set_loop(2.0, 2.0))
        self.assertFalse(engine.set_loop(3.0, 1.0))
        self.assertIsNone(engine.loop)
        engine.close()


class TimelineLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_region_follows_seconds_through_rebuild(self):
        view = TimelineView()
        audio = np.zeros((128, 2), dtype=np.float32)
        tracks = [{"name": "Audio", "start": 0.0, "duration": 1.0, "_audio": audio}]
        view.rebuild(tracks, lambda *_args: None, 120, "4/4")
        view.set_loop_region(2.0, 4.0)
        band = view._loop_region_item
        self.assertIsInstance(band, QGraphicsRectItem)
        self.assertAlmostEqual(band.rect().x(), HEADER_WIDTH + 2.0 * PIXELS_PER_SECOND)
        self.assertAlmostEqual(band.rect().width(), 2.0 * PIXELS_PER_SECOND)
        view.pixels_per_second = 180.0
        view.rebuild(tracks, lambda *_args: None, 90, "3/4")
        band = view._loop_region_item
        self.assertAlmostEqual(band.rect().x(), HEADER_WIDTH + 2.0 * 180.0)
        self.assertAlmostEqual(band.rect().width(), 2.0 * 180.0)
        self.assertEqual(view.snap_to_grid, True)
        view.set_loop_region(None, None)
        self.assertIsNone(view._loop_region_item)
        view.close()
        view.deleteLater()
        self.app.processEvents()


class WindowLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_selected_clip_loop_leaves_session_grid_alone(self):
        events = [(1.0, 1), (2.5, 2), (3.0, 1), (4.5, 1), (6.0, 1), (7.5, 1), (9.0, 1)]
        jobs = _Jobs(events)
        self.window.beat_jobs = jobs
        audio = np.zeros((128, 2), dtype=np.float32)
        self.window.project.tracks.append(
            {
                "name": "Song",
                "path": "/tmp/soniclab-downbeat-loop.wav",
                "start": 4.0,
                "offset": 1.0,
                "duration": 8.0,
                "_audio": audio,
            }
        )
        self.window.engine.add_clip(audio, 4.0)
        self.window.engine.set_clip_region(0, 4.0, 1.0, 8.0)
        self.window._refresh_timeline()
        self.window._select_track(0)
        self.window.engine.seek(6.1)
        bpm = self.window.project.bpm
        signature = list(self.window.project.time_signature)
        self.window.loop_actions[1].trigger()
        self.assertEqual(jobs.calls[-1], "/tmp/soniclab-downbeat-loop.wav")
        self.assertEqual(events, [(1.0, 1), (2.5, 2), (3.0, 1), (4.5, 1), (6.0, 1), (7.5, 1), (9.0, 1)])
        start, end = self.window.engine.loop
        self.assertAlmostEqual(start, 6.0, places=4)
        self.assertAlmostEqual(end, 9.0, places=4)
        self.assertAlmostEqual(self.window.timeline._loop_region[0], 6.0)
        self.assertAlmostEqual(self.window.timeline._loop_region[1], 9.0)
        self.assertTrue(self.window.loop_btn.isChecked())
        self.assertEqual(self.window.project.bpm, bpm)
        self.assertEqual(self.window.project.time_signature, signature)
        self.assertTrue(self.window.timeline.snap_to_grid)
        self.window._loop_downbeats(5)
        self.assertAlmostEqual(self.window.engine.loop[0], 6.0, places=4)
        self.assertAlmostEqual(self.window.engine.loop[1], 9.0, places=4)
        self.assertIn("Not enough downbeats", self.window.statusBar().currentMessage())

    def test_shortcuts_do_not_repeat_existing_bindings(self):
        sequences = []
        for action in self.window.findChildren(QAction):
            text = action.shortcut().toString()
            if text:
                sequences.append(text)
        for sequence in DEFAULT_LOOP_SHORTCUTS.values():
            self.assertEqual(sequences.count(sequence), 1)
        self.window._loop_downbeats(1)
        self.assertEqual(self.window.statusBar().currentMessage(), "Select a clip to loop")
