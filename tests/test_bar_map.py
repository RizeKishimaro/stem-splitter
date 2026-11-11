import os
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from soniclab.core.bar_map import (
    bar_index,
    bar_span,
    bars_from_downbeats,
    beat_lines,
    extend,
    format_clock,
    grid_bars,
    move_line,
    parse_clock,
    position,
    set_span,
)
from soniclab.core.project import Project
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import HEADER_WIDTH, PIXELS_PER_SECOND


class _Jobs:
    def __init__(self, events):
        self.events = tuple(events)

    def events_for(self, path):
        return self.events if path == "/tmp/soniclab-drums.wav" else ()

    def result_for(self, path):
        return None

    def submit(self, path):
        return "missing"

    def poll(self):
        return None

    def cancel(self):
        pass


class BarMapTests(unittest.TestCase):
    def test_grid_and_downbeats(self):
        self.assertEqual(grid_bars(120, 4, 4, 5.0)[:4], [0.0, 2.0, 4.0, 6.0])
        # Clip at 1.0 s plays its file from 0.5 s; the 0.2 s downbeat is cut off.
        events = [(0.2, 1), (0.5, 1), (1.0, 2), (2.6, 1), (4.5, 1)]
        self.assertEqual(bars_from_downbeats(events, 1.0, 0.5), [1.0, 3.1, 5.0])

    def test_beats_split_each_bar_evenly(self):
        marks = beat_lines([0.0, 2.0, 5.0], 4)
        self.assertEqual(
            marks,
            [(0.0, True), (0.5, False), (1.0, False), (1.5, False),
             (2.0, True), (2.75, False), (3.5, False), (4.25, False), (5.0, True)],
        )

    def test_positions_past_the_map_repeat_the_last_bar(self):
        lines = [1.0, 3.0, 6.0]
        self.assertEqual(extend(lines, 10.0, 2.0), [1.0, 3.0, 6.0, 9.0, 12.0])
        self.assertEqual(bar_index(lines, 7.0, 2.0), 2)
        self.assertEqual(bar_span(lines, 3, 2.0), (9.0, 12.0))
        self.assertEqual(position(lines, 4.5, 4, 2.0), (2, 3))
        self.assertEqual(position(lines, 0.5, 4, 2.0), (0, 4))

    def test_moving_and_typing_bars(self):
        lines = [0.0, 2.0, 4.0, 6.0]
        self.assertEqual(move_line(lines, 1, 2.5), [0.0, 2.5, 4.0, 6.0])
        self.assertEqual(move_line(lines, 1, 9.0), [0.0, 3.9, 4.0, 6.0])
        self.assertEqual(move_line(lines, 1, 2.5, ripple=True), [0.0, 2.5, 4.5, 6.5])
        self.assertEqual(set_span(lines, 1, 1.8, 4.2, 2.0), [0.0, 1.8, 4.2, 6.0])
        self.assertEqual(set_span(lines, 4, 8.0, 9.5, 2.0), [0.0, 2.0, 4.0, 6.0, 8.0, 9.5])
        with self.assertRaises(ValueError):
            set_span(lines, 1, 3.0, 2.0, 2.0)
        with self.assertRaises(ValueError):
            set_span(lines, 1, 2.0, 6.5, 2.0)

    def test_clock(self):
        self.assertEqual(format_clock(3725.456), "01:02:05.46")
        self.assertAlmostEqual(parse_clock("01:02:05.46"), 3725.46)
        self.assertAlmostEqual(parse_clock("2:05.5"), 125.5)
        self.assertAlmostEqual(parse_clock("4.25"), 4.25)
        with self.assertRaises(ValueError):
            parse_clock("1:xx")


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        # Drum downbeats every 2.2 s from 0.3 s: slower than the 120 BPM grid.
        events = [(0.3 + 2.2 * bar + 0.55 * beat, beat + 1) for bar in range(5) for beat in range(4)]
        self.window.beat_jobs = _Jobs(events)
        audio = np.zeros((128, 2), dtype=np.float32)
        for name, path in (("Drums", "/tmp/soniclab-drums.wav"), ("Bass", "/tmp/soniclab-bass.wav")):
            self.window.project.tracks.append(
                {"name": name, "path": path, "start": 0.0, "offset": 0.0, "duration": 12.0, "_audio": audio}
            )
            self.window.engine.add_clip(audio, 0.0)
        self.window._refresh_timeline()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_bars_follow_the_drum_downbeats(self):
        window = self.window
        self.assertEqual(window.timeline.bar_map, [])
        window._select_track(0)
        window._toggle_drum_track()
        expected = [round(0.3 + 2.2 * bar, 6) for bar in range(5)]
        self.assertEqual([round(x, 6) for x in window.timeline.bar_map], expected)
        self.assertEqual(window._musical_position(0.3 + 2.2 + 0.6), "002.2")
        self.assertEqual(window.project.bars, [])

    def test_drag_edit_undo_and_save(self):
        window = self.window
        view = window.timeline
        # Drag the bar-2 line (2.0 s on the 120 BPM grid) to 2.5 s.
        pps = PIXELS_PER_SECOND
        start = view.mapFromScene(QPointF(HEADER_WIDTH + 2.0 * pps, 10))
        end = view.mapFromScene(QPointF(HEADER_WIDTH + 2.5 * pps, 10))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        self.app.processEvents()
        self.assertAlmostEqual(window.project.bars[1], 2.5, places=1)
        self.assertEqual(window.project.bars[2], 4.0)
        self.assertTrue(window._apply_bar_span(2, "00:00:04.10", "00:00:06.30"))
        self.assertAlmostEqual(window.project.bars[2], 4.1)
        self.assertAlmostEqual(window.project.bars[3], 6.3)
        self.assertFalse(window._apply_bar_span(2, "00:00:06.00", "00:00:05.00"))
        with tempfile.TemporaryDirectory() as folder:
            window.current_path = os.path.join(folder, "song.soniclab")
            window._save_project()
            saved = Project.load(window.current_path)
        self.assertAlmostEqual(saved.bars[3], 6.3)
        window._undo()
        self.assertEqual(window.project.bars[2], 4.0)
        window._undo()
        self.assertEqual(window.project.bars, [])
        self.assertEqual(window.timeline.bar_map, [])

    def test_add_remove_and_reset(self):
        window = self.window
        window.engine.seek(3.0)
        window._add_bar_line()
        self.assertIn(3.0, window.project.bars)
        window.engine.seek(3.1)
        window._remove_bar_line()
        self.assertNotIn(3.0, window.project.bars)
        window._reset_bars()
        self.assertEqual(window.project.bars, [])


if __name__ == "__main__":
    unittest.main()
