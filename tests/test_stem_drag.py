"""Only stem blocks drag; canvases, tracks, scroll and panels stay put while
blocks and tracks move between canvases or audio is imported."""

import os
import tempfile
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
import soundfile as sf
from PySide6.QtCore import QAbstractAnimation, QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import PIXELS_PER_SECOND, ClipItem


class _Jobs:
    def events_for(self, path):
        return ()

    def result_for(self, path):
        return None

    def submit(self, path):
        return "missing"

    def poll(self):
        return None

    def cancel(self):
        pass


class StemDragTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs()
        self.window.resize(1400, 900)
        self.audio = {}
        for name, level in (("drums", 0.5), ("bass", 0.1), ("vocals", 0.2)):
            audio = np.full((OUTPUT_RATE * 8, 2), level, dtype=np.float32)
            self.audio[name] = audio
            self.window.project.tracks.append(
                {"name": name, "path": None, "start": 0.0, "duration": 8.0, "offset": 0.0, "_audio": audio}
            )
        self.window.project.bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.window.show()
        for _ in range(3):
            self.app.processEvents()
        self.main, self.second = self.window.timeline, self.window.timeline2

    def tearDown(self):
        self.window.engine.preview_stop_all()
        for timeline in (self.main, self.second):
            timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    # -- helpers

    def clips(self, view):
        return {
            item.clip_index: item
            for item in view.scene_obj.items()
            if isinstance(item, ClipItem) and isValid(item)
        }

    def names(self):
        return [track["name"] for track in self.window.project.tracks]

    def layout(self):
        """Everything that must not move: window, canvases, panels, scroll."""
        window = self.window
        widgets = (
            window.canvas_split, self.main, self.second, window.headers, window.headers2,
            window.mixer, window.statusBar(),
        )
        return (
            window.geometry().getRect(),
            [widget.geometry().getRect() for widget in widgets],
            window.canvas_split.sizes(),
            [view.horizontalScrollBar().value() for view in (self.main, self.second)],
            [view.verticalScrollBar().value() for view in (self.main, self.second)],
            [view.pixels_per_second for view in (self.main, self.second)],
        )

    def send(self, view, kind, at, buttons, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QMouseEvent(kind, QPointF(at), QPointF(view.viewport().mapToGlobal(at)),
                            Qt.MouseButton.LeftButton, buttons, modifiers)
        QApplication.sendEvent(view.viewport(), event)
        self.app.processEvents()

    def drag(self, source, scene_from, target, scene_to, modifiers=Qt.KeyboardModifier.NoModifier, before_release=None):
        start = source.mapFromScene(scene_from)
        end = source.viewport().mapFromGlobal(target.viewport().mapToGlobal(target.mapFromScene(scene_to)))
        middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
        left = Qt.MouseButton.LeftButton
        self.send(source, QEvent.Type.MouseButtonPress, start, left, modifiers)
        self.send(source, QEvent.Type.MouseMove, middle, left, modifiers)
        self.send(source, QEvent.Type.MouseMove, end, left, modifiers)
        if before_release is not None:
            before_release()
        self.send(source, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton, modifiers)
        for _ in range(3):
            self.app.processEvents()

    def grab_point(self, clip, seconds):
        return QPointF(clip.pos().x() + seconds * clip.pixels_per_second, clip.pos().y() + clip.rect().height() * 0.75)

    # -- what drags

    def test_a_left_drag_moves_one_block_and_nothing_else(self):
        clips = self.clips(self.main)
        drums, bass = clips[0], clips[1]
        bass_pos = QPointF(bass.pos())
        before = self.layout()
        header = self.window.headers._headers[0].geometry().getRect()
        grab = self.grab_point(drums, 2.5)  # bar 2
        self.drag(self.main, grab, self.main, QPointF(grab.x() + 2.0 * PIXELS_PER_SECOND, grab.y()))
        self.assertIn("drums · bar 2", self.names())
        piece = next(track for track in self.window.project.tracks if track["name"] == "drums · bar 2")
        self.assertAlmostEqual(piece["start"], 4.0, places=2)
        self.assertIs(piece["_audio"], self.audio["drums"])
        rest = [track for track in self.window.project.tracks if track["name"] == "drums"]
        self.assertEqual(sorted(round(track["start"], 3) for track in rest), [0.0, 4.0])  # bars 1 and 3-4 stay
        self.assertEqual(self.clips(self.main)[self.names().index("bass")].pos(), bass_pos)
        self.assertEqual(self.layout(), before)
        self.assertEqual(self.window.headers._headers[0].geometry().getRect(), header)

    def test_shift_drag_never_moves_a_whole_split_track(self):
        drums = self.clips(self.main)[0]
        self.assertFalse(drums.flags() & ClipItem.GraphicsItemFlag.ItemIsMovable)
        pos = QPointF(drums.pos())
        grab = self.grab_point(drums, 2.5)
        self.drag(self.main, grab, self.main, QPointF(grab.x() + 300, grab.y() + 90), Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])
        self.assertEqual(self.clips(self.main)[0].pos(), pos)
        self.assertEqual(self.window.project.tracks[0]["start"], 0.0)

    def test_empty_canvas_and_track_header_drags_move_nothing(self):
        before = self.layout()
        starts = [track["start"] for track in self.window.project.tracks]
        empty = QPointF(3.0 * PIXELS_PER_SECOND, self.main.sceneRect().height() - 20)
        self.drag(self.main, empty, self.main, QPointF(empty.x() + 200, empty.y() - 10))
        header = self.window.headers._headers[1]
        geometry = header.geometry().getRect()
        QTest.mousePress(header, Qt.MouseButton.LeftButton, pos=QPoint(100, 30))
        QTest.mouseMove(header, QPoint(160, 70))
        QTest.mouseRelease(header, Qt.MouseButton.LeftButton, pos=QPoint(160, 70))
        self.app.processEvents()
        self.assertEqual(header.geometry().getRect(), geometry)
        self.assertEqual([track["start"] for track in self.window.project.tracks], starts)
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])
        self.assertEqual(self.layout(), before)

    def test_play_button_plays_and_does_not_drag(self):
        with patch.object(self.window.engine, "_ensure_stream", lambda: None):
            bass = self.clips(self.main)[1]
            button = next(child for child in bass.childItems() if hasattr(child, "set_playing") and child.index == 1)
            at = self.main.mapFromScene(button.sceneBoundingRect().center())
            left = Qt.MouseButton.LeftButton
            self.send(self.main, QEvent.Type.MouseButtonPress, at, left)
            self.send(self.main, QEvent.Type.MouseButtonRelease, at, Qt.MouseButton.NoButton)
            self.assertTrue(self.window.engine.previews()[(1, 1)][0])
            self.assertTrue(button.playing)
            self.send(self.main, QEvent.Type.MouseButtonPress, at, left)
            self.send(self.main, QEvent.Type.MouseButtonRelease, at, Qt.MouseButton.NoButton)
            self.assertFalse(self.window.engine.previews()[(1, 1)][0])
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])

    # -- cancel and refuse

    def test_escape_cancels_a_block_drag_and_restores_it(self):
        drums = self.clips(self.main)[0]
        grab = self.grab_point(drums, 4.5)

        def escape():
            QApplication.sendEvent(self.main, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
            self.app.processEvents()

        self.drag(self.main, grab, self.second, QPointF(grab.x() + 100, 60), before_release=escape)
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])
        self.assertEqual(self.clips(self.second), {})
        self.assertEqual(self.window.project.tracks[0]["start"], 0.0)

    def test_a_drop_outside_both_canvases_keeps_the_block(self):
        drums = self.clips(self.main)[0]
        grab = self.grab_point(drums, 4.5)
        start = self.main.mapFromScene(grab)
        outside = self.main.viewport().mapFromGlobal(self.window.mixer.mapToGlobal(QPoint(40, 40)))
        left = Qt.MouseButton.LeftButton
        self.send(self.main, QEvent.Type.MouseButtonPress, start, left)
        self.send(self.main, QEvent.Type.MouseMove, QPoint(start.x() + 30, start.y()), left)
        self.send(self.main, QEvent.Type.MouseMove, outside, left)
        self.send(self.main, QEvent.Type.MouseButtonRelease, outside, Qt.MouseButton.NoButton)
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])
        self.assertEqual(self.window.project.tracks[0]["start"], 0.0)
        self.assertIs(self.window.project.tracks[0]["_audio"], self.audio["drums"])

    # -- transfers

    def test_transfer_keeps_audio_layout_and_scroll(self):
        window = self.window
        for view in (self.main, self.second):
            view.zoom_horizontal(3.0)
        self.app.processEvents()
        self.main.horizontalScrollBar().setValue(200)
        before = self.layout()
        drums = self.clips(self.main)[0]
        grab = self.grab_point(drums, 2.5)  # bar 2, on screen at this scroll
        self.drag(self.main, grab, self.second, QPointF(grab.x(), 60))
        piece = next(track for track in window.project.tracks if track["name"] == "drums · bar 2")
        self.assertEqual(piece["canvas"], 1)
        self.assertIs(piece["_audio"], self.audio["drums"])
        self.assertAlmostEqual(piece["offset"], 2.0)
        self.assertAlmostEqual(piece["duration"], 2.0, places=3)
        self.assertAlmostEqual(float(window.engine.render(int(piece["start"] * OUTPUT_RATE) + 10, 2)[0, 0]),
                               0.5 + 0.1 + 0.2, places=3)
        self.assertEqual(self.layout(), before)

    def test_side_by_side_transfer_does_not_scroll_the_destination(self):
        window = self.window
        window._set_canvas_layout(True)
        for view in (self.main, self.second):
            view.zoom_horizontal(6.0)
        self.app.processEvents()
        self.second.horizontalScrollBar().setValue(0)
        window.project.tracks[2]["start"] = 7.5
        window._reload_engine()
        window._refresh_timeline()
        self.app.processEvents()
        before = self.layout()
        window._move_track_to_canvas(2, 1)  # lands off screen on canvas 2
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual(self.layout(), before)

    def test_several_blocks_move_in_one_batch(self):
        window = self.window
        self.main.set_picks({0: {2}, 1: {2}, 2: {2}})
        refreshes = []
        original = window._refresh_timeline

        def counted():
            refreshes.append(1)
            original()

        clip = self.clips(self.main)[0]
        grab = self.grab_point(clip, 4.5)
        steps = len(window.history._undo) if hasattr(window.history, "_undo") else None
        with patch.object(window, "_refresh_timeline", counted):
            self.drag(self.main, grab, self.second, QPointF(grab.x(), 60))
        moved = [track for track in window.project.tracks if track.get("canvas") == 1]
        self.assertEqual(len(moved), 3)
        self.assertEqual(len(refreshes), 1)
        if steps is not None:
            self.assertEqual(len(window.history._undo), steps + 1)
        window._undo()
        self.assertEqual(self.names(), ["drums", "bass", "vocals"])

    def test_stem_playback_carries_on_through_a_transfer(self):
        window = self.window
        engine = window.engine
        with patch.object(engine, "_ensure_stream", lambda: None):
            window._start_blocks([(0, 1), (1, 1)])
            window._move_track_to_canvas(2, 1)
            for _ in range(3):
                self.app.processEvents()
            out = np.zeros((256, 2), np.float32)
            engine._callback(out, 256, None, None)
        self.assertAlmostEqual(float(out[0, 0]), 0.6, places=4)
        self.assertEqual(sorted(engine.previews()), [(0, 1), (1, 1)])
        self.assertTrue(self.second.isVisible())


class ImportLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_importing_audio_moves_no_panel_and_runs_no_animation(self):
        window = MainWindow()
        window.resize(1200, 800)
        window.show()
        for _ in range(3):
            self.app.processEvents()
        folder = tempfile.mkdtemp()
        paths = []
        for index in range(12):  # more strips than the mixer can show
            path = os.path.join(folder, f"take {index}.wav")
            sf.write(path, np.full((OUTPUT_RATE, 2), 0.1, np.float32), OUTPUT_RATE)
            paths.append(path)
        panels = (window.canvas_split, window.timeline, window.timeline2, window.mixer, window.statusBar())
        before = [panel.geometry().getRect() for panel in panels]
        strips = window.mixer.scroll.viewport().height()
        window._import_paths(paths, split=False)
        window.finish_loading()
        for _ in range(5):
            self.app.processEvents()
        self.assertEqual(len(window.project.tracks), 12)
        self.assertEqual([panel.geometry().getRect() for panel in panels], before)
        # The strips overflow the mixer, yet keep their height: no scroll bar
        # appears and pushes them up.
        self.assertGreater(window.mixer.body.width(), window.mixer.scroll.viewport().width())
        self.assertEqual(window.mixer.scroll.viewport().height(), strips)

        # The split banner keeps one place and one size as its text changes.
        with patch.object(window.stem_jobs, "pending", return_value=2):
            window._stem_names = {"job": "A"}
            window._show_stem_progress(0.05)
            self.app.processEvents()
            bar = window.stem_bar.geometry().getRect()
            banner = window.stem_banner.geometry().getRect()
            window._stem_names = {"job": "A much longer song name from a long album title"}
            window._show_stem_progress(0.57)
            self.app.processEvents()
            self.assertEqual(window.stem_bar.geometry().getRect(), bar)
            self.assertEqual(window.stem_banner.geometry().getRect(), banner)
            window._stem_names = {}
        running = [
            animation for animation in window.findChildren(QAbstractAnimation)
            if animation.state() == QAbstractAnimation.State.Running
        ]
        self.assertEqual(running, [])
        window.timeline.clear_scene()
        window.timeline2.clear_scene()
        window.close()
        window.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
