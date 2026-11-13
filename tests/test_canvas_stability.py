"""Stable canvases and stem-block-only dragging.

Only a stem (bar) block moves with a left-drag; its track, the other blocks,
the track headers and both canvases stay put. Imports and transfers change
the canvases' contents without moving, resizing or rebuilding the widgets
around them.
"""

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
from PySide6.QtCore import QAbstractAnimation, QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QToolBar, QWidget
from shiboken6 import isValid

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.ui import timeline as timeline_module
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


class _Geometry(QObject):
    """Counts moves, resizes, shows and hides of widgets, by class name."""

    KINDS = (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide)

    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def eventFilter(self, watched, event):
        if event.type() in self.KINDS and isinstance(watched, QWidget):
            self.events.append((event.type().name, watched.objectName() or type(watched).__name__))
        return False


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs()
        self.window.resize(1400, 900)
        drums = np.zeros((OUTPUT_RATE * 8, 2), dtype=np.float32)
        drums[OUTPUT_RATE * 2 : OUTPUT_RATE * 4] = 0.5  # bar 2 is loud
        bass = np.full((OUTPUT_RATE * 8, 2), 0.1, dtype=np.float32)
        for name, audio in (("drums", drums), ("bass", bass)):
            self.window.project.tracks.append(
                {"name": name, "path": None, "start": 0.0, "duration": 8.0, "offset": 0.0, "_audio": audio}
            )
        self.window.project.bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.window.show()
        for _ in range(3):
            self.app.processEvents()

    def tearDown(self):
        self.window.engine.preview_stop_all()
        for timeline in (self.window.timeline, self.window.timeline2):
            timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    # -- helpers

    def clips(self, view):
        return {
            item.clip_index: item
            for item in view.scene_obj.items()
            if isinstance(item, ClipItem) and isValid(item)
        }

    def track(self, name):
        return next(track for track in self.window.project.tracks if track["name"] == name)

    def send(self, source, kind, at, buttons):
        QApplication.sendEvent(source.viewport(), QMouseEvent(
            kind, QPointF(at), QPointF(source.viewport().mapToGlobal(at)),
            Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
        self.app.processEvents()

    def drag(self, source, scene_from, target_view, scene_to, release=True):
        start = source.mapFromScene(scene_from)
        end_global = target_view.viewport().mapToGlobal(target_view.mapFromScene(scene_to))
        end = source.viewport().mapFromGlobal(end_global)
        self.send(source, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
        self.send(source, QEvent.Type.MouseMove, (start + end) / 2, Qt.MouseButton.LeftButton)
        self.send(source, QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton)
        if release:
            self.send(source, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
            for _ in range(3):
                self.app.processEvents()
        return end

    def frame(self):
        """What must never move: both canvases, headers, toolbar, docks."""
        window = self.window
        widgets = [window.timeline, window.timeline2, window.headers, window.headers2,
                   *window.findChildren(QToolBar), *window.findChildren(QDockWidget)]
        return (
            [widget.geometry().getRect() for widget in widgets],
            window.canvas_split.sizes(),
            window.timeline.horizontalScrollBar().value(),
            window.timeline2.horizontalScrollBar().value(),
            window.timeline.pixels_per_second,
            window.timeline2.pixels_per_second,
            window.geometry().getRect(),
        )

    def grab_point(self, clip, seconds):
        pps = clip.pixels_per_second
        return QPointF(clip.pos().x() + seconds * pps, clip.pos().y() + clip.rect().height() * 0.75)

    # -- only a stem block drags

    def test_a_left_drag_moves_only_that_block(self):
        window = self.window
        view = window.timeline
        drums_clip = self.clips(view)[0]
        bass_clip = self.clips(view)[1]
        bass_x = bass_clip.pos().x()
        frame = self.frame()
        grab = self.grab_point(drums_clip, 2.5)  # bar 2
        self.drag(view, grab, view, QPointF(grab.x() + PIXELS_PER_SECOND, grab.y()))
        piece = self.track("drums · bar 2")
        self.assertAlmostEqual(piece["start"], 3.0, places=2)
        # The rest of the drums track is where it was, split around the gap.
        rests = [t for t in window.project.tracks if t["name"] == "drums"]
        self.assertEqual(sorted(round(t["start"], 3) for t in rests), [0.0, 4.0])
        self.assertAlmostEqual(self.clips(view)[window.project.tracks.index(self.track("bass"))].pos().x(), bass_x)
        self.assertEqual(self.frame(), frame)

    def test_a_click_does_not_move_a_block(self):
        view = self.window.timeline
        clip = self.clips(view)[0]
        grab = self.grab_point(clip, 0.5)
        tiny = QPointF(grab.x() + QApplication.startDragDistance() / 3, grab.y())
        self.drag(view, grab, view, tiny)
        self.assertEqual([t["name"] for t in self.window.project.tracks], ["drums", "bass"])
        self.assertEqual(self.track("drums")["start"], 0.0)

    def test_dragging_empty_canvas_or_a_header_moves_nothing(self):
        window = self.window
        before = [dict((k, v) for k, v in t.items() if k != "_audio") for t in window.project.tracks]
        frame = self.frame()
        view = window.timeline
        bass = self.clips(view)[1]
        empty = QPointF(10.0 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() + 40)
        self.drag(view, empty, view, QPointF(empty.x() - 200, empty.y()))
        header = window.headers._headers[0]
        middle = header.name_label.geometry().center()
        QTest.mousePress(header, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, middle)
        QTest.mouseMove(header, middle + QPoint(0, 120))
        QTest.mouseRelease(header, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, middle + QPoint(0, 120))
        self.app.processEvents()
        after = [dict((k, v) for k, v in t.items() if k != "_audio") for t in window.project.tracks]
        self.assertEqual(after, before)
        self.assertEqual(self.frame(), frame)

    def test_play_pause_buttons_still_work(self):
        window = self.window
        with patch.object(window.engine, "_ensure_stream", lambda: None):
            clip = self.clips(window.timeline)[0]
            button = next(c for c in clip.childItems() if type(c).__name__ == "BarPlayButton")
            centre = clip.mapToScene(button.mapRectToParent(button.rect()).center())
            self.drag(window.timeline, centre, window.timeline, centre)
            self.assertTrue(button.playing)
            self.drag(window.timeline, centre, window.timeline, centre)
            self.assertFalse(button.playing)
        self.assertEqual([t["name"] for t in window.project.tracks], ["drums", "bass"])

    # -- transfers between canvases

    def test_transfer_keeps_the_audio_and_the_canvases_still(self):
        window = self.window
        window.timeline.zoom_horizontal(1.5)
        window.timeline2.zoom_horizontal(1.5)
        window.timeline.horizontalScrollBar().setValue(40)
        self.app.processEvents()
        frame = self.frame()
        drums = self.track("drums")
        clip = self.clips(window.timeline)[0]
        grab = self.grab_point(clip, 2.5)
        drop = QPointF(grab.x(), 60)
        self.drag(window.timeline, grab, window.timeline2, drop)
        piece = self.track("drums · bar 2")
        self.assertEqual(piece["canvas"], 1)
        self.assertIs(piece["_audio"], drums["_audio"])  # shared, not copied
        self.assertAlmostEqual(piece["duration"], 2.0, places=3)
        self.assertAlmostEqual(piece["offset"], 2.0, places=3)
        # The piece sounds exactly like bar 2 did, at its new place.
        at = int((piece["start"] + 0.5) * OUTPUT_RATE)
        self.assertAlmostEqual(float(window.engine.render(at, 4)[0, 0]), 0.5 + 0.1, places=4)
        self.assertEqual(self.frame(), frame)

    def test_escape_cancels_a_drag_and_puts_the_block_back(self):
        window = self.window
        view = window.timeline
        clip = self.clips(view)[0]
        grab = self.grab_point(clip, 2.5)
        self.drag(view, grab, view, QPointF(grab.x() + 2 * PIXELS_PER_SECOND, grab.y()), release=False)
        self.assertIsNotNone(view._block_drag)
        self.assertIsNotNone(clip._hidden_span)
        QTest.keyClick(view.viewport(), Qt.Key.Key_Escape)
        self.assertIsNone(view._block_drag)
        self.assertIsNone(clip._hidden_span)
        end = view.mapFromScene(QPointF(grab.x() + 2 * PIXELS_PER_SECOND, grab.y()))
        self.send(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
        self.assertEqual([t["name"] for t in window.project.tracks], ["drums", "bass"])
        self.assertEqual(self.track("drums")["duration"], 8.0)

    def test_a_drop_outside_both_canvases_changes_nothing(self):
        window = self.window
        view = window.timeline
        clip = self.clips(view)[0]
        grab = self.grab_point(clip, 2.5)
        start = view.mapFromScene(grab)
        toolbar = window.findChildren(QToolBar)[0]
        outside = view.viewport().mapFromGlobal(toolbar.mapToGlobal(toolbar.rect().center()))
        self.send(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseMove, (start + outside) / 2, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseMove, outside, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseButtonRelease, outside, Qt.MouseButton.NoButton)
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual([t["name"] for t in window.project.tracks], ["drums", "bass"])
        self.assertEqual(self.track("drums")["duration"], 8.0)
        clip = self.clips(view)[0]
        self.assertIsNone(clip._hidden_span)

    def test_a_one_block_piece_dropped_outside_goes_back(self):
        window = self.window
        window._move_block(0, 2.0, 2.0, 2.5, False, 2, 1)  # bar 2 on its own row
        piece = self.track("drums · bar 2")
        view = window.timeline
        clip = self.clips(view)[window.project.tracks.index(piece)]
        start = view.mapFromScene(self.grab_point(clip, 0.3))
        toolbar = window.findChildren(QToolBar)[0]
        outside = view.viewport().mapFromGlobal(toolbar.mapToGlobal(toolbar.rect().center()))
        self.send(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseMove, (start + outside) / 2, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseMove, outside, Qt.MouseButton.LeftButton)
        self.send(view, QEvent.Type.MouseButtonRelease, outside, Qt.MouseButton.NoButton)
        for _ in range(3):
            self.app.processEvents()
        self.assertAlmostEqual(piece["start"], 2.5, places=3)
        self.assertNotIn("canvas", piece)

    def test_picked_blocks_move_in_one_batch(self):
        window = self.window
        window.timeline.set_picks({0: {1}, 1: {1}})
        rebuilds = []
        original = timeline_module.TimelineView.rebuild

        def counted(view, *args, **kwargs):
            rebuilds.append(view.canvas)
            return original(view, *args, **kwargs)

        frame = self.frame()
        with patch.object(timeline_module.TimelineView, "rebuild", counted):
            clip = self.clips(window.timeline)[0]
            grab = self.grab_point(clip, 2.5)
            self.drag(window.timeline, grab, window.timeline2, QPointF(grab.x(), 60))
        moved = sorted(t["name"] for t in window.project.tracks if t.get("canvas") == 1)
        self.assertEqual(moved, ["bass · bar 2", "drums · bar 2"])
        self.assertEqual(sorted(rebuilds), [0, 1])  # one refresh for both
        self.assertEqual(self.frame(), frame)

    def test_other_blocks_keep_playing_through_a_transfer(self):
        window = self.window
        engine = window.engine
        with patch.object(engine, "_ensure_stream", lambda: None):
            window._start_blocks([(1, 3)])  # bass, bar 4
            out = np.zeros((256, 2), np.float32)
            engine._callback(out, 256, None, None)
            clip = self.clips(window.timeline)[0]
            grab = self.grab_point(clip, 2.5)
            self.drag(window.timeline, grab, window.timeline2, QPointF(grab.x(), 60))
            for _ in range(3):
                self.app.processEvents()
            bass = window.project.tracks.index(self.track("bass"))
            self.assertIn((bass, 3), engine.previews())
            engine._callback(out, 256, None, None)
            self.assertAlmostEqual(float(out[0, 0]), 0.1, places=4)

    # -- imports

    def test_import_while_playing_leaves_the_frame_alone(self):
        window = self.window
        path = os.path.join(self.folder.name, "new.wav")
        sf.write(path, np.full((OUTPUT_RATE * 3, 2), 0.2, np.float32), OUTPUT_RATE)
        window.split_on_import_action.setChecked(False)
        window.engine._playing = True
        window.engine.seek(1.0)
        frame = self.frame()
        spy = _Geometry()
        self.app.installEventFilter(spy)
        try:
            window._import_paths([path])
            self.assertEqual(len(window.project.tracks), 2)  # decoding in the background
            window.finish_loading()
            self.app.processEvents()
        finally:
            self.app.removeEventFilter(spy)
            window.engine._playing = False
        self.assertEqual([t["name"] for t in window.project.tracks], ["drums", "bass", "new"])
        self.assertEqual(self.frame(), frame)
        # Existing headers and mixer strips are kept; only the new track's
        # widgets are made, so nothing else is hidden or re-shown.
        hidden = [name for kind, name in spy.events if kind == "Hide"]
        self.assertLessEqual(len(hidden), 2, hidden)
        running = [a for a in window.findChildren(QAbstractAnimation) if a.state() == QAbstractAnimation.State.Running]
        self.assertEqual(running, [])

    def test_a_transfer_keeps_the_other_header_and_mixer_widgets(self):
        window = self.window
        drums_header = window.headers._headers[0]
        strips = list(window.mixer._strips)
        window._move_track_to_canvas(1, 1)
        self.assertIs(window.headers._headers[0], drums_header)
        self.assertEqual(window.mixer._strips, strips)  # the same strips, not rebuilt

    def test_canvas_two_open_at_launch(self):
        self.assertTrue(self.window.timeline2.isVisible())


if __name__ == "__main__":
    unittest.main()
