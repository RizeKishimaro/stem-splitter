"""Second canvas: a second arrangement under the main timeline.

Tracks on it play with the song. Whole tracks and single bars move between
the canvases by dragging (real mouse events here) or with the Track menu.
"""

import os
import tempfile
import unittest
from unittest.mock import PropertyMock, patch
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SONICLAB_DEMUCS_PYTHON", "")

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.core.project import Project
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import PIXELS_PER_SECOND, ClipItem


class _Jobs:
    def events_for(self, path):
        return ()

    def result_for(self, path):
        return None

    def submit(self, path):
        return "queued"

    def poll(self):
        return None

    def cancel(self):
        pass


class SecondCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
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
        self.app.processEvents()

    def tearDown(self):
        for timeline in (self.window.timeline, self.window.timeline2):
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

    def drag(self, source, scene_from: QPointF, target_view, scene_to: QPointF, modifiers=Qt.KeyboardModifier.NoModifier):
        """Press in ``source`` and release over ``target_view``. Every event
        goes to the source viewport, as with a real mouse grab."""
        start = source.mapFromScene(scene_from)
        end_global = target_view.viewport().mapToGlobal(target_view.mapFromScene(scene_to))
        end = source.viewport().mapFromGlobal(end_global)
        middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
        for kind, at, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, middle, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton),
        ):
            event = QMouseEvent(kind, QPointF(at), QPointF(source.viewport().mapToGlobal(at)),
                                Qt.MouseButton.LeftButton, buttons, modifiers)
            QApplication.sendEvent(source.viewport(), event)
            self.app.processEvents()
        for _ in range(3):
            self.app.processEvents()

    def track(self, name):
        return next(track for track in self.window.project.tracks if track["name"] == name)

    def open_second_canvas(self):
        self.window.second_canvas_action.setChecked(True)
        self.app.processEvents()
        self.assertTrue(self.window.timeline2.isVisible())

    # -- tests

    def test_second_canvas_is_open_at_launch_and_can_be_hidden(self):
        window = self.window
        self.assertTrue(window.timeline2.isVisible())
        self.open_second_canvas()
        self.assertEqual(self.clips(window.timeline2), {})
        self.assertEqual(sorted(self.clips(window.timeline)), [0, 1])
        window.second_canvas_action.setChecked(False)
        self.app.processEvents()
        self.assertFalse(window.timeline2.isVisible())

    def test_ctrl_drag_a_bar_onto_the_second_canvas(self):
        window = self.window
        self.open_second_canvas()
        drums = self.clips(window.timeline)[0]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        drop = QPointF(drums.pos().x() + 5.3 * PIXELS_PER_SECOND, 60)
        self.drag(window.timeline, grab, window.timeline2, drop, Qt.KeyboardModifier.ControlModifier)

        piece = self.track("drums · bar 2")
        self.assertEqual(piece["canvas"], 1)
        self.assertTrue(piece["bar_block"])
        self.assertAlmostEqual(piece["start"], 5.0, places=2)
        self.assertAlmostEqual(piece["offset"], 2.0)
        index = window.project.tracks.index(piece)
        self.assertIn(index, self.clips(window.timeline2))
        self.assertNotIn(index, self.clips(window.timeline))
        # The rest of the drums stay on the main canvas, with a gap where bar 2 was.
        main = [track for track in window.project.tracks if track.get("canvas", 0) == 0]
        self.assertEqual(sorted(track["name"] for track in main), ["bass", "drums", "drums"])
        # It plays: bar 2's audio now sounds at 5 s, and not at 2.5 s.
        self.assertAlmostEqual(float(window.engine.render(int(5.5 * OUTPUT_RATE), 4)[0, 0]), 0.5 + 0.1, places=3)
        self.assertAlmostEqual(float(window.engine.render(int(2.5 * OUTPUT_RATE), 4)[0, 0]), 0.1, places=3)

        # One undo puts it back.
        window._undo()
        self.assertEqual([track["name"] for track in window.project.tracks], ["drums", "bass"])
        self.assertEqual(self.clips(window.timeline2), {})

    def test_drag_one_block_across_and_back(self):
        # A left-drag moves the one bar block under the mouse; the rest of
        # the track stays on its canvas, at its time.
        window = self.window
        self.open_second_canvas()
        bass = self.clips(window.timeline)[1]
        audio = self.track("bass")["_audio"]
        grab = QPointF(bass.pos().x() + 0.5 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() * 0.75)
        drop = QPointF(grab.x() + 1.0 * PIXELS_PER_SECOND, 60)
        self.drag(window.timeline, grab, window.timeline2, drop)
        piece = self.track("bass · bar 1")
        self.assertEqual(piece["canvas"], 1)
        self.assertAlmostEqual(piece["start"], 1.0, places=2)
        self.assertAlmostEqual(piece["duration"], 2.0, places=3)
        self.assertIs(piece["_audio"], audio)  # the same audio, not a copy
        rest = self.track("bass")
        self.assertNotIn("canvas", rest)
        self.assertAlmostEqual(rest["start"], 2.0, places=3)  # bars 2-4 stay put
        self.assertAlmostEqual(self.track("drums")["start"], 0.0)
        self.assertAlmostEqual(self.track("drums")["duration"], 8.0)

        # And back, from the second canvas to the main one.
        index = window.project.tracks.index(piece)
        clip = self.clips(window.timeline2)[index]
        grab = QPointF(clip.pos().x() + 0.5 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
        drums = self.clips(window.timeline)[0]
        drop = QPointF(grab.x() - 1.0 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() + 20)
        self.drag(window.timeline2, grab, window.timeline, drop)
        self.assertNotIn("canvas", piece)
        self.assertAlmostEqual(piece["start"], 0.0, places=2)
        self.assertEqual(self.clips(window.timeline2), {})

    def test_menu_command_save_and_reopen(self):
        window = self.window
        window._select_track(1)
        window.canvas_move_action.trigger()
        self.app.processEvents()
        self.assertTrue(window.timeline2.isVisible())
        self.assertEqual(self.track("bass")["canvas"], 1)
        self.assertEqual(window._selected, 1)
        self.assertTrue(self.clips(window.timeline2)[1].isSelected())
        # Tracks on the second canvas keep it open.
        window.second_canvas_action.setChecked(False)
        self.app.processEvents()
        self.assertTrue(window.timeline2.isVisible())
        # The field is saved like the other track fields.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.soniclab"
            project = Project()
            project.tracks = [{k: v for k, v in t.items() if not k.startswith("_")} for t in window.project.tracks]
            project.save(path)
            self.assertEqual(Project.load(path).tracks[1]["canvas"], 1)
        window.canvas_move_action.trigger()  # and back
        self.app.processEvents()
        self.assertNotIn("canvas", self.track("bass"))

    def test_selection_moves_between_canvases(self):
        window = self.window
        window._move_track_to_canvas(1, 1)
        window._select_track(0)
        self.assertTrue(self.clips(window.timeline)[0].isSelected())
        window._select_track(1)
        self.assertEqual(window._selected, 1)
        self.assertTrue(self.clips(window.timeline2)[1].isSelected())
        self.assertFalse(self.clips(window.timeline)[0].isSelected())


    def test_side_by_side_layout(self):
        window = self.window
        self.open_second_canvas()
        self.assertEqual(window.canvas_split.orientation(), Qt.Orientation.Vertical)
        window.canvas_side_action.trigger()
        self.app.processEvents()
        self.assertEqual(window.canvas_split.orientation(), Qt.Orientation.Horizontal)
        self.assertTrue(window.canvas_side_action.isChecked())
        # Side by side, the second canvas is to the right of the main one.
        main = window.timeline.mapTo(window, QPoint(0, 0))
        second = window.timeline2.mapTo(window, QPoint(0, 0))
        self.assertGreater(second.x(), main.x() + window.timeline.width() // 2)
        self.assertLess(abs(second.y() - main.y()), 5)

        # Each scrolls on its own, so two parts of the song can be seen.
        first_bar = window.timeline.horizontalScrollBar()
        second_bar = window.timeline2.horizontalScrollBar()
        self.assertGreater(second_bar.maximum(), 300)
        second_bar.setValue(300)
        self.assertEqual(first_bar.value(), 0)

        # A bar Ctrl-dragged to the right-hand canvas still lands there.
        drums = self.clips(window.timeline)[0]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        drop = window.timeline2.mapToScene(QPoint(60, 60))
        self.drag(window.timeline, grab, window.timeline2, drop, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.track("drums · bar 2")["canvas"], 1)

        # Stacked again, the time axes line up again.
        window.canvas_stacked_action.trigger()
        self.app.processEvents()
        self.assertEqual(window.canvas_split.orientation(), Qt.Orientation.Vertical)
        self.assertEqual(first_bar.value(), second_bar.value())
        first_bar.setValue(120)
        self.assertEqual(second_bar.value(), 120)


    def test_a_placed_bar_on_top_of_its_track_can_be_grabbed_again(self):
        # Regression: a bar dropped onto its own row lands over the rest of the
        # track; that clip was drawn on top and took every later grab.
        window = self.window
        drums = self.clips(window.timeline)[0]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        self.drag(window.timeline, grab, window.timeline, QPointF(grab.x() + 3 * PIXELS_PER_SECOND, grab.y()),
                  Qt.KeyboardModifier.ControlModifier)
        piece = self.track("drums · bar 2")
        tail = next(t for t in window.project.tracks if t["name"] == "drums" and t["start"] > 0)
        self.assertAlmostEqual(piece["start"], 5.0, places=2)
        self.assertAlmostEqual(tail["start"], 4.0, places=2)  # the piece sits on top of it
        for expected in (6.0, 7.0):
            clip = self.clips(window.timeline)[window.project.tracks.index(piece)]
            point = QPointF(clip.pos().x() + 0.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
            self.drag(window.timeline, point, window.timeline, QPointF(point.x() + PIXELS_PER_SECOND, point.y()))
            self.assertAlmostEqual(piece["start"], expected, places=2)
            self.assertAlmostEqual(tail["start"], 4.0, places=2)

    def test_each_canvas_plays_on_its_own(self):
        window = self.window
        self.open_second_canvas()
        window._move_track_to_canvas(1, 1)  # bass to canvas 2
        window.project.tracks[0]["solo"] = True  # a solo on the main canvas
        window._push_mix(0, window.project.tracks[0])
        engine = window.engine
        state = {"playing": False}
        # No audio device in tests: stand in for starting and stopping output.
        with patch.object(type(engine), "playing", new_callable=PropertyMock) as playing, \
                patch.object(engine, "play", lambda: state.update(playing=True)), \
                patch.object(engine, "pause", lambda: state.update(playing=False)):
            playing.side_effect = lambda: state["playing"]
            self._play_canvases(window, engine)

    def _play_canvases(self, window, engine):
        try:
            at = int(2.5 * OUTPUT_RATE)  # drums 0.5, bass 0.1
            self.assertAlmostEqual(float(engine.render(at, 4)[0, 0]), 0.5, places=3)  # solo drums only

            window.headers2.play_requested.emit()  # canvas 2's ▶
            self.assertTrue(engine.playing)
            self.assertEqual(window._canvas_focus, 1)
            self.assertAlmostEqual(float(engine.render(at, 4)[0, 0]), 0.1, places=3)  # bass only
            self.assertTrue(window.headers2.play_button.isChecked())
            self.assertFalse(window.headers.play_button.isChecked())
            # The tracks' own settings are untouched.
            self.assertTrue(window.project.tracks[0]["solo"])
            self.assertNotIn("muted", window.project.tracks[1])

            window.headers.play_requested.emit()  # main canvas's ▶ switches over
            self.assertEqual(window._canvas_focus, 0)
            self.assertAlmostEqual(float(engine.render(at, 4)[0, 0]), 0.5, places=3)

            window.headers.play_requested.emit()  # again: stop
            self.assertFalse(engine.playing)
            self.assertFalse(window.headers.play_button.isChecked())

            window._hover_canvas = 0
            window._toggle_transport()  # Space plays the canvas under the mouse
            self.assertEqual(window._canvas_focus, 0)
            window._pause()
            window.play_both_action.trigger()  # Audio > Play Both Canvases
            self.assertIsNone(window._canvas_focus)
            window.project.tracks[0]["solo"] = False
            window._push_mix(0, window.project.tracks[0])
            self.assertAlmostEqual(float(engine.render(at, 4)[0, 0]), 0.6, places=3)
        finally:
            window._canvas_focus = None


    def visible_in(self, view, index):
        clip = self.clips(view)[index]
        shown = view.mapToScene(view.viewport().rect()).boundingRect()
        return shown.intersects(clip.sceneBoundingRect())

    def test_canvases_keep_their_size_when_stems_move(self):
        window = self.window
        self.open_second_canvas()
        window._settle_canvases()
        self.app.processEvents()
        sizes = window.canvas_split.sizes()
        heights = (window.timeline.height(), window.timeline2.height())
        scroll = window.timeline.horizontalScrollBar().value()
        window._move_track_to_canvas(1, 1)
        window._move_track_to_canvas(0, 1)
        window._move_track_to_canvas(1, 0)
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual(window.canvas_split.sizes(), sizes)
        self.assertEqual((window.timeline.height(), window.timeline2.height()), heights)
        self.assertEqual(window.timeline.horizontalScrollBar().value(), scroll)

        # Side by side: narrow track headers, and the canvas starts at its clip.
        window._move_track_to_canvas(0, 0)
        window._move_track_to_canvas(1, 1, 30.0)  # far into the song
        window.timeline2.horizontalScrollBar().setValue(0)
        window.canvas_side_action.trigger()
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual(window.headers2.width(), 150)
        self.assertTrue(self.visible_in(window.timeline2, 1))

    def test_play_button_never_drags_and_a_click_plays(self):
        # On a short placed bar the play button covers the middle. It is a
        # button: pressing it and moving does not pick the bar up.
        window = self.window
        window._move_block(0, 2.0, 2.0, 2.5, False, 2, 1)  # bar 2 on its own row
        piece = self.track("drums · bar 2")
        clip = self.clips(window.timeline)[window.project.tracks.index(piece)]
        button = next(child for child in clip.childItems() if type(child).__name__ == "BarPlayButton")
        centre = clip.mapToScene(button.mapRectToParent(button.rect()).center())
        top = clip.pos().y()
        played = []
        window.timeline.bar_play_requested.connect(lambda bar, track: played.append(bar))
        self.drag(window.timeline, centre, window.timeline, QPointF(centre.x() + PIXELS_PER_SECOND, centre.y()))
        self.assertAlmostEqual(piece["start"], 2.5, places=2)
        self.assertEqual(played, [])
        # A plain click on it plays the bar and does not move anything.
        self.drag(window.timeline, centre, window.timeline, centre)
        self.assertEqual(played, [1])
        self.assertAlmostEqual(piece["start"], 2.5, places=2)
        clip = self.clips(window.timeline)[window.project.tracks.index(piece)]
        self.assertAlmostEqual(clip.pos().y(), top, places=1)
        self.assertTrue(clip.isVisible())

    def test_a_bar_play_button_plays_that_stem_on_its_own(self):
        window = self.window
        engine = window.engine
        engine.seek(7.0)
        with patch.object(engine, "_ensure_stream", lambda: None):
            window.timeline.bar_play_requested.emit(1, 0)  # drums' bar 2 ▶
            self.assertFalse(engine.playing)  # the transport is not used
            self.assertEqual(engine.position, 7.0)
            out = np.zeros((64, 2), np.float32)
            engine._callback(out, 64, None, None)
            self.assertAlmostEqual(float(out[0, 0]), 0.5, places=3)  # drums only
            window.timeline.bar_play_requested.emit(1, 1)  # bass's bar 2 ▶ as well
            engine._callback(out, 64, None, None)
            self.assertAlmostEqual(float(out[0, 0]), 0.6, places=3)  # both together
            window._stop_blocks()
        self.assertNotIn("muted", window.project.tracks[1])  # own settings untouched

    def test_the_silent_canvas_does_not_scroll_while_the_other_plays(self):
        window = self.window
        self.open_second_canvas()
        window._move_track_to_canvas(1, 1)
        for view in (window.timeline, window.timeline2):
            view.zoom_horizontal(8.0)
        self.app.processEvents()
        bar, bar2 = window.timeline.horizontalScrollBar(), window.timeline2.horizontalScrollBar()
        self.assertGreater(bar.maximum(), 100)
        engine = window.engine
        state = {"playing": False}
        with patch.object(type(engine), "playing", new_callable=PropertyMock) as playing, \
                patch.object(engine, "play", lambda: state.update(playing=True)), \
                patch.object(engine, "pause", lambda: state.update(playing=False)):
            playing.side_effect = lambda: state["playing"]
            bar.setValue(50)  # stopped: stacked canvases scroll together
            self.assertEqual(bar2.value(), 50)
            window.headers.play_requested.emit()  # the main canvas alone
            bar.setValue(100)
            self.assertEqual(bar2.value(), 50)
            window.headers.play_requested.emit()  # stop
            bar.setValue(80)
            self.assertEqual(bar2.value(), 80)

    def playhead_seconds(self, view):
        return (view.playhead.line().x1() - 0) / view.pixels_per_second

    def test_a_canvas_playhead_rests_while_that_canvas_is_silent(self):
        window = self.window
        self.open_second_canvas()
        window._move_track_to_canvas(1, 1)
        engine = window.engine
        state = {"playing": False}
        with patch.object(type(engine), "playing", new_callable=PropertyMock) as playing, \
                patch.object(engine, "play", lambda: state.update(playing=True)), \
                patch.object(engine, "pause", lambda: state.update(playing=False)):
            playing.side_effect = lambda: state["playing"]
            # Stopped: clicking canvas 2's ruler moves only canvas 2's playhead.
            window.timeline2.seek_requested.emit(4.0)
            window._refresh_transport()
            self.assertAlmostEqual(self.playhead_seconds(window.timeline2), 4.0, places=2)
            self.assertAlmostEqual(self.playhead_seconds(window.timeline), 0.0, places=2)
            window.timeline.seek_requested.emit(1.0)
            window._refresh_transport()
            self.assertAlmostEqual(self.playhead_seconds(window.timeline), 1.0, places=2)
            self.assertAlmostEqual(self.playhead_seconds(window.timeline2), 4.0, places=2)

            # Playing the main canvas alone: canvas 2's playhead stays put.
            window.headers.play_requested.emit()
            engine.seek(2.5)
            window._refresh_transport()
            self.assertAlmostEqual(self.playhead_seconds(window.timeline), 2.5, places=2)
            self.assertAlmostEqual(self.playhead_seconds(window.timeline2), 4.0, places=2)
            window.headers.play_requested.emit()  # stop

            # Canvas 2's ▶ starts from canvas 2's own playhead.
            window.headers2.play_requested.emit()
            self.assertAlmostEqual(engine.position, 4.0, places=2)
            engine.seek(5.0)
            window._refresh_transport()
            self.assertAlmostEqual(self.playhead_seconds(window.timeline2), 5.0, places=2)
            self.assertAlmostEqual(self.playhead_seconds(window.timeline), 2.5, places=2)
            window.headers2.play_requested.emit()

            # Play Both Canvases: both playheads move together.
            window.play_both_action.trigger()
            engine.seek(6.0)
            window._refresh_transport()
            self.assertAlmostEqual(self.playhead_seconds(window.timeline), 6.0, places=2)
            self.assertAlmostEqual(self.playhead_seconds(window.timeline2), 6.0, places=2)
            window._pause()


    def test_tracks_can_go_back_before_bar_one(self):
        # Bar 1 is the first downbeat, here 3 s into the song. The beat grid
        # used to start there, so a block could never be dragged back to 0 s.
        window = self.window
        window.project.bars = [3.0, 5.0, 7.0, 9.0, 11.0]
        drums = window.project.tracks[0]
        drums["start"] = 3.0
        window._reload_engine()
        window._refresh_timeline()
        view = window.timeline
        snap = [round((x - 0) / view.pixels_per_second, 3) for x in view._snap_points[:8]]
        self.assertEqual(snap[:7], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # intro beats, then bar 1
        clip = self.clips(view)[0]
        grab = QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(view, grab, view, QPointF(grab.x() - 3.0 * PIXELS_PER_SECOND - 60, grab.y()))
        piece = self.track("drums · bar 1")
        self.assertAlmostEqual(piece["start"], 0.0, places=3)
        # And to an intro beat.
        clip = self.clips(view)[window.project.tracks.index(piece)]
        grab = QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(view, grab, view, QPointF(grab.x() + 1.0 * PIXELS_PER_SECOND + 4, grab.y()))
        self.assertAlmostEqual(piece["start"], 1.0, places=3)

    def test_ctrl_drag_again_after_placing_on_a_track_or_canvas(self):
        # Bars are moved with Ctrl-drag. A placed bar is a selected piece of
        # its own, and Ctrl-clicking a selected item deselected it, so the
        # next Ctrl-drag did nothing.
        window = self.window
        self.open_second_canvas()
        ctrl = Qt.KeyboardModifier.ControlModifier
        drums = self.clips(window.timeline)[0]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        # On a track: a new row under the others.
        self.drag(window.timeline, grab, window.timeline, QPointF(grab.x() + PIXELS_PER_SECOND, grab.y() + 160), ctrl)
        piece = self.track("drums · bar 2")
        self.assertEqual(piece.get("canvas", 0), 0)
        for expected in (4.0, 5.0, 6.0):
            index = window.project.tracks.index(piece)
            clip = self.clips(window.timeline)[index]
            self.assertTrue(clip.isSelected())
            point = QPointF(clip.pos().x() + 0.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
            self.drag(window.timeline, point, window.timeline, QPointF(point.x() + PIXELS_PER_SECOND, point.y()), ctrl)
            self.assertAlmostEqual(piece["start"], expected, places=2)
        # On the second canvas.
        index = window.project.tracks.index(piece)
        clip = self.clips(window.timeline)[index]
        point = QPointF(clip.pos().x() + 0.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(window.timeline, point, window.timeline2, QPointF(point.x(), 60), ctrl)
        self.assertEqual(piece["canvas"], 1)
        for _ in range(2):
            index = window.project.tracks.index(piece)
            clip = self.clips(window.timeline2)[index]
            before = piece["start"]
            point = QPointF(clip.pos().x() + 0.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
            self.drag(window.timeline2, point, window.timeline2, QPointF(point.x() + PIXELS_PER_SECOND, point.y()), ctrl)
            self.assertAlmostEqual(piece["start"], before + 1.0, places=2)


    def test_a_bar_drags_from_its_downbeat_marker_and_a_click_opens_the_menu(self):
        # The yellow downbeat marker sits on a placed bar's top-left corner.
        # Pressing it used to open the bar menu at once, so the bar could not
        # be picked up there.
        window = self.window
        window.timeline.set_show_detected(True)
        window._move_block(0, 2.0, 2.0, 2.5, False, 2, 1)
        piece = self.track("drums · bar 2")
        piece["downbeats"] = [[2.0, 1], [2.5, 2], [3.0, 3], [3.5, 4]]  # file time; the bar starts at offset 2.0
        window._refresh_timeline()
        window._apply_detected_beats()
        menus = []
        window.timeline.downbeat_clicked.connect(lambda bar, where: menus.append(bar))
        window._show_bar_menu = lambda *args: None  # no real menu in tests

        def marker_point():
            clip = self.clips(window.timeline)[window.project.tracks.index(piece)]
            self.assertTrue(clip._downbeat_hit(0.0) is not None)
            return clip.mapToScene(QPointF(3.0, 8.0))

        start = marker_point()
        self.drag(window.timeline, start, window.timeline, QPointF(start.x() + PIXELS_PER_SECOND, start.y()))
        self.assertAlmostEqual(piece["start"], 3.5, places=2)
        self.assertEqual(menus, [])
        start = marker_point()
        self.drag(window.timeline, start, window.timeline, start)  # a click
        self.assertEqual(len(menus), 1)
        self.assertAlmostEqual(piece["start"], 3.5, places=2)


    def test_play_plays_the_canvas_under_the_mouse(self):
        window = self.window
        self.open_second_canvas()
        window._move_track_to_canvas(1, 1)
        engine = window.engine
        state = {"playing": False}
        with patch.object(type(engine), "playing", new_callable=PropertyMock) as playing, \
                patch.object(engine, "play", lambda: state.update(playing=True)), \
                patch.object(engine, "pause", lambda: state.update(playing=False)):
            playing.side_effect = lambda: state["playing"]
            QApplication.sendEvent(window.timeline2.viewport(), QEvent(QEvent.Type.Enter))
            self.assertEqual(window._hover_canvas, 1)
            window.timeline2.seek_requested.emit(3.0)
            window.play_btn.click()  # the toolbar ▶
            self.assertEqual(window._canvas_focus, 1)
            self.assertAlmostEqual(engine.position, 3.0, places=2)
            self.assertAlmostEqual(float(engine.render(int(3.5 * OUTPUT_RATE), 4)[0, 0]), 0.1, places=3)  # bass only
            window._toggle_transport()  # Space pauses
            self.assertFalse(engine.playing)
            QApplication.sendEvent(window.timeline.viewport(), QEvent(QEvent.Type.Enter))
            self.assertEqual(window._hover_canvas, 0)
            window._toggle_transport()  # Space now plays the main canvas
            self.assertEqual(window._canvas_focus, 0)
            self.assertAlmostEqual(float(engine.render(int(2.5 * OUTPUT_RATE), 4)[0, 0]), 0.5, places=3)  # drums only
            window._pause()
            window._canvas_focus = None

    def test_a_bar_dropped_on_another_track_joins_that_track(self):
        window = self.window
        drums = self.clips(window.timeline)[0]
        bass = self.clips(window.timeline)[1]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        drop = QPointF(grab.x() + 2 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() * 0.75)
        self.drag(window.timeline, grab, window.timeline, drop, Qt.KeyboardModifier.ControlModifier)
        piece = self.track("drums · bar 2")
        bass_track = self.track("bass")
        rows, _primaries = window.timeline.lane_rows(window.project.tracks)
        self.assertEqual(rows[window.project.tracks.index(piece)], rows[window.project.tracks.index(bass_track)])
        self.assertAlmostEqual(piece["start"], 4.0, places=2)
        # And it can be moved again, to another track and back.
        drums_row_clip = self.clips(window.timeline)[0]
        for target in (drums_row_clip, None):
            clip = self.clips(window.timeline)[window.project.tracks.index(piece)]
            point = QPointF(clip.pos().x() + 30, clip.pos().y() + clip.rect().height() * 0.75)
            if target is None:
                bass_clip = self.clips(window.timeline)[window.project.tracks.index(bass_track)]
                to_y = bass_clip.pos().y() + bass_clip.rect().height() * 0.75
            else:
                to_y = target.pos().y() + target.rect().height() * 0.75
            self.drag(window.timeline, point, window.timeline, QPointF(point.x(), to_y))
            rows, _primaries = window.timeline.lane_rows(window.project.tracks)
            expected = window.project.tracks.index(self.track("drums")) if target is not None else window.project.tracks.index(bass_track)
            self.assertEqual(rows[window.project.tracks.index(piece)], rows[expected])
            self.assertAlmostEqual(piece["start"], 4.0, places=2)


    def test_dropping_on_the_other_canvas_track_names_counts(self):
        window = self.window
        self.open_second_canvas()
        window.canvas_side_action.trigger()
        window._move_track_to_canvas(1, 1)
        self.app.processEvents()
        bass = self.clips(window.timeline2)[1]
        grab = QPointF(bass.pos().x() + 30, bass.pos().y() + bass.rect().height() * 0.75)
        start = window.timeline2.mapFromScene(grab)
        # The middle of canvas 1's track-name column, not its timeline.
        names = window.headers.mapTo(window, window.headers.rect().center())
        end = window.timeline2.viewport().mapFrom(window, names)
        self.assertFalse(window.timeline.viewport().rect().contains(window.timeline.viewport().mapFrom(window, names)))
        for kind, at, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, (start + end) / 2, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton),
        ):
            QApplication.sendEvent(window.timeline2.viewport(), QMouseEvent(
                kind, QPointF(at), QPointF(window.timeline2.viewport().mapToGlobal(at)),
                Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
            self.app.processEvents()
        for _ in range(3):
            self.app.processEvents()
        # The grabbed block lands on the main canvas; the rest stays on canvas 2.
        self.assertNotIn("canvas", self.track("bass · bar 1"))
        self.assertEqual(self.track("bass")["canvas"], 1)

    def test_left_drag_on_empty_space_selects_and_moves_together(self):
        window = self.window
        view = window.timeline
        view.resize(1200, 400)
        self.app.processEvents()
        drums, bass = self.clips(view)[0], self.clips(view)[1]
        # From empty space below the tracks, up over both clips: picks every bar.
        below = QPointF(10.5 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() + 30)
        above = QPointF(0.5 * PIXELS_PER_SECOND, drums.pos().y() + 10)
        self.assertIsNone(view.clip_at(view.mapFromScene(below)))
        self.drag(view, below, view, above)
        self.assertEqual(view.picks(), {0: {0, 1, 2, 3}, 1: {0, 1, 2, 3}})
        # Dragging one picked block moves every picked block, one second later.
        point = QPointF(drums.pos().x() + 30, drums.pos().y() + drums.rect().height() * 0.75)
        self.drag(view, point, view, QPointF(point.x() + PIXELS_PER_SECOND, point.y()))
        engine = window.engine
        self.assertAlmostEqual(float(engine.render(int(3.5 * OUTPUT_RATE), 4)[0, 0]), 0.6, places=3)
        self.assertAlmostEqual(float(engine.render(int(2.5 * OUTPUT_RATE), 4)[0, 0]), 0.1, places=3)
        self.assertAlmostEqual(float(engine.render(int(0.5 * OUTPUT_RATE), 4)[0, 0]), 0.0, places=3)
        # A plain click on empty space still moves the playhead.
        seeks = []
        view.seek_requested.connect(seeks.append)
        self.drag(view, below, view, below)
        self.assertEqual(len(seeks), 1)

    def test_middle_drag_pans(self):
        view = self.window.timeline
        view.resize(400, 300)
        self.app.processEvents()
        bar = view.horizontalScrollBar()
        bar.setValue(200)
        start = QPoint(200, 200)
        for kind, at, button, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton),
            (QEvent.Type.MouseMove, start + QPoint(-120, 0), Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton),
            (QEvent.Type.MouseButtonRelease, start + QPoint(-120, 0), Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton),
        ):
            QApplication.sendEvent(view.viewport(), QMouseEvent(
                kind, QPointF(at), QPointF(view.viewport().mapToGlobal(at)), button, buttons,
                Qt.KeyboardModifier.NoModifier))
            self.app.processEvents()
        self.assertEqual(bar.value(), 320)


    def row_of(self, track):
        view = self.window.timeline2 if track.get("canvas") == 1 else self.window.timeline
        rows, _ = view.lane_rows(self.window.project.tracks)
        return rows[self.window.project.tracks.index(track)]

    def test_selected_bars_cross_canvases_together_and_back(self):
        window = self.window
        self.open_second_canvas()
        # Two bars out of drums, each on its own row of the main canvas.
        window._move_block(0, 2.0, 2.0, 2.0, False, 2, 1)
        tail = next(i for i, t in enumerate(window.project.tracks) if t["name"] == "drums" and t["start"] >= 4.0)
        window._move_block(tail, 0.0, 2.0, 4.0, False, len(window.project.tracks), 2)
        first, second = self.track("drums · bar 2"), self.track("drums · bar 3")
        for track in (first, second):
            self.clips(window.timeline)[window.project.tracks.index(track)].setSelected(True)
        self.assertEqual(len(window._selected_indices()), 2)
        rows_before = (self.row_of(first), self.row_of(second))
        clip = self.clips(window.timeline)[window.project.tracks.index(first)]
        grab = QPointF(clip.pos().x() + 30, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(window.timeline, grab, window.timeline2, QPointF(grab.x() + PIXELS_PER_SECOND, 60))
        for track, start in ((first, 3.0), (second, 5.0)):
            self.assertEqual(track["canvas"], 1)
            self.assertAlmostEqual(track["start"], start, places=2)
        self.assertEqual(self.row_of(second) - self.row_of(first), rows_before[1] - rows_before[0])
        # Both back to the main canvas, onto the drums row.
        for track in (first, second):
            self.clips(window.timeline2)[window.project.tracks.index(track)].setSelected(True)
        clip = self.clips(window.timeline2)[window.project.tracks.index(first)]
        grab = QPointF(clip.pos().x() + 30, clip.pos().y() + clip.rect().height() * 0.75)
        drums = self.clips(window.timeline)[window.project.tracks.index(self.track("bass"))]
        self.drag(window.timeline2, grab, window.timeline, QPointF(grab.x(), drums.pos().y() + 20))
        for track in (first, second):
            self.assertNotIn("canvas", track)
        self.assertEqual(self.row_of(first), self.row_of(self.track("bass")))

    def test_a_bar_sharing_a_row_crosses_alone_and_again(self):
        window = self.window
        self.open_second_canvas()
        window._move_track_to_canvas(1, 1)  # bass on canvas 2
        # Bar 2 of drums onto the bass row on canvas 2.
        drums = self.clips(window.timeline)[0]
        grab = QPointF(drums.pos().x() + 2.3 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        bass = self.clips(window.timeline2)[window.project.tracks.index(self.track("bass"))]
        self.drag(window.timeline, grab, window.timeline2,
                  QPointF(grab.x() + 3 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() * 0.75),
                  Qt.KeyboardModifier.ControlModifier)
        piece = self.track("drums · bar 2")
        self.assertEqual(piece["canvas"], 1)
        self.assertEqual(self.row_of(piece), self.row_of(self.track("bass")))  # joined the bass row
        for target, expected in ((window.timeline, 0), (window.timeline2, 1), (window.timeline, 0)):
            source = window.timeline2 if piece.get("canvas") == 1 else window.timeline
            clip = self.clips(source)[window.project.tracks.index(piece)]
            point = QPointF(clip.pos().x() + 30, clip.pos().y() + clip.rect().height() * 0.75)
            self.drag(source, point, target, target.mapToScene(QPoint(60, target.viewport().height() - 8)))
            self.assertEqual(piece.get("canvas", 0), expected)
            self.assertEqual(self.track("bass")["canvas"], 1)  # its row-mate stays


    def box(self, view, scene_from, scene_to, modifiers=Qt.KeyboardModifier.NoModifier):
        self.drag(view, scene_from, view, scene_to, modifiers)

    def test_box_picks_bars_across_tracks_and_drags_them_together(self):
        window = self.window
        view = window.timeline
        view.resize(1400, 400)
        self.app.processEvents()
        drums, bass = self.clips(view)[0], self.clips(view)[1]
        # A box over bars 2 and 3 (2-6 s) of both tracks, from empty space below.
        below = QPointF(5.5 * PIXELS_PER_SECOND, bass.pos().y() + bass.rect().height() + 30)
        above = QPointF(2.5 * PIXELS_PER_SECOND, drums.pos().y() + 10)
        self.box(view, below, above)
        self.assertEqual(view.picks(), {0: {1, 2}, 1: {1, 2}})
        self.assertEqual(window._selected_indices(), [])  # bars, not whole tracks
        # Drag one picked bar one bar later: all four go, on their own rows below.
        clip = self.clips(view)[0]
        grab = QPointF(clip.pos().x() + 2.3 * PIXELS_PER_SECOND, clip.pos().y() + clip.rect().height() * 0.75)
        lowest = self.clips(view)[1]
        self.drag(view, grab, view, QPointF(grab.x() + 2 * PIXELS_PER_SECOND, lowest.pos().y() + lowest.rect().height() + 40))
        names = sorted(t["name"] for t in window.project.tracks)
        self.assertEqual(names, ["bass", "bass", "bass · bar 2", "bass · bar 3", "drums", "drums", "drums · bar 2", "drums · bar 3"])
        for name, start in (("drums · bar 2", 4.0), ("drums · bar 3", 6.0), ("bass · bar 2", 4.0), ("bass · bar 3", 6.0)):
            self.assertAlmostEqual(self.track(name)["start"], start, places=2)
        # The rest of each track stays, with a gap: 0-2 s and 6-8 s.
        drums_parts = sorted((t["start"], t["duration"]) for t in window.project.tracks if t["name"] == "drums")
        self.assertEqual([(round(a, 2), round(b, 2)) for a, b in drums_parts], [(0.0, 2.0), (6.0, 2.0)])
        # Bars of one track stay on one row; the two tracks' bars keep their order.
        rows, _ = view.lane_rows(window.project.tracks)
        row = lambda name: rows[window.project.tracks.index(self.track(name))]
        self.assertEqual(row("drums · bar 2"), row("drums · bar 3"))
        self.assertEqual(row("bass · bar 2"), row("drums · bar 2") + 1)
        # Audio: bar 2 of drums (loud) now plays at 4-6 s.
        self.assertGreater(float(window.engine.render(int(4.5 * OUTPUT_RATE), 4)[0, 0]), 0.55)
        # The moved bars stay picked and go on together to canvas 2.
        self.open_second_canvas()
        self.assertEqual(len(view.picked_blocks()), 4)
        clip = self.clips(view)[window.project.tracks.index(self.track("drums · bar 2"))]
        grab = QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(view, grab, window.timeline2, QPointF(grab.x(), 60))
        for name in ("drums · bar 2", "drums · bar 3", "bass · bar 2", "bass · bar 3"):
            self.assertEqual(self.track(name).get("canvas"), 1, name)
        # And back.
        clip = self.clips(window.timeline2)[window.project.tracks.index(self.track("bass · bar 3"))]
        grab = QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.75)
        self.drag(window.timeline2, grab, view, QPointF(grab.x(), view.mapToScene(QPoint(0, view.viewport().height() - 20)).y()))
        for name in ("drums · bar 2", "drums · bar 3", "bass · bar 2", "bass · bar 3"):
            self.assertNotIn("canvas", self.track(name), name)
        # One undo per move.
        window._undo()
        self.assertEqual(self.track("drums · bar 2").get("canvas"), 1)

    def test_ctrl_box_adds_bars_and_a_click_elsewhere_clears(self):
        window = self.window
        view = window.timeline
        view.resize(1400, 400)
        self.app.processEvents()
        drums, bass = self.clips(view)[0], self.clips(view)[1]
        under = bass.pos().y() + bass.rect().height() + 30
        self.box(view, QPointF(2.5 * PIXELS_PER_SECOND, under), QPointF(2.6 * PIXELS_PER_SECOND, drums.pos().y() + 10))
        self.assertEqual(view.picks(), {0: {1}, 1: {1}})
        self.box(view, QPointF(6.5 * PIXELS_PER_SECOND, under), QPointF(6.6 * PIXELS_PER_SECOND, bass.pos().y() + 10),
                 Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(view.picks(), {0: {1}, 1: {1, 3}})
        # A plain press on an unpicked part of a clip clears the bars.
        point = QPointF(drums.pos().x() + 0.5 * PIXELS_PER_SECOND, drums.pos().y() + drums.rect().height() * 0.75)
        self.drag(view, point, view, point)
        self.assertEqual(view.picks(), {})

    def test_dragged_selection_is_outlined_on_the_other_canvas(self):
        # Picked bar blocks of two stems move to canvas 2 together.
        window = self.window
        self.open_second_canvas()
        view = window.timeline
        view.set_picks({0: {0}, 1: {0}})
        clip = self.clips(view)[0]
        start = view.mapFromScene(QPointF(clip.pos().x() + 30, clip.pos().y() + clip.rect().height() * 0.75))
        target = window.timeline2.viewport().mapToGlobal(QPoint(80, 60))
        end = view.viewport().mapFromGlobal(target)
        for kind, at, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, (start + end) / 2, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
        ):
            QApplication.sendEvent(view.viewport(), QMouseEvent(kind, QPointF(at), QPointF(view.viewport().mapToGlobal(at)),
                                                                Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
            self.app.processEvents()
        # Both dragged blocks are outlined on canvas 2, one row apart.
        ghosts = window.timeline2._ghosts
        self.assertEqual(len(ghosts), 2)
        self.assertGreater(ghosts[1].rect().top(), ghosts[0].rect().top())
        QApplication.sendEvent(view.viewport(), QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(end), QPointF(target),
                                                            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual(window.timeline2._ghosts, [])
        moved = sorted(track["name"] for track in window.project.tracks if track.get("canvas") == 1)
        self.assertEqual(moved, ["bass · bar 1", "drums · bar 1"])
        self.assertNotIn("canvas", self.track("drums"))  # the rest stays
        self.assertNotIn("canvas", self.track("bass"))


if __name__ == "__main__":
    unittest.main()
