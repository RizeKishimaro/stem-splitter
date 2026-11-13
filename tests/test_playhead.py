import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SONICLAB_DEMUCS_PYTHON"] = ""
# Keep QSettings (recent projects, toggles) out of the real ~/.config.
if "SONICLAB_TEST_CONFIG" not in os.environ:
    os.environ["SONICLAB_TEST_CONFIG"] = os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="soniclab-test-config-")

import numpy as np
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem

from soniclab.core import audio_engine
from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import (
    FOLLOW_CENTER,
    FOLLOW_OFF,
    FOLLOW_PAGE,
    RULER_HEIGHT,
    WAVE_TILES_KEPT,
    ClipItem,
    PlayheadItem,
    TimelineView,
)

BLOCK = 512


class _Clock:
    """Stands in for time.monotonic inside the engine."""

    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def _latency(seconds):
    return SimpleNamespace(currentTime=1.0, outputBufferDacTime=1.0 + seconds)


class HeardPositionTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        self.engine.add_clip(np.zeros((OUTPUT_RATE * 4, 2), np.float32), 0.0)
        self.clock = _Clock()
        patcher = patch.object(audio_engine.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _block(self, latency=0.0):
        out = np.zeros((BLOCK, 2), np.float32)
        self.engine._callback(out, BLOCK, _latency(latency), None)

    def test_stopped_it_is_the_position(self):
        self.engine.seek(1.25)
        self.assertEqual(self.engine.heard_position(), 1.25)

    def test_interpolates_between_blocks_and_allows_for_latency(self):
        self.engine.seek(1.0)
        self.engine._playing = True
        self._block(latency=0.02)  # sounds at 100.02
        self.assertAlmostEqual(self.engine.heard_position(100.0), 1.0)  # not yet heard
        self.assertAlmostEqual(self.engine.heard_position(100.025), 1.005, places=6)
        # Late next block: hold at the end of this one instead of running on.
        self.assertAlmostEqual(self.engine.heard_position(100.5), 1.0 + BLOCK / OUTPUT_RATE, places=6)
        self.clock.now += BLOCK / OUTPUT_RATE
        self._block(latency=0.02)
        later = 100.02 + 1.5 * BLOCK / OUTPUT_RATE
        self.assertAlmostEqual(self.engine.heard_position(later), 1.0 + 1.5 * BLOCK / OUTPUT_RATE, places=6)

    def test_loop_restart_lands_exactly_on_the_loop_start(self):
        start, end = 1.0, 1.0 + 300 / OUTPUT_RATE
        self.engine.set_loop(start, end)
        self.engine.seek(start)
        self.engine._playing = True
        for _ in range(50):  # many restarts: no drift
            self._block()
            self.clock.now += BLOCK / OUTPUT_RATE
        anchor = self.clock.now - BLOCK / OUTPUT_RATE
        frame = self.engine._heard[-1][1][0][0]
        # 300 frames to the loop end, then 212 from the loop start.
        seconds = self.engine.heard_position(anchor + (300 - frame + int(start * OUTPUT_RATE) + 5) / OUTPUT_RATE)
        self.assertGreaterEqual(seconds, start)
        self.assertLess(seconds, end)
        for t in np.linspace(anchor, anchor + BLOCK / OUTPUT_RATE, 40):
            self.assertTrue(start <= self.engine.heard_position(t) <= end)

    def test_a_seek_while_playing_shows_the_new_place_at_once(self):
        self.engine._playing = True
        self._block(latency=0.05)
        self.engine.seek(3.0)
        self.assertEqual(self.engine.heard_position(), 3.0)
        self._block(latency=0.05)
        self.assertEqual(self.engine.heard_position(self.clock.now + 0.01), 3.0)  # held until heard
        self.assertGreater(self.engine.heard_position(self.clock.now + 0.06), 3.0)

    def test_pause_and_stop_drop_the_history(self):
        self.engine._playing = True
        self._block()
        self.engine.pause()
        self.assertEqual(self.engine.heard_position(), self.engine.position)
        self.engine.stop()
        self.assertEqual(self.engine.heard_position(), 0.0)


class _ViewCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = TimelineView()
        self.view.resize(600, 260)
        audio = np.full((OUTPUT_RATE * 60, 2), 0.3, dtype=np.float32)
        self.tracks = [{"name": "Long", "start": 0.0, "_audio": audio, "duration": 60.0}]
        self.view.rebuild(self.tracks, lambda *_args: None, 120, "4/4")
        self.view.show()
        self.app.processEvents()
        self.bar = self.view.horizontalScrollBar()

    def tearDown(self):
        self.view.clear_scene()
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def left(self):
        return self.view.mapToScene(0, 0).x()


class FollowTests(_ViewCase):
    def test_one_shared_playhead_drawn_above_the_clips(self):
        heads = [item for item in self.view.scene_obj.items() if isinstance(item, PlayheadItem)]
        self.assertEqual(len(heads), 1)
        clip = next(item for item in self.view.scene_obj.items() if isinstance(item, ClipItem))
        self.assertGreater(heads[0].zValue(), clip.zValue())
        self.view.set_playhead(2.0)
        self.assertEqual(self.view.playhead.line().x1(), 2.0 * self.view.pixels_per_second)
        self.assertEqual(self.view.playhead.line().y1(), 0)

    def test_off_never_scrolls(self):
        for seconds in np.arange(0.0, 30.0, 0.25):
            self.view.set_playhead(seconds)
            self.view.follow_playhead(FOLLOW_OFF, 0.016)
        self.assertEqual(self.bar.value(), 0)

    def test_follow_glides_at_the_right_edge(self):
        width = self.view.viewport().width()
        self.view.set_playhead(1.0)
        self.view.follow_playhead(FOLLOW_PAGE, 0.016)
        self.assertEqual(self.bar.value(), 0)  # still on screen: no scroll
        steps = []
        seconds = 1.0
        while seconds < 20.0:
            seconds += 1 / 60
            self.view.set_playhead(seconds)
            before = self.bar.value()
            self.view.follow_playhead(FOLLOW_PAGE, 1 / 60)
            steps.append(self.bar.value() - before)
            x = self.view.playhead_x() - self.left()
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, width)
        self.assertGreater(self.bar.value(), 0)
        self.assertLessEqual(max(steps), 3)  # a glide, never a page jump

    def test_follow_eases_back_to_a_playhead_behind_the_view(self):
        self.bar.setValue(2000)
        self.view.set_playhead(1.0)
        values = []
        for _ in range(60):
            self.view.follow_playhead(FOLLOW_PAGE, 1 / 60)
            values.append(self.bar.value())
        self.assertLess(values[0], 2000)
        self.assertGreater(values[0], 200)  # eased, not snapped
        self.assertLess(values[-1], 100)
        self.assertEqual(values, sorted(values, reverse=True))

    def test_center_holds_the_playhead_in_the_middle(self):
        width = self.view.viewport().width()
        self.view.set_playhead(15.0)
        for _ in range(90):
            self.view.follow_playhead(FOLLOW_CENTER, 1 / 60)
        seconds = 15.0
        for _ in range(60):
            seconds += 1 / 60
            self.view.set_playhead(seconds)
            self.view.follow_playhead(FOLLOW_CENTER, 1 / 60)
            self.assertAlmostEqual(self.view.playhead_x() - self.left(), width / 2, delta=1.0)

    def test_a_manual_scroll_pauses_following(self):
        self.view.set_playhead(20.0)
        self.view._note_user_scroll()
        self.view.follow_playhead(FOLLOW_CENTER, 0.016)
        self.assertEqual(self.bar.value(), 0)


class RulerTests(_ViewCase):
    def test_dragging_the_handle_scrubs_without_moving_clips(self):
        sought = []
        self.view.seek_requested.connect(sought.append)
        clip = next(item for item in self.view.scene_obj.items() if isinstance(item, ClipItem))
        clip_x = clip.pos().x()
        pps = self.view.pixels_per_second
        self.view.set_playhead(1.0)
        handle = self.view.mapFromScene(QPointF(pps, 4))
        self.assertTrue(self.view.on_playhead_handle(handle))
        viewport = self.view.viewport()
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, handle)
        for seconds in (1.5, 2.0, 2.5):
            QTest.mouseMove(viewport, self.view.mapFromScene(QPointF(seconds * pps, 60)))
        QTest.mouseRelease(
            viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(2.5 * pps, 60)),
        )
        self.assertAlmostEqual(sought[-1], 2.5, places=2)
        self.assertGreaterEqual(len(sought), 3)
        self.assertAlmostEqual(self.view.playhead_x(), 2.5 * pps, delta=1.5)
        self.assertEqual(clip.pos().x(), clip_x)
        self.assertEqual(self.bar.value(), 0)

    def test_clicking_the_ruler_moves_the_playhead(self):
        sought = []
        self.view.seek_requested.connect(sought.append)
        pps = self.view.pixels_per_second
        QTest.mouseClick(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            self.view.mapFromScene(QPointF(3 * pps, RULER_HEIGHT - 4)),
        )
        self.assertAlmostEqual(sought[-1], 3.0, places=2)
        self.assertAlmostEqual(self.view.playhead_x(), 3 * pps, delta=1.5)


class WaveTileTests(_ViewCase):
    def test_paint_draws_only_the_exposed_strip_from_cached_tiles(self):
        self.view.zoom_horizontal(4.0)  # 21 600 px: many tiles
        clip = next(item for item in self.view.scene_obj.items() if isinstance(item, ClipItem))
        self.assertTrue(clip.flags() & ClipItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)
        image = QImage(64, 128, QImage.Format.Format_ARGB32)
        calls = []
        original = clip._rasterize

        def counting(start, stop):
            calls.append((start, stop))
            return original(start, stop)

        clip._rasterize = counting
        option = QStyleOptionGraphicsItem()
        for _ in range(30):
            for left in (5000.0, 5003.0, 9000.0):
                option.exposedRect = clip.rect().adjusted(left, 0, 0, 0)
                option.exposedRect.setWidth(8)
                painter = QPainter(image)
                clip.paint(painter, option)
                painter.end()
        self.assertEqual(len(calls), 2)  # one tile each, drawn once
        self.assertTrue(all(stop - start <= 1024 for start, stop in calls))
        for left in range(0, 20000, 1024):
            option.exposedRect = clip.rect().adjusted(left, 0, 0, 0)
            option.exposedRect.setWidth(4)
            painter = QPainter(image)
            clip.paint(painter, option)
            painter.end()
        self.assertLessEqual(len(clip._tiles), WAVE_TILES_KEPT)


class WindowFollowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        audio = np.full((OUTPUT_RATE * 40, 2), 0.2, dtype=np.float32)
        self.window.project.tracks.append(
            {"name": "Loop me", "path": None, "start": 0.0, "offset": 0.0, "duration": 40.0, "_audio": audio}
        )
        self.window.engine.add_clip(audio, 0.0)
        self.window.engine._capture_enabled = False
        self.window._refresh_timeline()
        self.window.resize(900, 600)
        self.window.show()
        self.app.processEvents()
        self.window.timeline.zoom_horizontal(4.0)

    def tearDown(self):
        self.window.engine._playing = False
        self.window.timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _play_blocks(self, count):
        engine = self.window.engine
        out = np.zeros((BLOCK, 2), np.float32)
        for _ in range(count):
            engine._callback(out, BLOCK, None, None)
            self.window._animate_playheads()

    def test_looping_one_track_with_follow_off_keeps_the_canvas_still(self):
        window = self.window
        self.assertEqual(window.follow_mode, FOLLOW_OFF)
        window._engage_loop(6.0, 8.0)
        bar = window.timeline.horizontalScrollBar()
        bar.setValue(1500)
        window.engine._playing = True
        values = set()
        positions = []
        for _ in range(400):  # several loop restarts
            self._play_blocks(1)
            values.add(bar.value())
            positions.append(window.timeline.playhead_x() / window.timeline.pixels_per_second)
        self.assertEqual(values, {1500})
        self.assertTrue(all(6.0 - 1e-6 <= seconds <= 8.0 + 1e-6 for seconds in positions))
        self.assertLess(min(positions), 6.05)

    def test_center_scrolls_and_off_returns_to_still(self):
        window = self.window
        window._set_follow_mode(FOLLOW_CENTER)
        bar = window.timeline.horizontalScrollBar()
        window.engine.seek(10.0)
        window.engine._playing = True
        self._play_blocks(200)
        self.assertGreater(bar.value(), 0)
        window._set_follow_mode(FOLLOW_OFF)
        held = bar.value()
        self._play_blocks(50)
        self.assertEqual(bar.value(), held)

    def test_toolbar_button_cycles_and_remembers_the_mode(self):
        window = self.window
        window.follow_btn.click()
        self.assertEqual(window.follow_mode, FOLLOW_PAGE)
        self.assertIn("Follow", window.follow_btn.text())
        self.assertEqual(QSettings("SonicLab", "SonicLab").value("view/followPlayhead"), FOLLOW_PAGE)
        window.follow_actions[FOLLOW_CENTER].trigger()
        self.assertEqual(window.follow_mode, FOLLOW_CENTER)
        window._cycle_follow_mode()
        self.assertEqual(window.follow_mode, FOLLOW_OFF)

    def test_frame_timer_runs_only_while_playing(self):
        window = self.window
        self.assertFalse(window.frame_timer.isActive())
        window.engine._playing = True
        window._refresh_transport()
        self.assertTrue(window.frame_timer.isActive())
        window.engine._playing = False
        window._animate_playheads()
        self.assertFalse(window.frame_timer.isActive())


if __name__ == "__main__":
    unittest.main()
