"""Stem blocks played on their own: Play/Pause per block, several at once,
apart from the transport, with the main playhead hidden; Canvas 2 at launch."""

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

from soniclab.core.audio_engine import LIMIT_CEILING, OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import MainWindow
from soniclab.ui.timeline import FOLLOW_CENTER, BarPlayButton, ClipItem, TimelineView

BLOCK = 1024
LEVELS = {"vocals": 0.1, "drums": 0.2, "bass": 0.4}


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


class EngineVoiceTests(unittest.TestCase):
    """The mixing itself, without the UI."""

    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        ramp = np.arange(OUTPUT_RATE * 4, dtype=np.float32) / (OUTPUT_RATE * 400)
        self.engine.add_clip(np.column_stack((ramp, ramp)), 0.0)  # sample n = n / (rate*400)
        self.engine.add_clip(np.full((OUTPUT_RATE * 4, 2), 0.25, np.float32), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_blocks(self, count, frames=BLOCK):
        out = np.zeros((frames, 2), np.float32)
        blocks = []
        for _ in range(count):
            self.engine._callback(out, frames, None, None)
            blocks.append(out.copy())
        return np.concatenate(blocks)

    def test_resume_continues_where_it_paused(self):
        self.engine.preview_play([("ramp", 0, 1.0, 3.0)])
        first = self.run_blocks(5)
        self.assertAlmostEqual(float(first[0, 0]), OUTPUT_RATE / (OUTPUT_RATE * 400), places=7)
        self.engine.preview_pause("ramp")
        silent = self.run_blocks(3)
        self.assertFalse(silent.any())
        self.engine.preview_play([("ramp", 0, 1.0, 3.0)])
        resumed = self.run_blocks(1)
        expected = (OUTPUT_RATE + 5 * BLOCK) / (OUTPUT_RATE * 400)
        self.assertAlmostEqual(float(resumed[0, 0]), expected, places=7)
        self.assertEqual(self.engine._voices["ramp"].cursor, OUTPUT_RATE + 6 * BLOCK)

    def test_a_voice_ends_on_its_last_sample(self):
        frames = OUTPUT_RATE // 10
        self.engine.preview_play([("b", 1, 0.5, 0.6)])
        out = self.run_blocks(2, frames + 100)
        self.assertEqual(int(np.count_nonzero(out[:, 0])), frames)
        self.assertEqual(self.engine.previews(), {})

    def test_voices_start_together_on_the_same_sample(self):
        self.engine.preview_play([("a", 0, 1.0, 2.0), ("b", 1, 1.0, 2.0)])
        out = self.run_blocks(1)
        np.testing.assert_allclose(out[:, 0], 0.25 + (OUTPUT_RATE + np.arange(BLOCK)) / (OUTPUT_RATE * 400), atol=1e-6)

    def test_voices_play_alongside_the_transport_without_moving_it(self):
        self.engine.seek(2.0)
        self.engine._playing = True
        self.engine.preview_play([("b", 1, 0.0, 1.0)])
        out = self.run_blocks(1)
        ramp = (2 * OUTPUT_RATE + np.arange(BLOCK)) / (OUTPUT_RATE * 400)
        np.testing.assert_allclose(out[:, 0], ramp + 0.25 + 0.25, atol=1e-6)  # song + block
        self.assertEqual(self.engine.position, 2.0 + BLOCK / OUTPUT_RATE)

    def test_the_limiter_holds_loud_stacks_under_full_scale(self):
        loud = np.full((OUTPUT_RATE * 2, 2), 0.45, np.float32)
        for _ in range(4):
            self.engine.add_clip(loud, 0.0)
        self.engine.preview_play([(index, index, 0.0, 1.5) for index in range(2, 6)])  # 1.8 summed
        out = self.run_blocks(40)
        self.assertLessEqual(float(np.max(np.abs(out))), LIMIT_CEILING + 1e-6)
        self.assertGreater(float(np.min(np.abs(out[:, 0]))), 0.9)  # steady, no dropouts
        self.engine.preview_stop_all()
        self.engine.preview_play([("quiet", 1, 0.0, 1.0)])
        quiet = self.run_blocks(30)
        self.assertAlmostEqual(float(quiet[-1, 0]), 0.25, places=5)  # released fully

    def test_no_limiter_and_no_change_without_voices(self):
        self.engine._playing = True
        self.engine.seek(0.5)
        live = self.run_blocks(4)
        np.testing.assert_array_equal(live, self.engine.render(int(0.5 * OUTPUT_RATE), 4 * BLOCK))

    def test_close_drops_every_voice(self):
        self.engine.preview_play([("a", 0, 0.0, 2.0)])
        self.engine.close()
        self.assertEqual(self.engine.previews(), {})


class StemBlockWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.beat_jobs = _Jobs()
        self.window.resize(1400, 900)
        for name, level in LEVELS.items():
            audio = np.full((OUTPUT_RATE * 8, 2), level, dtype=np.float32)
            self.window.project.tracks.append(
                {"name": name, "path": None, "start": 0.0, "duration": 8.0, "offset": 0.0, "_audio": audio}
            )
        self.window.project.bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.window.show()
        self.app.processEvents()
        self.engine = self.window.engine
        self.engine._capture_enabled = False
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.engine.preview_stop_all()
        for timeline in (self.window.timeline, self.window.timeline2):
            timeline.clear_scene()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    # -- helpers

    def press(self, track, bar):
        """Click a block's ▶ (or ⏸)."""
        self.window.timeline.bar_play_requested.emit(bar, track)

    def button(self, track, bar) -> BarPlayButton:
        for item in self.window.timeline.scene_obj.items():
            if isinstance(item, ClipItem) and item.clip_index == track:
                for child in item.childItems():
                    if isinstance(child, BarPlayButton) and child.index == bar:
                        return child
        raise AssertionError(f"no button for {track}, {bar}")

    def mix(self, blocks=1):
        out = np.zeros((BLOCK, 2), np.float32)
        levels = []
        for _ in range(blocks):
            self.engine._callback(out, BLOCK, None, None)
            levels.append(float(out[-1, 0]))
            self.window._animate_playheads()
        return levels[-1]

    # -- Play / Pause

    def test_play_becomes_pause_and_back(self):
        self.assertFalse(self.button(0, 1).playing)
        self.press(0, 1)
        self.assertTrue(self.button(0, 1).playing)
        self.assertAlmostEqual(self.mix(), LEVELS["vocals"], places=5)
        self.press(0, 1)
        self.assertFalse(self.button(0, 1).playing)
        self.assertEqual(self.engine.previews()[(0, 1)][0], False)  # paused, not gone
        self.assertEqual(self.mix(), 0.0)
        self.press(0, 1)
        self.assertTrue(self.button(0, 1).playing)

    def test_resume_continues_from_the_paused_place(self):
        self.press(1, 0)
        self.mix(10)
        self.press(1, 0)
        paused_at = self.engine._voices[(1, 0)].cursor
        self.mix(5)
        self.assertEqual(self.engine._voices[(1, 0)].cursor, paused_at)
        self.press(1, 0)
        self.mix(1)
        self.assertEqual(self.engine._voices[(1, 0)].cursor, paused_at + BLOCK)

    def test_the_button_resets_when_the_block_finishes(self):
        self.press(2, 3)
        self.assertTrue(self.button(2, 3).playing)
        self.mix(2 * OUTPUT_RATE // BLOCK + 2)
        self.assertNotIn((2, 3), self.engine.previews())
        self.assertFalse(self.button(2, 3).playing)
        self.assertTrue(self.window.timeline.playhead.isVisible())
        self.assertFalse(self.window.frame_timer.isActive())

    def test_another_block_leaves_the_first_button_alone(self):
        self.press(0, 1)
        self.press(1, 2)
        self.assertTrue(self.button(0, 1).playing)
        self.assertTrue(self.button(1, 2).playing)
        self.assertFalse(self.button(0, 2).playing)
        self.assertFalse(self.button(1, 1).playing)

    def test_a_failed_start_shows_play_and_an_error(self):
        def broken():
            raise RuntimeError("no output device")

        with patch.object(self.engine, "_ensure_stream", broken), \
                patch.object(QMessageBox, "critical") as critical:
            self.press(0, 0)
        critical.assert_called_once()
        self.assertFalse(self.button(0, 0).playing)
        self.assertEqual(self.engine.previews(), {})

    # -- several at once

    def test_three_blocks_play_together_and_pause_or_stop_alone(self):
        self.press(0, 1)
        self.press(1, 1)
        self.press(2, 1)
        self.assertAlmostEqual(self.mix(), sum(LEVELS.values()), places=5)
        self.press(0, 1)  # pause vocals
        self.assertAlmostEqual(self.mix(), LEVELS["drums"] + LEVELS["bass"], places=5)
        states = self.engine.previews()
        self.assertEqual([states[(i, 1)][0] for i in range(3)], [False, True, True])
        self.window._stop_block((1, 1))  # stop drums
        self.assertAlmostEqual(self.mix(), LEVELS["bass"], places=5)
        self.assertTrue(self.button(2, 1).playing)
        self.assertNotIn((1, 1), self.engine.previews())
        self.assertIn((0, 1), self.engine.previews())  # vocals still paused

    def test_picked_blocks_start_on_the_same_sample(self):
        self.window.timeline.set_picks({0: {2}, 1: {2}, 2: {2}})
        self.window._play_picked_blocks()
        cursors = {voice.cursor for voice in self.engine._voices.values()}
        self.assertEqual(cursors, {int(4.0 * OUTPUT_RATE)})
        self.mix(3)
        self.assertEqual(len({voice.cursor for voice in self.engine._voices.values()}), 1)

    # -- apart from the transport

    def test_transport_playhead_and_scroll_are_left_alone(self):
        window = self.window
        window._set_follow_mode(FOLLOW_CENTER)
        window.timeline.zoom_horizontal(6.0)
        self.engine.seek(5.25)
        window._refresh_transport()
        bar = window.timeline.horizontalScrollBar()
        bar.setValue(400)
        x = window.timeline.playhead_x()
        before = (self.engine.position, self.engine.loop, self.engine.playing, list(window._canvas_positions))
        self.press(0, 0)
        self.press(1, 3)
        self.assertFalse(window.timeline.playhead.isVisible())
        values = set()
        for _ in range(60):
            self.mix()
            window._refresh_transport()
            values.add(bar.value())
        self.assertEqual(values, {400})
        self.assertEqual(window.follow_mode, FOLLOW_CENTER)
        after = (self.engine.position, self.engine.loop, self.engine.playing, list(window._canvas_positions))
        self.assertEqual(after, before)
        self.assertEqual(window.timeline.playhead_x(), x)
        window._stop_blocks()
        self.assertTrue(window.timeline.playhead.isVisible())
        self.assertEqual(window.timeline.playhead_x(), x)
        self.assertEqual(bar.value(), 400)

    def test_the_song_playing_keeps_its_playhead_shown(self):
        self.press(0, 0)
        self.assertFalse(self.window.timeline.playhead.isVisible())
        self.engine._playing = True
        self.window._animate_playheads()
        self.assertTrue(self.window.timeline.playhead.isVisible())
        self.engine._playing = False

    def test_main_stop_also_stops_blocks(self):
        self.press(0, 0)
        self.press(1, 0)
        self.window._stop()
        self.assertEqual(self.engine.previews(), {})
        self.assertFalse(self.button(0, 0).playing)

    def test_deleting_a_track_stops_its_voices(self):
        self.press(0, 1)
        self.window._select_track(0)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window._delete_track()
        self.app.processEvents()
        self.assertNotIn((0, 1), self.engine.previews())

    # -- timing

    def test_parallel_voices_fit_well_inside_the_audio_deadline(self):
        for track in range(3):
            for bar in range(4):
                self.press(track, bar)
        self.engine._playing = True  # the song plays too
        out = np.zeros((512, 2), np.float32)
        costs = []
        for _ in range(300):
            started = time.perf_counter()
            self.engine._callback(out, 512, None, None)
            costs.append(time.perf_counter() - started)
        self.engine._playing = False
        budget = 512 / OUTPUT_RATE
        self.assertLess(float(np.percentile(costs, 99)), budget / 3)

    def test_a_frame_with_block_cursors_stays_cheap(self):
        for track in range(3):
            self.press(track, 1)
        costs = []
        out = np.zeros((BLOCK, 2), np.float32)
        for _ in range(60):
            self.engine._callback(out, BLOCK, None, None)
            started = time.perf_counter()
            self.window._animate_playheads()
            self.app.processEvents()
            costs.append(time.perf_counter() - started)
        self.assertLess(float(np.median(costs)), 0.016)


class CanvasTwoLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_open_at_every_launch_with_one_instance(self):
        for _ in range(2):
            window = MainWindow()
            window.show()
            self.app.processEvents()
            self.assertTrue(window.timeline2.isVisible())
            self.assertTrue(window.second_canvas_action.isChecked())
            for _toggle in range(3):
                window.second_canvas_action.setChecked(False)
                window.second_canvas_action.setChecked(True)
            self.app.processEvents()
            views = window.findChildren(TimelineView)
            self.assertEqual(sorted(view.canvas for view in views), [0, 1])
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
