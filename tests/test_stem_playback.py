"""Split-stem playback: play buttons, cached splits, independent voices,
pause and resume.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_stem_playback
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from soniclab.core.audio_engine import AudioEngine, OUTPUT_RATE
from soniclab.ui.main_window import stem_block
from soniclab.ui.timeline import BarPlayButton
from tests.test_stem_sync import BLOCK, SharedClockWindowTests, ramp, run

ORDER = ("drums", "bass", "other", "vocals")


class PausePositionEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        for scale in (1, 2):
            self.engine.add_clip(ramp(scale), 0.0)
        patcher = patch.object(self.engine, "_ensure_stream", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_paused_block_resumes_where_it_was_heard(self):
        self.engine.preview_play([("a", 0, 1.0, 6.0)])
        run(self.engine, 20)
        heard = self.engine.previews()["a"][1]
        self.engine.preview_pause("a")
        self.assertAlmostEqual(self.engine.previews()["a"][1], heard, delta=2 * BLOCK / OUTPUT_RATE)
        held = self.engine.previews()["a"][1]
        run(self.engine, 10)
        self.assertEqual(self.engine.previews()["a"], (False, held))
        self.engine.preview_resume(["a"])
        self.assertAlmostEqual(self.engine.previews()["a"][1], held, places=3)

    def test_a_stem_resumes_at_its_songs_place_and_a_paused_song_holds(self):
        song = [("a", 0, False, 0.0, 2.0), ("b", 1, False, 0.0, 2.0)]
        self.engine.stem_play("song", song, 0.0, loop=(0.0, 2.0))
        run(self.engine, 10)
        self.engine.preview_pause("b")
        run(self.engine, 30)  # the song plays on without b
        self.engine.preview_resume(["b"])
        heard = self.engine.previews()
        self.assertEqual(heard["a"], heard["b"])  # back on the song's place, not its own
        self.engine.preview_pause("a")
        self.engine.preview_pause("b")
        held = self.engine._groups["song"].cursor
        run(self.engine, 20)  # the whole song paused: its clock holds
        self.assertEqual(self.engine._groups["song"].cursor, held)
        self.engine.preview_resume(["a", "b"])
        run(self.engine, 1)
        self.assertEqual(self.engine._groups["song"].cursor, held + BLOCK)

    def test_stopping_everything_leaves_nothing_behind(self):
        self.engine.stem_play("g", [("s", 0)], 0.0, loop=(0.0, 1.0))
        self.engine.stem_play("h", [("l", 1)], 0.0, loop=(0.0, 1.0))
        run(self.engine, 4)
        self.engine.preview_stop_all()
        self.assertEqual(self.engine.previews(), {})
        self.assertEqual(self.engine._groups, {})
        self.assertIsNone(self.engine.sounding_time())
        run(self.engine, 2)  # an idle callback adds nothing
        self.assertEqual(self.engine._voices, {})


class StemPlaybackWindowTests(SharedClockWindowTests):
    def stems_of(self, tracks) -> dict:
        return {name: track["path"] for name, track in zip(ORDER, tracks)}

    def playing(self) -> set:
        return {key for key, state in self.engine.previews().items() if state[0]}

    def hidden_buttons(self) -> list:
        """Block ▶ that are hidden although their block is not selected."""
        found = []
        for view in self.window._timelines():
            chosen = view.selected_blocks()
            chosen = set(chosen) if len(chosen) > 1 else set()
            for clip in view._clip_items():
                for child in clip.childItems():
                    if isinstance(child, BarPlayButton) and not child.isVisible():
                        if (clip.clip_index, child.index) not in chosen:
                            found.append((view.canvas, clip.clip_index, child.index))
        return found

    def missing_buttons(self) -> list:
        """Blocks wide enough for a ▶ that have none."""
        found = []
        for view in self.window._timelines():
            for clip in view._clip_items():
                bars = {child.index for child in clip.childItems() if isinstance(child, BarPlayButton)}
                for span in view._spans_of(clip):
                    if float(span[2]) - float(span[1]) >= 24 and int(span[0]) not in bars:
                        found.append((view.canvas, clip.clip_index, int(span[0])))
        return found

    def test_new_stems_show_every_play_button(self):
        first = self.split("Song")
        self.window._finish_split_stems("Song", self.stems_of(first), ORDER, None)
        self.app.processEvents()
        self.assertEqual(self.missing_buttons(), [])
        self.assertEqual(self.hidden_buttons(), [])
        # Pointed at, not selected whole: a whole stem's ▶ would give way.
        self.assertEqual([view.selected_indices() for view in self.window._timelines()], [[], []])
        self.assertEqual(self.window.project.tracks[self.window._selected]["name"], "Song drums")

    def test_play_buttons_survive_refresh_canvas_transfer_and_reload(self):
        drums, bass, _other, vocals = self.split("Song")
        self.window._refresh_timeline()
        self.app.processEvents()
        self.assertEqual(self.missing_buttons(), [])
        view = self.window._view_for(self.index(bass))
        view.set_picks({self.index(bass): {1}})
        self.window._move_selected_to_other_canvas()
        self.app.processEvents()
        self.assertEqual(self.missing_buttons(), [])
        self.assertEqual(self.hidden_buttons(), [])
        path = os.path.join(self.root, "stems.soniclab")
        self.window.current_path = path
        self.window._save_project()
        self.window._load_project(path)
        self.app.processEvents()
        self.assertEqual(self.missing_buttons(), [])
        self.assertEqual(self.hidden_buttons(), [])
        # Reloaded blocks play on their own, not as a whole-track loop.
        self.press(0, 1)
        self.assertEqual(self.playing(), {(0, stem_block(1))})

    def test_a_whole_selected_stem_is_drawn_selected(self):
        drums, *_rest = self.split("Song")
        view = self.window._view_for(self.index(drums))
        self.window._select_track(self.index(drums))  # a header click selects it whole
        self.app.processEvents()
        clip = view._clip_for(self.index(drums))
        self.assertEqual(self.hidden_buttons(), [])  # every hidden ▶ is on a selected block
        self.assertTrue(any(button.isVisible() for button in view._combined_buttons))
        self.assertTrue(clip.isSelected())

    def test_a_cached_split_plays_pauses_and_resumes_like_a_fresh_one(self):
        fresh = self.split("Song")
        self.window._finish_split_stems("Song", self.stems_of(fresh), ORDER, None)  # same files again
        self.app.processEvents()
        tracks = self.window.project.tracks
        cached = tracks[len(fresh):]
        self.assertEqual(cached[0]["path"], fresh[0]["path"])
        a, b = self.index(fresh[0]), self.index(cached[1])
        self.press(a, 0)
        self.mix(10)
        self.press(b, 1)
        self.mix(10)
        self.assertEqual(self.playing(), {(a, stem_block(0)), (b, stem_block(1))})
        self.press(b, 1)  # pause the cached stem only
        held = self.engine.previews()[(b, stem_block(1))]
        self.assertFalse(held[0])
        self.assertIn((a, stem_block(0)), self.playing())
        self.mix(20)
        self.assertEqual(self.engine.previews()[(b, stem_block(1))], held)  # holds still
        self.press(b, 1)  # resume: back on the song's clock, in step with a
        states = self.engine.previews()
        self.assertAlmostEqual(states[(b, stem_block(1))][1] - states[(a, stem_block(0))][1], 2.0, places=3)
        self.assertEqual(self.playing(), {(a, stem_block(0)), (b, stem_block(1))})
        # One song, one clock: the cached stems joined the fresh split's.
        self.assertEqual(self.engine._voices[(a, stem_block(0))].group, self.engine._voices[(b, stem_block(1))].group)
        # The canvas shows what the engine plays.
        self.window._sync_previews()
        clip = self.window._view_for(b)._clip_for(b)
        button = next(c for c in clip.childItems() if isinstance(c, BarPlayButton) and c.index == 1)
        self.assertTrue(button.playing)

    def test_play_on_one_stem_leaves_the_others_playing(self):
        drums, bass, other, vocals = self.split("Song")
        started = []
        for track, bar in ((drums, 0), (bass, 1), (other, 2), (vocals, 3)):
            self.press(self.index(track), bar)
            started.append((self.index(track), stem_block(bar)))
            self.mix(3)
            self.assertEqual(self.playing(), set(started))
        # Stems of one split stay on one clock, a sample apart at most.
        mixed = self.mix(4)
        self.assertTrue(np.all(np.isfinite(mixed)))
        self.press(self.index(bass), 1)  # pausing one leaves the rest
        self.assertEqual(self.playing(), set(started) - {(self.index(bass), stem_block(1))})

    def test_pause_on_a_block_in_a_loop_pauses_that_voice_and_starts_no_other(self):
        drums, bass, _other, vocals = self.split("Song")
        view = self.window._view_for(self.index(drums))
        view.set_picks({self.index(drums): {0}, self.index(vocals): {0}})
        self.press(self.index(drums), 0)  # the selection as one loop
        self.mix(10)
        self.window._deselect_all()  # the loop plays on, unselected
        voices = set(self.engine.previews())
        drum, vocal = (self.index(drums), stem_block(0)), (self.index(vocals), stem_block(0))
        self.press(self.index(drums), 0)  # its ⏸
        self.assertEqual(set(self.engine.previews()), voices)  # no second voice
        self.assertFalse(self.engine.previews()[drum][0])
        self.assertTrue(self.engine.previews()[vocal][0])
        self.mix(10)
        self.press(self.index(drums), 0)  # ▶ again: back in step with the song
        states = self.engine.previews()
        self.assertTrue(states[drum][0])
        self.assertAlmostEqual(states[drum][1], states[vocal][1], places=3)

    def test_previews_clean_up_when_their_voices_end(self):
        drums, *_rest = self.split("Song")
        self.press(self.index(drums), 0)
        self.mix(2)
        self.assertTrue(self.window._preview_tracks)
        self.engine.preview_stop_all()
        self.window._sync_previews()
        self.assertEqual(self.window._preview_tracks, {})
        clip = self.window._view_for(self.index(drums))._clip_for(self.index(drums))
        self.assertEqual(clip._preview, {})


# The inherited shared-clock tests already run from test_stem_sync.
for _name in [name for name in vars(SharedClockWindowTests) if name.startswith("test_")]:
    setattr(StemPlaybackWindowTests, _name, None)
del SharedClockWindowTests


if __name__ == "__main__":
    unittest.main()
