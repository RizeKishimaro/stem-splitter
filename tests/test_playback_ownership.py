"""Duplicate voices and simultaneous multi-block starts, measured in the mix.

An individual stem that is then played as part of a selection must be one
voice, not two, and the mix must be the sum of the selected stems once.
Selected short blocks of one split must all be audible in the first audio
block, each from its own source offset.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_playback_ownership
"""

from __future__ import annotations

import unittest

import numpy as np

from soniclab.core.audio_engine import OUTPUT_RATE, AudioEngine
from soniclab.ui.main_window import stem_block, stem_group
from tests.test_short_blocks import BLOCK as TONE_BLOCK
from tests.test_short_blocks import ON, WINDOW, ShortBlockTests, tone_of
from tests.test_stem_sync import BLOCK, STEP, SharedClockWindowTests, lag_samples, ramp, run


class SpanOwnershipTests(unittest.TestCase):
    """The engine retires a voice that covers the same stretch of a clip."""

    def setUp(self):
        self.engine = AudioEngine()
        self.engine._capture_enabled = False
        self.engine.add_clip(ramp(1), 0.0)
        self.engine.add_clip(ramp(2), 0.0)
        self.engine._ensure_stream = lambda: None

    def test_a_second_voice_of_the_same_stretch_replaces_the_first(self):
        self.engine.stem_play("song", [("solo", 0, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        run(self.engine, 4)
        cursor = self.engine._groups["song"].cursor
        self.engine.stem_play("song", [("group", 0, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        self.assertEqual(set(self.engine._voices), {"group"})
        self.assertEqual(self.engine._groups["song"].cursor, cursor)
        mixed = run(self.engine, 6)
        self.assertLess(lag_samples(mixed, cursor, (1,)), 0.25)
        self.assertEqual(self.engine.diagnostics()["duplicates"], [])

    def test_a_neighbouring_stretch_of_the_same_clip_is_kept(self):
        self.engine.stem_play("song", [("a", 0, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        self.engine.stem_play("song", [("b", 0, False, 2.0, 2.0)], 0.0, loop=(0.0, 4.0))
        self.assertEqual(set(self.engine._voices), {"a", "b"})
        mixed = run(self.engine, 4)
        # Both stretches of stem 1, from the clock at 0: frame n and frame n+2s.
        error = lag_samples(mixed, 0, (1, 1), (0, 2 * OUTPUT_RATE))
        self.assertLess(error, 0.25)

    def test_another_song_is_not_retired(self):
        self.engine.stem_play("song", [("a", 0, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        self.engine.stem_play("other", [("b", 1, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        self.engine.stem_play("song", [("c", 0, False, 0.0, 2.0)], 0.0, loop=(0.0, 2.0))
        self.assertEqual(set(self.engine._voices), {"c", "b"})
        self.assertEqual(self.engine._voices["b"].group, "other")


class IndividualToGroupTests(SharedClockWindowTests):
    def _select_clips(self, *tracks):
        view = self.window._view_for(self.index(tracks[0]))
        view.set_picks({})
        view.scene_obj.clearSelection()
        for track in tracks:
            view._clip_for(self.index(track)).setSelected(True)
        self.app.processEvents()
        return view

    def _play_selection(self, view):
        block = view.selected_blocks()[0]
        self.window._toggle_block(*block)

    def test_group_play_adopts_the_stem_already_playing(self):
        drums, bass, other, vocals = self.split("Song")
        d, b, o, v = (self.index(track) for track in (drums, bass, other, vocals))
        self.press(d, 0)
        run(self.engine, 10)
        group = stem_group(drums)
        cursor = self.engine._groups[group].cursor
        self.assertGreater(cursor, 0)
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        playing = [voice for voice in self.engine._voices.values() if voice.playing]
        counts = {}
        for voice in playing:
            counts[voice.clip] = counts.get(voice.clip, 0) + 1
        self.assertEqual(counts, {d: 1, b: 1, o: 1})
        self.assertNotIn(v, counts)
        self.assertGreaterEqual(self.engine._groups[group].cursor, cursor)
        self.assertLess(self.engine._groups[group].cursor, cursor + BLOCK)
        self.assertEqual(self.engine.diagnostics()["duplicates"], [])
        at = self.engine._groups[group].cursor
        mixed = run(self.engine, 8)
        error = lag_samples(mixed, at, (1, 2, 3))
        doubled = lag_samples(mixed, at, (2, 2, 3))
        self.assertLess(error, 0.25)
        self.assertGreater(doubled, 100)  # a second drums voice would fit this, not the single one

    def test_an_unselected_stem_keeps_its_own_voice(self):
        drums, bass, other, vocals = self.split("Song")
        d, b, o, v = (self.index(track) for track in (drums, bass, other, vocals))
        self.press(v, 3)
        run(self.engine, 6)
        vocals_key = (v, stem_block(3))
        generation = self.engine._voices[vocals_key].generation
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        self.assertEqual(self.engine._voices[vocals_key].generation, generation)
        self.assertTrue(self.engine._voices[vocals_key].playing)
        playing = [voice.clip for voice in self.engine._voices.values() if voice.playing]
        self.assertEqual(playing.count(d), 1)
        self.assertEqual(playing.count(v), 1)
        self.assertEqual(len(playing), 4)
        group = self.engine._groups[stem_group(drums)]
        at = group.cursor
        mixed = run(self.engine, 6)
        clock = at + np.arange(len(mixed))
        loop = group.loop
        expected = np.zeros(len(mixed))
        for voice in self.engine._voices.values():
            if not voice.playing:
                continue
            length = voice.length if voice.length and voice.length < loop[1] - loop[0] else loop[1] - loop[0]
            phase = (clock - loop[0]) % length
            expected += (loop[0] + voice.shift + phase) * (voice.clip + 1) * STEP
        self.assertLess(float(np.max(np.abs(expected - mixed)) / STEP), 0.25)

    def test_rapid_individual_then_group_play_does_not_double(self):
        drums, bass, other, _vocals = self.split("Song")
        d = self.index(drums)
        self.press(d, 0)
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        self.assertEqual(sum(voice.clip == d for voice in self.engine._voices.values()), 1)
        mixed = run(self.engine, 4)
        self.assertLess(lag_samples(mixed, 0, (1, 2, 3)), 0.25)

    def test_group_pause_and_resume_keep_one_voice_each(self):
        drums, bass, other, _vocals = self.split("Song")
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        before = {voice.key: voice.generation for voice in self.engine._voices.values()}
        run(self.engine, 5)
        cursor = self.engine._groups[stem_group(drums)].cursor
        self._play_selection(view)  # pause: the clock holds the sample that was heard
        self.assertTrue(all(not voice.playing for voice in self.engine._voices.values()))
        held = self.engine._groups[stem_group(drums)].cursor
        self.assertLessEqual(held, cursor)
        self.assertGreaterEqual(held, cursor - BLOCK - 8)
        silent = run(self.engine, 3)
        self.assertEqual(self.engine._groups[stem_group(drums)].cursor, held)
        self.assertTrue(np.all(np.abs(silent) < 1e-8))
        self._play_selection(view)  # resume
        after = {voice.key: voice.generation for voice in self.engine._voices.values()}
        self.assertEqual(after, before)
        mixed = run(self.engine, 4)
        self.assertLess(lag_samples(mixed, held, (1, 2, 3)), 0.25)

    def test_stopping_one_stem_leaves_the_others_playing(self):
        drums, bass, other, _vocals = self.split("Song")
        d, b, o = (self.index(track) for track in (drums, bass, other))
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        run(self.engine, 4)
        self.window._deselect_all()
        self.window._stop_block_at(d, 0)
        playing = {voice.clip for voice in self.engine._voices.values() if voice.playing}
        self.assertEqual(playing, {b, o})
        at = self.engine._groups[stem_group(drums)].cursor
        mixed = run(self.engine, 4)
        self.assertLess(lag_samples(mixed, at, (2, 3)), 0.25)

    def test_play_stop_play_does_not_accumulate_voices(self):
        drums, bass, other, _vocals = self.split("Song")
        view = self._select_clips(drums, bass, other)
        for _ in range(4):
            self._play_selection(view)
            run(self.engine, 2)
            self.window._stop_blocks()
        self.assertEqual(self.engine._voices, {})
        self._play_selection(view)
        self.assertEqual(len(self.engine._voices), 3)
        mixed = run(self.engine, 4)
        self.assertLess(lag_samples(mixed, 0, (1, 2, 3)), 0.25)

    def test_two_songs_stay_independent_through_a_group_takeover(self):
        one = self.split("One")
        two = self.split("Two")
        drums, bass, other = one[:3]
        other_song = two[0]
        self.press(self.index(other_song), 0)
        run(self.engine, 8)
        other_group = stem_group(other_song)
        other_at = self.engine._groups[other_group].cursor
        view = self._select_clips(drums, bass, other)
        self._play_selection(view)
        self.assertEqual(self.engine._groups[other_group].cursor, other_at)
        song = stem_group(drums)
        self.assertNotEqual(song, other_group)
        playing = [voice for voice in self.engine._voices.values() if voice.playing]
        self.assertEqual(sum(voice.group == song for voice in playing), 3)
        self.assertEqual(sum(voice.group == other_group for voice in playing), 1)
        mixed = run(self.engine, 4)
        # Song One starts at 0 (scales 1+2+3). Song Two's drums (scale 1) is already at other_at.
        at = self.engine._groups[song].cursor - len(mixed)
        expected_frame = at + np.arange(len(mixed))
        other_frame = other_at + np.arange(len(mixed))
        expected = (expected_frame * (1 + 2 + 3) + other_frame * 1) * STEP
        error = float(np.max(np.abs(expected - mixed)) / STEP)
        self.assertLess(error, 0.25)
        self.assertEqual(self.engine._groups[other_group].cursor, other_at + len(mixed))


class SimultaneousBlockTests(ShortBlockTests):
    def _onsets(self, mixed, count):
        found = []
        for segment in range(count):
            level = self.levels(mixed, tone_of(segment))
            hits = np.nonzero(level > ON)[0]
            self.assertTrue(len(hits), f"tone {segment} never started: {np.round(level[:8], 3)}")
            found.append(int(hits[0]) * WINDOW)
        return found

    def _play_selected(self, pieces):
        """Pick these blocks (one split track's) and press the shared ▶."""
        index = pieces[0][0]
        view = self.window._view_for(index)
        view.set_picks({index: {bar for _track, bar in pieces}})
        self.window._toggle_block(*pieces[0])

    def test_four_125ms_blocks_start_in_one_callback(self):
        drums, *_rest = self.make("Song", [0.125] * 4)
        pieces = self.cut(drums, [0.125] * 4)
        self._play_selected(pieces)
        self.assertEqual(len({voice.group for voice in self.engine._voices.values()}), 1)
        self.assertEqual(len(self.engine._voices), 4)
        loop = next(iter(self.engine._groups.values())).loop
        self.assertEqual(loop[1] - loop[0], int(round(0.125 * OUTPUT_RATE)))
        mixed = self.mix(int(0.4 * OUTPUT_RATE / TONE_BLOCK))
        onsets = self._onsets(mixed, 4)
        self.assertEqual(max(onsets) - min(onsets), 0)
        self.assertEqual(onsets[0], 0)
        for segment in range(4):
            level = self.levels(mixed, tone_of(segment))
            self.assertGreater(float(np.median(level)), 0.15)
            self.assertLess(float(np.median(level)), 0.25)

    def test_individual_then_group_does_not_double_a_short_block(self):
        drums, *_rest = self.make("Song", [0.125] * 4)
        pieces = self.cut(drums, [0.125] * 4)
        view = self.window._view_for(pieces[0][0])
        view.set_picks({})
        self.toggle(pieces[0])
        self.mix(6)
        cursor = next(iter(self.engine._groups.values())).cursor
        self.assertGreater(cursor, 0)
        self._play_selected(pieces)
        self.assertEqual(len(self.engine._voices), 4)
        self.assertEqual(sum(key == self.voice_of(pieces[0]) for key in self.engine._voices), 1)
        self.assertGreaterEqual(next(iter(self.engine._groups.values())).cursor, cursor)
        mixed = self.mix(int(0.3 * OUTPUT_RATE / TONE_BLOCK))
        onsets = self._onsets(mixed, 4)
        self.assertEqual(max(onsets) - min(onsets), 0)
        for segment in range(4):
            level = self.levels(mixed, tone_of(segment))
            self.assertLess(float(np.median(level)), 0.25)

    def test_short_blocks_of_different_lengths_and_splits(self):
        lengths = [0.5, 0.25, 0.125, 0.125]
        drums, bass, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        bass_pieces = self.cut(bass, lengths)
        self._play_selected(pieces)
        mixed = self.mix(int(0.2 * OUTPUT_RATE / TONE_BLOCK))
        onsets = self._onsets(mixed, 4)
        self.assertEqual(max(onsets) - min(onsets), 0)
        # A block of the other stem is another track of the same split: it
        # joins the clock, it does not wait, and the first four stay one
        # voice each.
        self.toggle(bass_pieces[2])
        joined = self.mix(int(0.2 * OUTPUT_RATE / TONE_BLOCK))
        self.assertEqual(self.onset(joined, 2), 0)
        for segment in range(4):
            self.assertPlaying(joined, segment, "after another track joined")
        self.assertEqual(sum(key == self.voice_of(pieces[0]) for key in self.engine._voices), 1)
        self.stop(pieces[1])
        left = self.mix(4)
        self.assertSilent(left[WINDOW:], 1, "after its stop")
        for segment in (0, 2, 3):
            self.assertPlaying(left, segment, "while another piece was stopped")

    def test_eight_62ms_blocks_and_a_second_song(self):
        lengths = [0.0625] * 8
        drums, *_rest = self.make("Song", lengths)
        other = self.make("Other", [0.125])
        self.tones = [tone_of(segment) for segment in range(8)]
        pieces = self.cut(drums, lengths)
        self._play_selected(pieces)
        mixed = self.mix(int(0.25 * OUTPUT_RATE / TONE_BLOCK))
        onsets = self._onsets(mixed, 8)
        self.assertEqual(max(onsets) - min(onsets), 0)
        self.assertEqual(onsets[0], 0)
        self.toggle(self.index(other[0]))
        self.assertEqual(len({voice.group for voice in self.engine._voices.values()}), 2)
        both = self.mix(int(0.2 * OUTPUT_RATE / TONE_BLOCK))
        for segment in range(8):
            self.assertPlaying(both, segment, "while another song plays")


def _only_own_tests(cls, parent):
    for name in dir(parent):
        if name.startswith("test_"):
            setattr(cls, name, None)


_only_own_tests(IndividualToGroupTests, SharedClockWindowTests)
_only_own_tests(SimultaneousBlockTests, ShortBlockTests)
# The imported suites would be collected too. Drop the names; they are run
# on their own.
del SharedClockWindowTests, ShortBlockTests


if __name__ == "__main__":
    unittest.main()
