"""Selection playback follows the selection, measured in the mix.

Every stem is a sine at its own frequency (see test_master_transport), so
the mixed output shows which stems sound, and a stem's phase shows which
sample of its file it plays against its song's clock.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_selection_playback
"""

from __future__ import annotations

import unittest

from soniclab.ui.main_window import stem_block
from soniclab.ui.timeline import BarPlayButton
from tests.test_master_transport import FREQS, STEMS, TOLERANCE, SongClockTests, lags


class SelectionPlaybackTests(SongClockTests):
    def heard(self, blocks: int = 10) -> set:
        """Names of the stems that sound in the next ``blocks``, after the
        event loop has run (a selection edit applies there, once the edit
        that made it is done)."""
        self.app.processEvents()
        mixed, _clocks = self.render(blocks)
        found = lags(mixed, {name: 0 for name in STEMS}, dict(zip(STEMS, FREQS)))
        return {name for name, lag in found.items() if lag is not None}

    def start_selection(self, view, picks: dict) -> None:
        view.set_picks(picks)
        track, bars = next(iter(picks.items()))
        self.window._toggle_block(track, min(bars))  # the selection's shared ▶
        self.mix(4)

    def button(self, index: int, bar: int) -> BarPlayButton:
        clip = self.window._view_for(index)._clip_for(index)
        return next(c for c in clip.childItems() if isinstance(c, BarPlayButton) and c.index == bar)

    def check_follow(self, view, a: int, b: int, a_name: str, b_name: str, bar: int = 1) -> None:
        """A and B selected and playing; A deselected stops at once and
        stays silent past its block's length while B plays on; A selected
        again joins on the song's clock where it is now."""
        self.assertEqual(self.heard(), {a_name, b_name})
        view.toggle_pick(a, bar)  # deselect A, by code: no mouse involved
        self.assertEqual(self.heard(2), {b_name}, "A did not stop at once")
        self.assertEqual(self.heard(int(3.0 * 44100 / 512)), {b_name}, "A came back by itself")
        self.window._sync_previews()
        self.assertFalse(self.button(a, bar).playing)  # the canvas shows what plays
        view.toggle_pick(a, bar)  # select A again
        self.assertEqual(self.heard(), {a_name, b_name})
        tracks = self.window.project.tracks
        found = self.measure("A rejoined the song's clock", [tracks[a], tracks[b]], (dict(zip(STEMS, FREQS))[a_name], dict(zip(STEMS, FREQS))[b_name]))
        self.assertTrue(all(abs(lag) < TOLERANCE for lag in found.values()))

    def test_only_selected_tracks_play_and_follow_the_selection(self):
        tracks = self.make("Song")
        view = self.window._view_for(0)
        a, b = self.index(tracks[0]), self.index(tracks[1])
        self.start_selection(view, {a: {1}, b: {1}})
        self.check_follow(view, a, b, "drums", "bass")

    def test_esc_stops_the_selection_but_not_a_block_played_on_its_own(self):
        tracks = self.make("Song")
        view = self.window._view_for(0)
        other = self.index(tracks[2])
        self.window._toggle_block(other, 3)  # its own ▶: not selection playback
        self.start_selection(view, {self.index(tracks[0]): {1}, self.index(tracks[1]): {1}})
        self.assertEqual(self.heard(), {"drums", "bass", "other"})
        self.window._deselect_all()
        self.assertEqual(self.heard(int(2.5 * 44100 / 512)), {"other"})
        self.assertTrue(self.engine.previews()[(other, stem_block(3))][0])

    def test_editing_another_track_does_not_start_it(self):
        tracks = self.make("Song")
        view = self.window._view_for(0)
        self.start_selection(view, {self.index(tracks[0]): {1}, self.index(tracks[1]): {1}})
        self.window._select_track(self.index(tracks[3]))  # a click on its name, to edit it
        self.app.processEvents()
        self.assertEqual(self.heard(), {"drums", "bass"})

    def test_split_blocks_follow_the_selection(self):
        tracks = self.make("Song")
        drums = self.index(tracks[0])
        self.window._split_stem_in_half(drums, 0.0, 2.0)  # bar 1 becomes 0-1 s and 1-2 s
        self.app.processEvents()
        view = self.window._view_for(drums)
        bar = view.block_at_time(drums, 1.5)  # the second half
        times = view.block_times(drums, bar)
        self.assertEqual((round(float(times[0]), 6), round(float(times[1]), 6)), (1.0, 2.0))
        bass = self.index(tracks[1])
        self.start_selection(view, {drums: {bar}, bass: {1}})
        self.check_follow(view, drums, bass, "drums", "bass", bar)

    def test_cached_stems_follow_the_selection(self):
        fresh = self.make("Song")
        stems = {name: track["path"] for name, track in zip(STEMS, fresh)}
        self.window._finish_split_stems("Song", stems, STEMS, None)
        self.app.processEvents()
        cached = self.window.project.tracks[len(fresh):]
        view = self.window._view_for(0)
        a, b = self.index(cached[0]), self.index(fresh[1])
        self.start_selection(view, {a: {1}, b: {1}})
        self.check_follow(view, a, b, "drums", "bass")

    def test_after_a_canvas_transfer(self):
        tracks = self.make("Song")
        view = self.window._view_for(0)
        a, b = self.index(tracks[0]), self.index(tracks[1])
        self.start_selection(view, {a: {1}, b: {1}})
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(b, 1)  # bass to Canvas 2, still playing
        self.app.processEvents()
        b = self.index(tracks[1])
        self.assertEqual(self.heard(), {"drums", "bass"})
        view.toggle_pick(self.index(tracks[0]), 1)  # deselect drums
        self.assertEqual(self.heard(4), {"bass"})


# The inherited song-clock tests run from test_master_transport.
for _name in [name for name in vars(SongClockTests) if name.startswith("test_")]:
    setattr(SelectionPlaybackTests, _name, None)
del SongClockTests


if __name__ == "__main__":
    unittest.main()
