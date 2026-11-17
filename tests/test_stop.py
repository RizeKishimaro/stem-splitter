"""Stop takes exactly one voice out of the mix, measured in the output.

Pieces of a stem each hold their own tone (see test_short_blocks), so the
mixed output shows which blocks sound. STOPPED means the voice is no longer
mixed: its tone is gone from the next audio block and never comes back by
itself, while every other voice plays on.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_stop
"""

from __future__ import annotations

import unittest

from soniclab.core.audio_engine import OUTPUT_RATE, STOPPED
from tests.test_short_blocks import BLOCK, ShortBlockTests


class StopTests(ShortBlockTests):
    def check_stop(self, pieces, wait: float = 1.0):
        """Play every piece; stop each in turn through the block menu's Stop:
        silent from the next block, for longer than its own length, while
        the rest play; then ▶ brings back only that one."""
        for piece in pieces:
            if not self.engine.previews().get(self.window._block_voice(piece, self.bar(piece), self.engine.previews()), (False,))[0]:
                self.toggle(piece)
        self.mix(int(0.3 * OUTPUT_RATE / BLOCK))
        for k, piece in enumerate(pieces):
            others = [j for j in range(len(pieces)) if j != k]
            self.window._stop_block_at(piece, self.bar(piece))
            key = self.window._block_voice(piece, self.bar(piece), self.engine.previews())
            self.assertEqual(self.engine.voice_state(key)[0], STOPPED)
            report = self.engine.voice_report()
            self.assertFalse(any(row["key"] == key and row["mixed"] for row in report))
            after = self.mix(int(wait * OUTPUT_RATE / BLOCK))  # beyond the block's own length
            self.assertSilent(after, k, "after Stop")  # from the first block on
            for j in others:
                self.assertPlaying(after, j, f"while {k} was stopped")
            self.toggle(piece)  # ▶: only this one comes back
            back = self.mix(int(0.3 * OUTPUT_RATE / BLOCK))
            self.assertEqual(self.onset(back, k), 0)
            for j in others:
                self.assertPlaying(back, j, f"after {k} played again")
        voices = [row for row in self.engine.voice_report() if row["key"][0] in pieces]
        self.assertEqual(len(voices), len(pieces))  # no stale or duplicate voice left behind
        self.assertTrue(all(row["mixed"] for row in voices))

    def test_stop_a_stem_while_it_plays(self):
        lengths = [0.25] * 2
        drums, bass, *_rest = self.make("Song", lengths)
        drum = self.cut(drums, lengths)[0]
        bass_piece = self.cut(bass, lengths)[1]
        self.check_stop([drum, bass_piece])

    def test_stop_very_short_blocks(self):
        lengths = [0.0625] * 4
        drums, *_rest = self.make("Song", lengths)
        self.check_stop(self.cut(drums, lengths), wait=0.5)

    def test_stop_cached_stems(self):
        lengths = [0.125] * 2
        fresh = self.make("Song", lengths)
        names = ("drums", "bass", "other", "vocals")
        self.window._finish_split_stems("Song", {n: t["path"] for n, t in zip(names, fresh)}, names, None)
        self.app.processEvents()
        cached = self.window.project.tracks[len(fresh):]
        self.check_stop([self.cut(fresh[0], lengths)[0], self.cut(cached[0], lengths)[1]])

    def test_stop_a_block_of_a_looping_selection(self):
        lengths = [0.125, 0.125, 0.25]
        drums, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        view = self.window._view_for(pieces[0])
        view.set_picks({piece: {self.bar(piece)} for piece in pieces})
        self.window._toggle_block(pieces[0], self.bar(pieces[0]))  # the selection loops
        self.mix(int(0.2 * OUTPUT_RATE / BLOCK))
        self.check_stop(pieces)

    def test_rapid_play_stop_play(self):
        lengths = [0.125] * 2
        drums, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        for piece in pieces:
            self.toggle(piece)
        self.mix(4)
        for _ in range(10):
            self.window._stop_block_at(pieces[0], self.bar(pieces[0]))
            self.toggle(pieces[0])
            self.window._stop_block_at(pieces[0], self.bar(pieces[0]))
        mixed = self.mix(int(0.5 * OUTPUT_RATE / BLOCK))
        self.assertSilent(mixed, 0, "after play/stop bursts ending in Stop")
        self.assertPlaying(mixed, 1, "through the bursts")
        self.assertEqual(len([row for row in self.engine.voice_report() if row["key"][0] == pieces[0]]), 0)
        # A Stop meant for an older voice does not stop a newer one.
        self.toggle(pieces[0])
        key = self.window._block_voice(pieces[0], self.bar(pieces[0]), self.engine.previews())
        old = self.engine.voice_state(key)[1]
        self.window._stop_block(key)
        self.toggle(pieces[0])
        self.window._stop_block(key, old)  # late: the voice it meant is gone
        mixed = self.mix(int(0.3 * OUTPUT_RATE / BLOCK))
        self.assertPlaying(mixed, 0, "after a stale Stop")
        self.assertPlaying(mixed, 1)

    def test_edits_leave_no_orphaned_voice(self):
        lengths = [0.25] * 2
        drums, _bass, _other, vocals = self.make("Song", lengths, loop_tone=250.0)
        pieces = self.cut(drums, lengths)
        vocals_index = self.index(vocals)
        self.window._toggle_block(vocals_index, 0)  # unrelated: plays on through the edits
        self.toggle(pieces[0])
        self.mix(10)
        # Moved to Canvas 2: it plays on, shown on its block there, and that
        # block's Stop silences it.
        moved = self.window.project.tracks[pieces[0]]
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(pieces[0], 1, start=3.0)
        self.app.processEvents()
        index = self.index(moved)
        bar = self.window._view_for(index).block_at_time(index, 3.0 + 1e-3)
        mixed = self.mix(10)
        self.assertPlaying(mixed, 0, "after the move")
        self.window._sync_previews()
        self.assertTrue(self.window._view_for(index)._clip_for(index)._preview)  # shown playing
        self.window._stop_block_at(index, bar)
        self.assertSilent(self.mix(int(0.6 * OUTPUT_RATE / BLOCK)), 0, "after Stop on the moved block")
        # Cut while it plays: the block it played is gone, and so is its voice.
        self.toggle(pieces[1])
        self.mix(4)
        piece = self.window.project.tracks[pieces[1]]
        self.window._split_stem_in_half(self.index(piece), 0.0, 0.25)
        self.app.processEvents()
        report = self.engine.voice_report()
        vocals_index = self.index(vocals)
        self.assertFalse(any(row["key"][0] != vocals_index for row in report), report)
        mixed = self.mix(int(0.6 * OUTPUT_RATE / BLOCK))
        self.assertSilent(mixed, 1, "after its block was cut")
        self.assertTrue(all(level > 0.1 for level in self.levels(mixed, 250.0)))  # unrelated: still playing

    def test_voice_report_lists_ids_blocks_states_and_mix(self):
        lengths = [0.25] * 2
        drums, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        self.toggle(pieces[0])
        self.toggle(pieces[1])
        self.toggle(pieces[1])  # paused
        text = self.window._voice_report_text()
        lines = text.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("playing" in line and "in the mix" in line and "bar 1" in line for line in lines), text)
        self.assertTrue(any("paused" in line and "not mixed" in line for line in lines), text)
        self.assertTrue(all("#" in line and "file" in line for line in lines), text)
        self.engine.preview_stop_all()
        self.assertIn("No voices", self.window._voice_report_text())


# The short-block tests run from test_short_blocks.
for _name in [name for name in vars(ShortBlockTests) if name.startswith("test_")]:
    setattr(StopTests, _name, None)
del ShortBlockTests


if __name__ == "__main__":
    unittest.main()
