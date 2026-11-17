"""Short split-stem blocks are independent voices, measured in the output.

A stem's file holds a different tone in each short segment (1 kHz apart),
and the stem is cut into pieces of one segment each with the scissors, so
every piece is identified by its tone in the mixed output. A tone's level is
read by projecting short windows onto it (lock-in), which places the sample
where a voice starts or stops to within one window.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_short_blocks
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import soundfile as sf

from soniclab.core.audio_engine import OUTPUT_RATE
from tests.test_stem_sync import SharedClockWindowTests

BLOCK = 512
WINDOW = 128  # samples per level reading
LEVEL = 0.2  # each tone's amplitude in the file
ON = 0.5 * LEVEL
# A stopped tone still reads a little where another piece wraps: the jump
# in that piece's phase is a click, which spreads over every frequency.
OFF = 0.3 * LEVEL


def tone_of(segment: int) -> float:
    return 500.0 + 1000.0 * segment


class ShortBlockTests(SharedClockWindowTests):
    def make(self, name: str, lengths, loop_tone: float | None = None):
        """A split whose stems hold one tone per segment of ``lengths``
        seconds (segment k is ``tone_of(k)``), then silence. With
        ``loop_tone`` the vocals hold that one tone throughout instead."""
        frames = int(OUTPUT_RATE * 8)
        edges = np.round(np.cumsum([0.0, *lengths]) * OUTPUT_RATE).astype(int)
        index = np.searchsorted(edges, np.arange(frames), side="right") - 1
        index = np.where(np.arange(frames) >= edges[-1], len(lengths), index)
        freqs = np.array([tone_of(k) for k in range(len(lengths))] + [0.0])[index]
        phase = 2 * np.pi * np.cumsum(freqs) / OUTPUT_RATE
        signal = np.where(index < len(lengths), LEVEL * np.sin(phase), 0.0).astype(np.float32)
        steady = None
        if loop_tone is not None:
            steady = (LEVEL * np.sin(2 * np.pi * loop_tone * np.arange(frames) / OUTPUT_RATE)).astype(np.float32)
        self.tones = [tone_of(k) for k in range(len(lengths))] + ([loop_tone] if loop_tone else [])
        stems = {}
        for stem in ("drums", "bass", "other", "vocals"):
            path = os.path.join(self.root, f"{name}-{stem}.wav")
            audio = steady if stem == "vocals" and steady is not None else signal
            sf.write(path, np.column_stack((audio, audio)), OUTPUT_RATE, subtype="FLOAT")
            stems[stem] = path
        added = self.window._add_stem_tracks(name, stems)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        return added

    def cut(self, track, lengths) -> list[tuple[int, int]]:
        """Split ``track`` with Split in Half so its blocks end on the segment
        edges of ``lengths``; the blocks for those segments, as ``(track
        index, bar)``, in time order. A split is a mark on the same track, so
        every block is on ``track`` itself."""
        window = self.window
        index = self.index(track)
        view = window._view_for(index)
        start = float(track.get("start", 0.0) or 0.0)
        offset = float(track.get("offset", 0.0) or 0.0)
        edges = [offset + edge for edge in np.cumsum([0.0, *lengths])]

        def blocks():
            spans = view._spans_of(view._clip_for(index))
            return [(offset + float(span[3]), offset + float(span[4]), float(span[3])) for span in spans]

        # Each edge inside a block gets a split whose block-half is centred on
        # it, so the mark lands on the edge exactly.
        for edge in edges[1:]:
            for a, b, local_a in blocks():
                if a + 1e-6 < edge < b - 1e-6:
                    half = min(edge - a, b - edge)
                    window._split_stem_in_half(index, edge - offset - half, 2 * half)
                    self.app.processEvents()
                    break
        result = []
        for k, length in enumerate(lengths):
            bar = view.block_at_time(index, start + edges[k] - offset + 1e-4)
            self.assertIsNotNone(bar, f"no block at {edges[k]}")
            times = view.block_times(index, bar)
            self.assertAlmostEqual(times[1] - times[0], length, places=6)
            result.append((index, bar))
        self.assertEqual(sum(t is track for t in window.project.tracks), 1)  # still one track
        return result

    @staticmethod
    def _track_of(block) -> int:
        return block[0] if isinstance(block, tuple) else int(block)

    def bar(self, block) -> int:
        """A block's bar; a track index alone means its first block."""
        if isinstance(block, tuple):
            return block[1]
        track = self.window.project.tracks[block]
        return self.window._view_for(block).block_at_time(block, float(track["start"]) + 1e-3)

    def toggle(self, block) -> None:
        self.window._toggle_block(self._track_of(block), self.bar(block))

    def stop(self, block) -> None:
        index = self._track_of(block)
        key = self.window._block_voice(index, self.bar(block), self.engine.previews())
        self.window._stop_block(key)

    def voice_of(self, block):
        """The voice key that plays ``block`` now (or would)."""
        return self.window._block_voice(self._track_of(block), self.bar(block), self.engine.previews())

    def mix(self, blocks: int) -> np.ndarray:
        out = np.zeros((BLOCK, 2), np.float32)
        chunks = []
        for n in range(blocks):
            self.engine._callback(out, BLOCK, None, None)
            chunks.append(out[:, 0].copy())
            if n % 3 == 0:
                self.app.processEvents()  # timers and posted callbacks run
        return np.concatenate(chunks)

    def levels(self, mixed: np.ndarray, freq: float) -> np.ndarray:
        """Amplitude of ``freq`` in each ``WINDOW`` of ``mixed``. Every tone
        that can be in the mix is fitted together (sin and cos of each), so
        one tone's reading holds none of the others."""
        count = len(mixed) // WINDOW
        frames = mixed[: count * WINDOW].reshape(count, WINDOW).astype(np.float64)
        t = np.arange(WINDOW) / OUTPUT_RATE
        tones = sorted(set(self.tones) | {freq})
        basis = np.stack([f(2 * np.pi * tone * t) for tone in tones for f in (np.sin, np.cos)], axis=1)
        coef, *_ = np.linalg.lstsq(basis, frames.T, rcond=None)
        at = 2 * tones.index(freq)
        return np.hypot(coef[at], coef[at + 1])

    def assertPlaying(self, mixed, segment, msg=""):
        """Sounding all through: never low two windows running. The one
        window across the piece's own loop point can read low, even near 0
        (the tone's phase jumps mid-window and the halves cancel); a voice
        that waited would be silent for a whole block, dozens of windows."""
        level = self.levels(mixed, tone_of(segment))
        low = level < ON
        self.assertFalse(np.any(low[1:] & low[:-1]), f"tone {segment} dropped {msg}: {np.round(level, 2)}")

    def assertSilent(self, mixed, segment, msg=""):
        level = self.levels(mixed, tone_of(segment))
        self.assertTrue(np.all(level < OFF), f"tone {segment} sounded {msg}: max {level.max():.3f}")

    def onset(self, mixed, segment) -> int:
        """First sample of the first window where ``segment``'s tone sounds."""
        level = self.levels(mixed, tone_of(segment))
        hits = np.nonzero(level > ON)[0]
        self.assertTrue(len(hits), f"tone {segment} never started")
        return int(hits[0]) * WINDOW

    def check_concurrent(self, pieces, segment):
        """Every piece plays at once, and stopping then restarting one (at
        once, and with play/stop/play bursts) never pauses or delays the
        others; the restarted one sounds from the next audio block."""
        for piece in pieces:
            self.toggle(piece)
        mixed = self.mix(int(0.6 * OUTPUT_RATE / BLOCK))
        for k in range(len(pieces)):
            self.assertPlaying(mixed, k, "while all play")
        for k, piece in enumerate(pieces):
            others = [j for j in range(len(pieces)) if j != k]
            self.stop(piece)
            gap = self.mix(4)
            self.assertSilent(gap[WINDOW:], k, "after its stop")
            for j in others:
                self.assertPlaying(gap, j, f"while {k} was stopped")
            self.toggle(piece)  # play again
            after = self.mix(int(0.4 * OUTPUT_RATE / BLOCK))
            self.assertEqual(self.onset(after, k), 0, f"restarted piece {k} waited")
            for j in others:
                self.assertPlaying(after, j, f"after {k} restarted")
        # Rapid Play/Stop/Play bursts on one piece, inside one audio block.
        for _ in range(5):
            self.stop(pieces[0])
            self.toggle(pieces[0])
            self.stop(pieces[0])
            self.toggle(pieces[0])
            burst = self.mix(int(0.1 * OUTPUT_RATE / BLOCK))
            for k in range(len(pieces)):
                self.assertPlaying(burst, k, "through a play/stop burst")
        keys = {self.voice_of(piece) for piece in pieces}
        playing = [key for key, state in self.engine.previews().items() if state[0] and key in keys]
        self.assertEqual(len(playing), len(pieces))  # one voice each, none duplicated
        ids = {self.engine.voice_state(key)[1] for key in playing}
        self.assertEqual(len(ids), len(pieces))

    def fresh(self):
        self.engine.preview_stop_all()
        self.window._preview_tracks.clear()
        self.window._sync_previews()

    def test_two_very_short_blocks_play_together(self):
        drums, *_rest = self.make("Song", [0.125] * 2)
        self.check_concurrent(self.cut(drums, [0.125] * 2), 0.125)

    def test_three_four_and_eight_short_blocks(self):
        for lengths in ([0.125, 0.125, 0.25], [0.125] * 4, [0.0625] * 8):
            count = len(lengths)
            with self.subTest(count=count):
                self.fresh()
                drums, *_rest = self.make(f"Song{count}", lengths)
                self.check_concurrent(self.cut(drums, lengths), 0.125)

    def test_blocks_of_different_lengths(self):
        lengths = [0.5, 0.25, 0.125, 0.125]
        drums, *_rest = self.make("Song", lengths)
        self.check_concurrent(self.cut(drums, lengths), 0.125)

    def test_selected_pieces_on_one_row_loop_together(self):
        # The pieces of a cut stem sit on one row. Selected and played with
        # the shared ▶, they still sound at once, not one after another.
        lengths = [0.125] * 4
        drums, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        index = pieces[0][0]
        view = self.window._view_for(index)
        view.set_picks({index: {bar for _track, bar in pieces}})
        self.window._toggle_block(*pieces[0])  # the shared ▶
        mixed = self.mix(int(0.6 * OUTPUT_RATE / BLOCK))
        for k in range(len(pieces)):
            self.assertPlaying(mixed, k, "in the selection loop")
            self.assertEqual(self.onset(mixed, k), 0)
        # Unselected, each piece stops and restarts alone.
        self.window._deselect_all()
        self.fresh()
        self.check_concurrent(pieces, 0.125)

    def test_cached_and_fresh_short_blocks(self):
        lengths = [0.125, 0.125]
        fresh = self.make("Song", lengths)
        names = ("drums", "bass", "other", "vocals")
        self.window._finish_split_stems("Song", {n: t["path"] for n, t in zip(names, fresh)}, names, None)
        self.app.processEvents()
        cached = self.window.project.tracks[len(fresh):]
        first = self.cut(fresh[0], lengths)[0]  # tone 0 from the fresh split
        second = self.cut(cached[0], lengths)[1]  # tone 1 from the cached one
        self.check_concurrent([first, second], 0.125)

    def test_blocks_on_other_tracks_and_canvases(self):
        lengths = [0.125] * 4
        drums, bass, *_rest = self.make("Song", lengths)
        self.cut(drums, lengths)
        self.window._toggle_second_canvas(True)
        self.window._move_track_to_canvas(self.index(drums), 1, start=4.0)  # the split track, marks and all
        self.app.processEvents()
        drum_blocks = self.cut(drums, lengths)  # the same blocks, found in their new place
        bass_blocks = self.cut(bass, lengths)
        self.assertEqual(drums.get("canvas"), 1)
        self.assertNotEqual(bass.get("canvas"), 1)
        # Tones 0 and 2 from drum blocks on Canvas 2, tone 1 from bass here.
        self.check_concurrent([drum_blocks[0], bass_blocks[1], drum_blocks[2]], 0.125)

    def test_short_blocks_beside_a_looping_selection(self):
        lengths = [0.125] * 2
        drums, _bass, _other, vocals = self.make("Song", lengths, loop_tone=250.0)
        pieces = self.cut(drums, lengths)
        vocals_index = self.index(vocals)
        view = self.window._view_for(vocals_index)
        view.set_picks({vocals_index: {0, 1}})  # the selection loops on its own
        self.window._toggle_block(vocals_index, 0)
        # It stays selected: deselecting would end selection playback. The
        # unselected pieces play, stop and restart with their own ▶ beside it.
        self.check_concurrent(pieces, 0.125)
        loop = self.mix(int(0.3 * OUTPUT_RATE / BLOCK))
        self.assertTrue(np.all(self.levels(loop, 250.0) > ON))  # the loop never stopped
        self.assertTrue(all(state[0] for key, state in self.engine.previews().items() if key[0] == vocals_index))


# The inherited shared-clock tests already run from test_stem_sync.
for _name in [name for name in vars(SharedClockWindowTests) if name.startswith("test_")]:
    setattr(ShortBlockTests, _name, None)
del SharedClockWindowTests


if __name__ == "__main__":
    unittest.main()
