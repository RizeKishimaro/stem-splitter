"""Mixed-selection playback shares one clock, and a split block moves canvases.

A selection that contains pieces of more than one song used to start one
clock per song, so the pieces did not stay together. Dragging a block that
is already its own clip must take that same clip to the other canvas.

Tones are distinct frequencies in the mixed audio. Canvas checks use the
track object, its id, and the audio array's id.

Run: QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_selection_canvas
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import soundfile as sf
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from soniclab.core.audio_engine import OUTPUT_RATE
from soniclab.ui.main_window import stem_group
from soniclab.ui.timeline import PIXELS_PER_SECOND, ClipItem, track_canvas
from tests.test_short_blocks import BLOCK, LEVEL, OFF, ON, WINDOW, ShortBlockTests, tone_of


NORMAL = 4500.0
OTHER = 5500.0


def _tone(freq: float, seconds: float) -> np.ndarray:
    count = max(1, int(round(seconds * OUTPUT_RATE)))
    phase = 2 * np.pi * freq * np.arange(count) / OUTPUT_RATE
    signal = (LEVEL * np.sin(phase)).astype(np.float32)
    return np.column_stack((signal, signal))


class SelectionCanvasTests(ShortBlockTests):
    def setUp(self):
        super().setUp()
        self.window.second_canvas_action.setChecked(True)
        self.app.processEvents()

    def drag(self, source, scene_from, target_view, scene_to):
        start = source.mapFromScene(scene_from)
        end_global = target_view.viewport().mapToGlobal(target_view.mapFromScene(scene_to))
        end = source.viewport().mapFromGlobal(end_global)
        middle = QPoint(int((start.x() + end.x()) / 2), int((start.y() + end.y()) / 2))
        for kind, at, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, middle, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton),
        ):
            QApplication.sendEvent(source.viewport(), QMouseEvent(
                kind, QPointF(at), QPointF(source.viewport().mapToGlobal(at)),
                Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier,
            ))
            self.app.processEvents()
        for _ in range(4):
            self.app.processEvents()

    def clips(self, view):
        return {
            item.clip_index: item
            for item in view.scene_obj.items()
            if isinstance(item, ClipItem) and isValid(item)
        }

    def add_tone(self, name: str, freq: float, seconds: float, start: float = 0.0) -> int:
        audio = _tone(freq, seconds)
        path = os.path.join(self.root, f"{name}.wav")
        sf.write(path, audio, OUTPUT_RATE, subtype="FLOAT")
        self.window.project.tracks.append({
            "name": name,
            "path": path,
            "start": float(start),
            "duration": float(seconds),
            "offset": 0.0,
            "_audio": audio,
            "armed": False,
        })
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        return len(self.window.project.tracks) - 1

    def play(self, *indices: int) -> None:
        for view in self.window._timelines():
            view.set_picks({})
        for index in indices:
            view = self.window._view_for(index)
            picks = view.picks()
            picks.setdefault(index, set()).add(self.bar(index))
            view.set_picks(picks)
        self.window._toggle_block(indices[0], self.bar(indices[0]))

    def onset_freq(self, mixed, freq: float) -> int:
        level = self.levels(mixed, freq)
        hits = np.nonzero(level > ON)[0]
        self.assertTrue(len(hits), f"{freq:g} Hz never started: {np.round(level[:8], 3)}")
        return int(hits[0]) * WINDOW

    def assert_level(self, mixed, freq: float, low: float, high: float, msg: str = "") -> None:
        level = self.levels(mixed, freq)
        median = float(np.median(level))
        self.assertGreater(median, low, f"{freq:g} Hz too quiet {msg}: {median:.3f}")
        self.assertLess(median, high, f"{freq:g} Hz too loud {msg}: {median:.3f}")

    def voices_of(self, index: int) -> list:
        return [voice for voice in self.engine._voices.values() if voice.clip == index]

    def test_split_blocks_and_a_normal_track_share_one_clock(self):
        lengths = [0.5, 0.25, 0.25]
        drums, *_rest = self.make("Song", lengths)
        pieces = self.cut(drums, lengths)
        normal = self.add_tone("room", NORMAL, 0.5)
        self.tones = [tone_of(0), tone_of(1), tone_of(2), NORMAL]
        # The middle piece is the gap: it must not sound.
        self.play(pieces[0], pieces[2], normal)
        selected = {pieces[0], pieces[2], normal}
        playing = [voice for voice in self.engine._voices.values() if voice.playing]
        self.assertEqual({voice.clip for voice in playing}, selected)
        self.assertEqual({voice.group for voice in playing}, {playing[0].group})
        self.assertTrue(playing[0].group.startswith("selection:"))
        self.assertEqual(len(self.engine._groups), 1)
        self.assertEqual(self.engine.diagnostics()["duplicates"], [])
        for index in selected:
            self.assertEqual(len(self.voices_of(index)), 1)
        # Offsets and lengths: the later piece reads its own tone, not the file start.
        by_clip = {voice.clip: voice for voice in playing}
        self.assertEqual(by_clip[pieces[2]].shift, int(round(0.75 * OUTPUT_RATE)))
        self.assertEqual(by_clip[pieces[0]].length, int(round(0.5 * OUTPUT_RATE)))
        self.assertEqual(by_clip[pieces[2]].length, int(round(0.25 * OUTPUT_RATE)))
        self.assertEqual(by_clip[normal].length, int(round(0.5 * OUTPUT_RATE)))
        mixed = self.mix(int(0.15 * OUTPUT_RATE / BLOCK))
        onsets = [self.onset_freq(mixed, freq) for freq in (tone_of(0), tone_of(2), NORMAL)]
        self.assertEqual(max(onsets) - min(onsets), 0)
        self.assertEqual(onsets[0], 0)
        self.assert_level(mixed, tone_of(0), 0.15, 0.25)
        self.assert_level(mixed, tone_of(2), 0.15, 0.25)
        self.assert_level(mixed, NORMAL, 0.15, 0.25)
        gap = self.levels(mixed, tone_of(1))
        self.assertTrue(np.all(gap < OFF), f"the gap sounded: {gap.max():.3f}")

    def test_two_splits_share_the_selection_clock(self):
        drums, *_rest = self.make("Song", [0.5, 0.5])
        other = self.make("Other", [0.5, 0.5])
        song = self.cut(drums, [0.5, 0.5])[0]
        foreign = self.index(other[0])
        # A different split, a different tone, so the mix can tell them apart.
        audio = _tone(OTHER, 0.5)
        self.window.project.tracks[foreign]["_audio"] = audio
        self.window.project.tracks[foreign]["duration"] = 0.5
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        self.assertNotEqual(stem_group(self.window.project.tracks[song]), stem_group(self.window.project.tracks[foreign]))
        self.tones = [tone_of(0), OTHER]
        self.play(song, foreign)
        playing = [voice for voice in self.engine._voices.values() if voice.playing]
        self.assertEqual(len(playing), 2)
        self.assertEqual(len({voice.group for voice in playing}), 1)
        self.assertTrue(playing[0].group.startswith("selection:"))
        mixed = self.mix(int(0.15 * OUTPUT_RATE / BLOCK))
        self.assertEqual(self.onset_freq(mixed, tone_of(0)), 0)
        self.assertEqual(self.onset_freq(mixed, OTHER), 0)
        self.assert_level(mixed, tone_of(0), 0.15, 0.25)
        self.assert_level(mixed, OTHER, 0.15, 0.25)

    def test_individual_playback_joins_the_selection_without_a_second_voice(self):
        drums, bass, *_rest = self.make("Song", [0.25, 0.25])
        pieces = self.cut(drums, [0.25, 0.25])
        bass_pieces = self.cut(bass, [0.25, 0.25])
        aside = bass_pieces[0]
        # The unselected preview has its own tone, on the song's own clock.
        self.window.project.tracks[aside]["_audio"] = _tone(OTHER, 0.25)
        normal = self.add_tone("room", NORMAL, 0.5)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        self.tones = [tone_of(0), tone_of(1), NORMAL, OTHER]
        self.toggle(aside)
        self.mix(6)
        song = stem_group(self.window.project.tracks[aside])
        aside_at = self.engine._groups[song].cursor
        self.assertGreater(aside_at, 0)
        aside_generation = self.engine.voice_state(self.window._block_voice(
            aside, self.bar(aside), self.engine.previews()
        ))[1]
        self.play(pieces[0], normal)
        self.assertEqual(len(self.voices_of(pieces[0])), 1)
        self.assertEqual(len(self.voices_of(normal)), 1)
        self.assertEqual(len(self.voices_of(aside)), 1)
        self.assertEqual(self.voices_of(aside)[0].group, song)
        self.assertNotEqual(self.voices_of(pieces[0])[0].group, song)
        self.assertGreaterEqual(self.engine._groups[song].cursor, aside_at)
        self.assertEqual(self.engine.voice_state(self.window._block_voice(
            aside, self.bar(aside), self.engine.previews()
        ))[1], aside_generation)
        mixed = self.mix(int(0.15 * OUTPUT_RATE / BLOCK))
        self.assertEqual(self.onset_freq(mixed, tone_of(0)), 0)
        self.assertEqual(self.onset_freq(mixed, NORMAL), 0)
        self.assert_level(mixed, tone_of(0), 0.15, 0.25, "doubled after joining the selection")
        self.assert_level(mixed, OTHER, 0.15, 0.25, "the unselected preview")
        self.assertGreater(self.engine._groups[song].cursor, aside_at)

    def test_pause_resume_and_stop_do_not_duplicate_or_restart_the_other_audio(self):
        drums, bass, *_rest = self.make("Song", [0.5, 0.5])
        piece = self.cut(drums, [0.5, 0.5])[0]
        aside = self.cut(bass, [0.5, 0.5])[0]
        self.window.project.tracks[aside]["_audio"] = _tone(OTHER, 0.5)
        normal = self.add_tone("room", NORMAL, 0.5)
        self.window._reload_engine()
        self.window._refresh_timeline()
        self.app.processEvents()
        self.tones = [tone_of(0), NORMAL, OTHER]
        self.toggle(aside)
        self.play(piece, normal)
        before = {voice.key: voice.generation for voice in self.engine._voices.values()}
        self.assertEqual(len(before), 3)
        self.play(piece, normal)  # every selected voice is playing: pause them
        paused = {voice.key: voice.generation for voice in self.engine._voices.values()}
        self.assertEqual(paused, before)
        self.assertTrue(self.voices_of(aside)[0].playing)
        self.assertFalse(self.voices_of(piece)[0].playing)
        self.assertFalse(self.voices_of(normal)[0].playing)
        self.play(piece, normal)  # resume
        resumed = {voice.key: voice.generation for voice in self.engine._voices.values()}
        self.assertEqual(resumed, before)
        self.assertTrue(all(voice.playing for voice in self.engine._voices.values()))
        song = stem_group(self.window.project.tracks[aside])
        held = self.engine._groups[song].cursor
        self.stop(normal)
        self.assertEqual(self.voices_of(normal), [])
        self.assertEqual(len(self.voices_of(piece)), 1)
        self.assertTrue(self.voices_of(piece)[0].playing)
        self.assertTrue(self.voices_of(aside)[0].playing)
        self.assertEqual(self.engine._groups[song].cursor, held)
        mixed = self.mix(4)
        self.assertGreater(self.engine._groups[song].cursor, held)
        self.assert_level(mixed, OTHER, 0.15, 0.25, "after another voice stopped")
        self.assert_level(mixed, tone_of(0), 0.15, 0.25, "after another voice stopped")

    def _half(self):
        drums, bass, *_rest = self.make("Song", [0.5, 0.5])
        pieces = self.cut(drums, [0.5, 0.5])
        first = self.window.project.tracks[pieces[0]]
        second = self.window.project.tracks[pieces[1]]
        self.assertTrue(first.get("bar_block"))
        return pieces, first, second, bass

    def _fields(self, track):
        return (
            id(track["_audio"]),
            round(float(track["offset"]), 6),
            round(float(track["duration"]), 6),
            track.get("stem_group"),
            bool(track.get("bar_block")),
        )

    def test_dragging_a_split_block_to_the_other_canvas_moves_it(self):
        pieces, first, second, bass = self._half()
        window = self.window
        ident = id(first)
        fields = self._fields(first)
        sibling = id(second)
        bass_id = id(bass)
        others = [id(track) for track in window.project.tracks if track is not first]
        view = window.timeline
        clip = view._clip_for(pieces[0])
        grab = QPointF(clip.pos().x() + 20, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, window.timeline2, QPointF(grab.x() + PIXELS_PER_SECOND, 40))
        tracks = window.project.tracks
        self.assertEqual(len([track for track in tracks if id(track) == ident]), 1)
        moved = next(track for track in tracks if id(track) == ident)
        self.assertIs(moved, first)
        self.assertEqual(track_canvas(moved), 1)
        self.assertEqual(self._fields(moved), fields)
        self.assertNotIn(tracks.index(moved), self.clips(view))
        self.assertIn(tracks.index(moved), self.clips(window.timeline2))
        self.assertEqual(track_canvas(second), 0)
        self.assertEqual(id(second), sibling)
        self.assertIn(id(bass), {id(track) for track in tracks})
        self.assertEqual(bass_id, id(bass))
        self.assertEqual(sorted(id(track) for track in tracks if track is not moved), sorted(others))

    def test_a_picked_split_block_moves_and_repeating_the_move_does_not_copy(self):
        pieces, first, second, _bass = self._half()
        window = self.window
        view = window.timeline
        view.set_picks({pieces[0]: {self.bar(pieces[0])}})
        ident = id(first)
        audio = id(first["_audio"])
        count = len(window.project.tracks)
        clip = view._clip_for(pieces[0])
        grab = QPointF(clip.pos().x() + 16, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, window.timeline2, QPointF(grab.x() + 30, 36))
        self.assertEqual(sum(track is first for track in window.project.tracks), 1)
        self.assertEqual(id(first), ident)
        self.assertEqual(track_canvas(first), 1)
        self.assertEqual(track_canvas(second), 0)
        # Back and forth through the model. The same object, never a second one.
        for target_canvas, source in ((0, window.timeline2), (1, window.timeline), (0, window.timeline2)):
            index = window.project.tracks.index(first)
            window._drop_on_canvas(source, [index], index, 0.0, 0)
            self.assertEqual(sum(track is first for track in window.project.tracks), 1)
            self.assertEqual(track_canvas(first), target_canvas)
            self.assertEqual(len(window.project.tracks), count)
        self.assertEqual(id(first["_audio"]), audio)
        self.assertIs(second["_audio"], first["_audio"])

    def test_a_playing_block_moved_back_and_forth_plays_on_as_the_same_voice(self):
        pieces, first, second, _bass = self._half()
        window = self.window
        self.tones = [tone_of(0), tone_of(1)]
        self.toggle(pieces[0])  # tone 0: the piece that moves
        self.toggle(pieces[1])  # tone 1: stays where it is
        self.mix(4)

        def voice_of(track):
            index = window.project.tracks.index(track)
            found = [(key, voice) for key, voice in self.engine._voices.items() if key[0] == index]
            self.assertEqual(len(found), 1, f"{track['name']}: {found}")
            return found[0]

        moved_generation = voice_of(first)[1].generation
        still_generation = voice_of(second)[1].generation
        view = window.timeline
        clip = view._clip_for(pieces[0])
        grab = QPointF(clip.pos().x() + 16, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, window.timeline2, QPointF(grab.x() + 30, 36))  # a real drag
        for target, source in ((0, window.timeline2), (1, window.timeline), (0, window.timeline2)):
            index = window.project.tracks.index(first)
            window._drop_on_canvas(source, [index], index, 0.0, 0)
            self.app.processEvents()
            self.assertEqual(track_canvas(first), target)
        self.assertEqual(sum(track is first for track in window.project.tracks), 1)  # moved, never copied
        # Each move ends in the mix: both pieces sound, no gap, no second copy.
        mixed = self.mix(int(0.6 * OUTPUT_RATE / BLOCK))
        self.assertPlaying(mixed, 0, "after the moves")
        self.assertPlaying(mixed, 1, "beside the moves")
        self.assert_level(mixed, tone_of(0), 0.15, 0.25, "doubled by a move")
        # The same voices throughout: not replaced, not restarted.
        self.assertEqual(voice_of(first)[1].generation, moved_generation)
        self.assertEqual(voice_of(second)[1].generation, still_generation)
        self.assertEqual(len(self.engine._voices), 2)

    def coverage(self, path) -> dict:
        """Which stretches of one stem file each canvas holds (seconds)."""
        held = {0: [], 1: []}
        for track in self.window.project.tracks:
            if track.get("path") == path:
                offset = float(track.get("offset", 0.0) or 0.0)
                held[track_canvas(track)].append((round(offset, 6), round(offset + float(track["duration"]), 6)))
        return {canvas: sorted(spans) for canvas, spans in held.items()}

    def assertLabelsInside(self, msg=""):
        """Every clip's name stays within its own clip, so it can never read
        as the label of the clip beside it."""
        for view in self.window._timelines():
            for clip in self.clips(view).values():
                title = clip._title
                inside = title.sceneBoundingRect().right() <= clip.sceneBoundingRect().right() + 0.5
                self.assertTrue(inside, f"{clip.name} label runs past its clip {msg}")

    def test_labels_stay_inside_their_clips_through_cuts_moves_and_trims(self):
        drums, *_rest = self.make("Song", [0.5, 0.5, 1.0])
        pieces = self.cut(drums, [0.5, 0.5, 1.0])
        self.assertLabelsInside("after cutting")
        piece = self.window.project.tracks[pieces[0]]
        self.assertEqual(self.window.timeline._clip_for(pieces[0])._title_text, piece["name"])  # the whole name is kept
        self.window.timeline.set_picks({pieces[0]: {self.bar(pieces[0])}, pieces[1]: {self.bar(pieces[1])}})
        self.app.processEvents()
        self.assertLabelsInside("with two pieces selected")
        for source in (self.window.timeline, self.window.timeline2, self.window.timeline):
            index = self.window.project.tracks.index(piece)
            self.window._drop_on_canvas(source, [index], index, 0.0, 0)
            self.app.processEvents()
            self.assertLabelsInside(f"after a move from canvas {source.canvas + 1}")
        clip = self.window._view_for(self.window.project.tracks.index(piece))._clip_for(self.window.project.tracks.index(piece))
        clip.setRect(0, 0, 200.0, clip.rect().height())  # wide again: the name comes back whole
        clip._fit_title()
        self.assertEqual(clip._title.text(), piece["name"])

    def test_dragging_picked_pieces_or_one_bar_moves_and_never_copies(self):
        drums, *_rest = self.make("Song", [0.5, 0.5, 1.0])
        pieces = self.cut(drums, [0.5, 0.5, 1.0])
        window = self.window
        tracks = window.project.tracks
        moving = [tracks[pieces[0]], tracks[pieces[1]]]
        audio = id(moving[0]["_audio"])
        count = len(tracks)
        view = window.timeline
        view.set_picks({pieces[0]: {self.bar(pieces[0])}, pieces[1]: {self.bar(pieces[1])}})
        clip = view._clip_for(pieces[0])
        grab = QPointF(clip.pos().x() + clip.rect().width() * 0.3, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, window.timeline2, QPointF(300, 40))
        held = self.coverage(drums["path"])
        self.assertEqual(held[1], [(0.0, 0.5), (0.5, 1.0)])  # both pieces arrived once
        self.assertNotIn((0.0, 0.5), held[0])  # and left canvas 1
        self.assertNotIn((0.5, 1.0), held[0])
        self.assertEqual(len(window.project.tracks), count)
        self.assertTrue(all(sum(t is piece for t in window.project.tracks) == 1 for piece in moving))
        self.assertTrue(all(track_canvas(piece) == 1 and id(piece["_audio"]) == audio for piece in moving))
        clips_two = self.clips(window.timeline2)
        self.assertEqual(sorted(clips_two), sorted(window.project.tracks.index(piece) for piece in moving))
        # One bar of the long rest: that bar alone leaves canvas 1.
        rest = next(i for i, t in enumerate(window.project.tracks) if t.get("path") == drums["path"] and float(t["start"]) >= 2.0 - 1e-6)
        clip = view._clip_for(rest)
        grab = QPointF(clip.pos().x() + clip.rect().width() * 0.1, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, window.timeline2, QPointF(400, 40))
        held = self.coverage(drums["path"])
        self.assertIn((2.0, 4.0), held[1])
        self.assertFalse(any(a < 4.0 - 1e-6 and b > 2.0 + 1e-6 for a, b in held[0]), held[0])  # no copy left behind
        self.assertIn((4.0, 8.0), held[0])
        self.assertLabelsInside("after the drags")

    def test_a_cancelled_drag_and_a_failed_transfer_lose_nothing(self):
        pieces, first, second, bass = self._half()
        window = self.window
        view = window.timeline
        before = [(id(track), track_canvas(track), round(track["start"], 6), round(track["duration"], 6), round(track.get("offset", 0.0), 6), id(track["_audio"])) for track in window.project.tracks]
        clip = view._clip_for(pieces[0])
        grab = QPointF(clip.pos().x() + 16, clip.pos().y() + clip.rect().height() * 0.7)
        self.drag(view, grab, view, QPointF(grab.x(), -4000))
        after = [(id(track), track_canvas(track), round(track["start"], 6), round(track["duration"], 6), round(track.get("offset", 0.0), 6), id(track["_audio"])) for track in window.project.tracks]
        self.assertEqual(after, before)
        self.assertIn(pieces[0], self.clips(view))
        # A drop onto a row whose join fails puts the clip back, audio included.
        window._move_track_to_canvas(window.project.tracks.index(bass), 1, start=0.0)
        index = window.project.tracks.index(first)
        snapshot = (
            id(first["_audio"]), track_canvas(first), float(first["start"]),
            float(first["duration"]), float(first["offset"]), first.get("stem_group"),
        )
        count = len(window.project.tracks)
        undo = window.history.can_undo

        def fail(*_args, **_kwargs):
            raise RuntimeError("transfer failed")

        window._join_row = fail
        try:
            with self.assertRaises(RuntimeError):
                window._drop_on_canvas(view, [index], index, 0.4, 0)
        finally:
            del window._join_row
        self.assertEqual(len(window.project.tracks), count)
        restored = next(track for track in window.project.tracks if id(track["_audio"]) == snapshot[0] and abs(float(track["duration"]) - snapshot[3]) < 1e-6 and abs(float(track["offset"]) - snapshot[4]) < 1e-6 and track.get("bar_block"))
        self.assertEqual(track_canvas(restored), 0)
        self.assertAlmostEqual(float(restored["start"]), snapshot[2])
        self.assertEqual(restored.get("stem_group"), snapshot[5])
        self.assertEqual(window.history.can_undo, undo)
        self.assertEqual(track_canvas(second), 0)
        self.assertEqual(id(second["_audio"]), snapshot[0])

    def test_playback_after_the_move_uses_the_moved_clip(self):
        pieces, first, second, _bass = self._half()
        window = self.window
        room = window.project.tracks[self.add_tone("room", NORMAL, 0.5)]
        # add_tone reloads after the cut, so find the halves again by identity.
        index = window.project.tracks.index(first)
        audio = id(first["_audio"])
        window._drop_on_canvas(window.timeline, [index], index, 0.0, 0)
        self.assertEqual(track_canvas(first), 1)
        self.assertEqual(len([track for track in window.project.tracks if id(track) == id(first)]), 1)
        moved = window.project.tracks.index(first)
        normal = window.project.tracks.index(room)
        self.tones = [tone_of(0), tone_of(1), NORMAL]
        self.play(moved, normal)
        playing = [voice for voice in self.engine._voices.values() if voice.playing]
        self.assertEqual({voice.clip for voice in playing}, {moved, normal})
        self.assertEqual(len({voice.group for voice in playing}), 1)
        self.assertEqual(self.engine._voices[next(voice.key for voice in playing if voice.clip == moved)].clip, moved)
        mixed = self.mix(int(0.15 * OUTPUT_RATE / BLOCK))
        self.assertEqual(self.onset_freq(mixed, tone_of(0)), 0)
        self.assertEqual(self.onset_freq(mixed, NORMAL), 0)
        self.assert_level(mixed, tone_of(0), 0.15, 0.25)
        gap = self.levels(mixed, tone_of(1))
        self.assertTrue(np.all(gap < OFF), f"the piece that stayed behind sounded: {gap.max():.3f}")
        self.assertEqual(track_canvas(second), 0)
        self.assertEqual(id(first["_audio"]), audio)

    def test_dragging_one_bar_of_a_long_stem_still_leaves_the_rest(self):
        drums, *_rest = self.make("Song", [0.25] * 4)
        # Not cut: four bars of one stem. Dragging one bar extracts it.
        count = len(self.window.project.tracks)
        audio = id(drums["_audio"])
        self.window._move_block(self.index(drums), 0.25, 0.25, 1.0, False, count, 1, canvas=1)
        tracks = self.window.project.tracks
        piece = next(track for track in tracks if track.get("bar_block") and track.get("path") == drums["path"])
        rest = [track for track in tracks if track.get("path") == drums["path"] and track is not piece]
        self.assertEqual(track_canvas(piece), 1)
        self.assertEqual(id(piece["_audio"]), audio)
        self.assertEqual(len(rest), 2)  # head and tail around the taken bar
        covered = sorted(
            (round(float(track.get("offset", 0.0)), 6), round(float(track.get("offset", 0.0)) + float(track["duration"]), 6))
            for track in rest + [piece]
        )
        self.assertEqual(covered, [(0.0, 0.25), (0.25, 0.5), (0.5, 8.0)])


def _only_own_tests(cls, parent):
    for name in dir(parent):
        if name.startswith("test_"):
            setattr(cls, name, None)


_only_own_tests(SelectionCanvasTests, ShortBlockTests)
del ShortBlockTests


if __name__ == "__main__":
    unittest.main()
