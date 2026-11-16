import threading
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

from soniclab.core.ring_buffer import StereoRing

OUTPUT_RATE = 44100
BLOCK_SIZE = 512
RING_SECONDS = 2
# Blocks of playback history kept for drawing the playhead (about 0.2 s).
HEARD_BLOCKS = 16
# Preview voices: the limiter keeps their sum under this peak, then lets go
# over about RELEASE_SECONDS. It only runs while a voice is playing, so the
# song itself plays and exports exactly as before.
LIMIT_CEILING = 0.98
LIMIT_RELEASE_SECONDS = 0.25
# A stem loop's wrap crossfades this many samples (about 1.5 ms): the loop
# start fades in while the audio just past the loop end fades out, so the
# jump has no click and the cycle stays exactly end - start samples long.
LOOP_XFADE = 64

# A block voice's state (``AudioEngine.voice_states``). Each voice has its
# own: pausing, resuming or stopping one never changes another's.
STOPPED = "stopped"  # no voice under the key
PLAYING = "playing"
PAUSED = "paused"  # holds its place until resumed or stopped, however long
COMPLETED = "completed"  # a block that plays once reached its end by itself


class _Voice:
    """One stem block played on its own, apart from the transport.

    A voice with a ``group`` is a stem of a split song. It has no cursor of
    its own: it plays the frame of its file that the group's clock is on,
    so every stem of one split stays on the same sample. ``shift`` (frames)
    moves it to another place in its file on the same clock: a stem looping
    bar 5 while the clock loops bar 2 plays bar 5 in step with it.
    ``length`` (frames, or None) is the voice's own loop inside its group's:
    a block shorter than the group's loop repeats on its own length from
    the loop start (one bar loops twice in a two-bar loop), wrapping on the
    sample with the same crossfade, so it never plays on into the next bar
    of its track.
    """

    __slots__ = (
        "key", "clip", "start", "end", "cursor", "playing", "group", "muted", "shift", "length", "cycle", "generation",
    )

    def __init__(self, key, clip: int, start: int, end: int, group=None):
        self.key = key
        self.clip = int(clip)
        self.start = int(start)
        self.end = int(end)
        self.cursor = int(start)
        self.playing = True
        self.group = group
        self.muted = False
        self.shift = 0
        self.length = None
        # A block of a combined loop. ``cycle`` is (0, its row's length):
        # each row loops on its own length, never cut short by a longer
        # one, and every row follows the shared clock (``_master``), so they
        # stay in time. The block sounds inside its own [start, end) of the
        # row. ``shift`` turns row time into its arrangement place.
        self.cycle = None
        # Set by the engine when the voice is made: a key reused for a new
        # voice gets a new number, so a pause, resume or stop meant for the
        # old one (a late click, a stale callback) cannot reach the new one.
        self.generation = 0


class _Group:
    """The shared clock of one split's stems: a frame in the source song
    (the stem files' own time, at OUTPUT_RATE). It moves on by each audio
    block while any member plays and holds while all are paused.

    With a ``loop`` (start, end frames) the clock wraps from end to start
    inside the audio callback, on the exact sample, for every member at
    once. ``fading`` is set from a wrap until the crossfade is done."""

    __slots__ = ("key", "cursor", "running", "loop", "fading", "segments", "next_cursor", "next_fading")

    def __init__(self, key, cursor: int, loop=None):
        self.key = key
        self.cursor = int(cursor)
        self.running = False
        self.loop = loop
        self.fading = False
        self.segments = ()
        self.next_cursor = self.cursor
        self.next_fading = False

    def plan(self, frames: int) -> None:
        """This block as ``(frame, span, fading)`` runs of the clock, and
        where the clock is after it. Wraps at the loop end; a loop moved
        to before the clock wraps on this block."""
        loop = self.loop
        cursor = self.cursor
        fading = self.fading
        segments = []
        remaining = int(frames)
        while remaining > 0:
            if loop is not None and cursor >= loop[1]:
                cursor = loop[0]
                fading = True
            span = remaining if loop is None else min(remaining, loop[1] - cursor)
            segments.append((cursor, span, fading))
            cursor += span
            remaining -= span
            if fading and (loop is None or cursor >= loop[0] + _xfade(loop)):
                fading = False
        self.segments = segments
        self.next_cursor = cursor
        self.next_fading = fading


def _loop_frames(loop):
    """``(start, end)`` seconds as frames, or None for no (or an empty) loop."""
    if loop is None:
        return None
    start = max(0, int(round(float(loop[0]) * OUTPUT_RATE)))
    end = int(round(float(loop[1]) * OUTPUT_RATE))
    return (start, end) if end > start else None


def _xfade(loop) -> int:
    return max(0, min(LOOP_XFADE, (loop[1] - loop[0]) // 4))


def _own_length(voice, loop):
    """A stem voice's own loop length in frames, or None when it loops with
    its group (no length, no loop, or as long as the loop)."""
    if voice.length is None or loop is None or voice.length <= 0 or voice.length >= loop[1] - loop[0]:
        return None
    return voice.length


def _wrap(frame: float, loop) -> float:
    """``frame`` folded into the loop, for frames past its end."""
    if loop is None or frame < loop[1]:
        return frame
    return loop[0] + (frame - loop[0]) % (loop[1] - loop[0])


def mix_clip_into(out, clip, start) -> float:
    """Add one clip's sound for ``len(out)`` frames from arrangement frame
    ``start`` into ``out``, with its gain, pan and fades. Mute and solo are
    the caller's business. Returns the peak it added."""
    frames = len(out)
    audio = clip["audio"]
    offset = int(clip.get("offset", 0))
    length = int(clip.get("length", max(0, len(audio) - offset)))
    gain = float(clip["gain"]) * float(clip.get("clip_gain", 1.0))
    pan = clip["pan"]
    left_gain = gain * (1.0 - max(0.0, pan))
    right_gain = gain * (1.0 + min(0.0, pan))
    fade_in = int(clip.get("fade_in", 0))
    fade_out = int(clip.get("fade_out", 0))
    file_start = offset + int(start) - clip["start"]
    overlap_start = max(offset, file_start)
    overlap_end = min(len(audio), offset + max(0, length), file_start + frames)
    if overlap_end <= overlap_start:
        return 0.0
    dest = overlap_start - file_start
    piece_l = audio[overlap_start:overlap_end, 0] * left_gain
    piece_r = audio[overlap_start:overlap_end, 1] * right_gain
    if fade_in or fade_out:
        rel = np.arange(overlap_start - offset, overlap_end - offset, dtype=np.float32)
        env = np.ones(rel.shape, dtype=np.float32)
        if fade_in > 0:
            env *= np.clip(rel / float(fade_in), 0.0, 1.0)
        if fade_out > 0:
            env *= np.clip((length - rel) / float(fade_out), 0.0, 1.0)
        piece_l = piece_l * env
        piece_r = piece_r * env
    out[dest : dest + len(piece_l), 0] += piece_l
    out[dest : dest + len(piece_r), 1] += piece_r
    return float(max(np.max(np.abs(piece_l)), np.max(np.abs(piece_r))))


def render_clips(clips, start, frames, output_gain=1.0, master_muted=False, include=None):
    """Mix ``frames`` samples from arrangement frame ``start``, offline.

    The same gain, pan, fade, mute, solo and clipping as playback, so an
    export sounds exactly like the transport. ``include`` limits the mix to
    those clip indices; mute and solo still apply.
    """
    start = int(start)
    frames = max(0, int(frames))
    out = np.zeros((frames, 2), dtype=np.float32)
    clips = list(clips)
    soloed = any(clip.get("solo") for clip in clips)
    for index, clip in enumerate(clips):
        if include is not None and index not in include:
            continue
        if clip["muted"] or (soloed and not clip.get("solo")):
            continue
        mix_clip_into(out, clip, start)
    if output_gain != 1.0:
        out *= output_gain
    if master_muted:
        out.fill(0)
    np.clip(out, -1.0, 1.0, out=out)
    return out


class AudioEngine:
    """Small synchronized multitrack mixer backed by one PortAudio stream.

    Callers use only the public members; docs/engine_interface.md lists them
    and is the surface a native engine must match.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._clips = []
        self._position = 0
        self._playing = False
        self._stream = None
        self._master_gain = 0.85
        self._output_gain = 1.0
        self._master_muted = False
        self._error = None
        self._capture_enabled = True
        self._master_ring = StereoRing(OUTPUT_RATE * RING_SECONDS)
        self._track_rings = []
        self._track_peaks = []
        self._master_peak = 0.0
        self._scratch = np.zeros((BLOCK_SIZE, 2), dtype=np.float32)
        # Loop crossfade buffers, made once so a wrap allocates no audio.
        self._fade_head = np.zeros((LOOP_XFADE, 2), dtype=np.float32)
        self._fade_tail = np.zeros((LOOP_XFADE, 2), dtype=np.float32)
        self._fade_ramp = np.arange(LOOP_XFADE, dtype=np.float32) / LOOP_XFADE
        self._loop_start = None
        self._loop_end = None
        # What the speakers play and when, for drawing the playhead: one
        # (monotonic time the block starts sounding, ((frame, span), ...))
        # entry per recent block. The callback swaps in a new tuple, so the
        # UI reads it without the lock and never waits on the audio thread.
        self._heard: tuple = ()
        # Preview voices by key, mixed on top of the transport. The UI reads
        # ``_preview_view`` (key -> (playing, start, end, frame, sounds at)),
        # a tuple swapped in whole, so it never waits on the audio thread.
        self._voices: dict = {}
        self._preview_view: dict = {}
        # Stem groups by key, and their clocks for the UI: key -> (running,
        # cursor, sounds at), swapped in whole like ``_preview_view``.
        self._groups: dict = {}
        self._group_view: dict = {}
        # The one clock every block loop shares: frames since the first of
        # the voices now playing started. It moves on while any plays and
        # holds while none does. A loop started (or resumed) while others
        # play joins it at its phase. Resuming keeps a voice's own place.
        self._master = 0
        # Voice generations (see _Voice.generation), and the keys of blocks
        # that played to their end by themselves, by generation.
        self._generations = 0
        self._completed: dict = {}
        self._state_view: dict = {}
        self._limit_gain = 1.0

    @property
    def position(self):
        with self._lock:
            return self._position / OUTPUT_RATE

    @property
    def playing(self):
        with self._lock:
            return self._playing

    def heard_position(self, now: float | None = None) -> float:
        """Seconds playing out of the speakers at ``now`` (``time.monotonic()``).

        For drawing only: it interpolates between audio blocks and allows
        for the output latency, so a playhead redrawn at any rate moves
        smoothly and lines up with what is heard. It never takes the engine
        lock. Stopped, or before the first block sounds, it is ``position``.
        """
        heard = self._heard
        if not heard:
            return self._position / OUTPUT_RATE
        now = time.monotonic() if now is None else float(now)
        for start, segments in reversed(heard):
            if start > now:
                continue
            offset = (now - start) * OUTPUT_RATE
            for frame, span in segments:
                if offset < span:
                    return (frame + offset) / OUTPUT_RATE
                offset -= span
            # The next block is late: hold at the end of this one.
            frame, span = segments[-1]
            return (frame + span) / OUTPUT_RATE
        # Nothing has sounded yet: the first block starts the playhead.
        return heard[0][1][0][0] / OUTPUT_RATE

    @property
    def output_gain(self) -> float:
        """Master output gain, as set by set_output_gain()."""
        with self._lock:
            return float(self._output_gain)

    @property
    def master_muted(self) -> bool:
        with self._lock:
            return bool(self._master_muted)

    @property
    def clip_count(self) -> int:
        """Number of clips. Clip index equals project track index."""
        with self._lock:
            return len(self._clips)

    def add_clip(self, audio, start_seconds=0.0) -> int:
        """Append a clip and return its index."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        if audio.shape[1] > 2:
            audio = audio[:, :2]
        with self._lock:
            self._clips.append({
                "audio": np.ascontiguousarray(audio),
                "start": max(0, int(start_seconds * OUTPUT_RATE)),
                "offset": 0,
                "length": int(audio.shape[0]),
                "fade_in": 0,
                "fade_out": 0,
                "clip_gain": 1.0,
                "gain": 1.0,
                "pan": 0.0,
                "muted": False,
                "solo": False,
            })
            self._track_rings.append(StereoRing(OUTPUT_RATE * RING_SECONDS))
            self._track_peaks.append(0.0)
            return len(self._clips) - 1

    def batch(self):
        """Hold the audio thread off while several calls change the clips,
        so a block is never mixed from half the changes (a clip missing, or
        heard before its mute or gain is set). Keep the work inside short."""
        return self._lock

    def clear_clips(self) -> None:
        """Remove every clip. Transport position and loop are unchanged."""
        self.replace_clips([])

    def replace_clips(self, clips):
        # Preview voices find their clip by index each block. Clips that
        # come back at the same index keep playing; a voice whose clip is
        # gone ends on the next block.
        with self._lock:
            self._clips = clips
            self._track_rings = [StereoRing(OUTPUT_RATE * RING_SECONDS) for _ in clips]
            self._track_peaks = [0.0 for _ in clips]

    def set_clip_start(self, index, seconds):
        with self._lock:
            if 0 <= index < len(self._clips):
                self._clips[index]["start"] = max(0, int(seconds * OUTPUT_RATE))

    def set_clip_gain(self, index, gain):
        with self._lock:
            if 0 <= index < len(self._clips):
                self._clips[index]["gain"] = float(gain)

    def set_clip_pan(self, index, pan):
        with self._lock:
            if 0 <= index < len(self._clips):
                self._clips[index]["pan"] = float(max(-1.0, min(1.0, pan)))

    def set_clip_muted(self, index, muted):
        with self._lock:
            if 0 <= index < len(self._clips):
                self._clips[index]["muted"] = bool(muted)

    def set_clip_solo(self, index, solo):
        with self._lock:
            if 0 <= index < len(self._clips):
                self._clips[index]["solo"] = bool(solo)

    def set_clip_region(self, index, start, offset, duration, fade_in=0.0, fade_out=0.0, clip_gain=1.0):
        with self._lock:
            if not (0 <= index < len(self._clips)):
                return
            clip = self._clips[index]
            clip["start"] = max(0, int(float(start) * OUTPUT_RATE))
            clip["offset"] = max(0, int(float(offset) * OUTPUT_RATE))
            clip["length"] = max(1, int(float(duration) * OUTPUT_RATE))
            clip["fade_in"] = max(0, int(float(fade_in) * OUTPUT_RATE))
            clip["fade_out"] = max(0, int(float(fade_out) * OUTPUT_RATE))
            clip["clip_gain"] = float(clip_gain)

    def render(self, start_frame, frames, include=None):
        """The mix of ``frames`` samples from ``start_frame``, as heard."""
        with self._lock:
            clips = [dict(clip) for clip in self._clips]
            gain, muted = self._output_gain, self._master_muted
        return render_clips(clips, start_frame, frames, gain, muted, include)

    def set_output_gain(self, gain):
        with self._lock:
            self._output_gain = float(max(0.0, min(1.5, gain)))

    def set_master_muted(self, muted):
        with self._lock:
            self._master_muted = bool(muted)

    def set_capture(self, enabled):
        with self._lock:
            self._capture_enabled = bool(enabled)
            if not enabled:
                self._master_ring.clear()
                for ring in self._track_rings:
                    ring.clear()

    def read_master(self, count):
        return self._master_ring.latest(count)

    def read_track(self, index, count):
        with self._lock:
            ring = self._track_rings[index] if 0 <= index < len(self._track_rings) else None
        if ring is None:
            return np.zeros((0, 2), dtype=np.float32)
        return ring.latest(count)

    def take_peaks(self):
        with self._lock:
            master = float(self._master_peak)
            tracks = list(self._track_peaks)
            self._master_peak = 0.0
            for index in range(len(self._track_peaks)):
                self._track_peaks[index] = 0.0
            return master, tracks

    def seek(self, seconds):
        with self._lock:
            self._position = max(0, int(seconds * OUTPUT_RATE))
            # Show the new place at once and hold it there until the first
            # block from it reaches the speakers.
            self._heard = ((time.monotonic(), ((self._position, 0),)),) if self._playing else ()

    def set_loop(self, start_seconds, end_seconds) -> bool:
        """Loop in arrangement samples. The end is exclusive."""
        start = int(round(float(start_seconds) * OUTPUT_RATE))
        end = int(round(float(end_seconds) * OUTPUT_RATE))
        if end <= start or start < 0:
            return False
        with self._lock:
            self._loop_start = start
            self._loop_end = end
        return True

    def clear_loop(self) -> None:
        with self._lock:
            self._loop_start = None
            self._loop_end = None

    @property
    def loop(self):
        with self._lock:
            if self._loop_start is None or self._loop_end is None:
                return None
            return (self._loop_start / OUTPUT_RATE, self._loop_end / OUTPUT_RATE)

    def _playback_segments(self, position: int, frames: int):
        """One linear span, or a wrap when this block crosses the loop end."""
        start = self._loop_start
        end = self._loop_end
        looping = start is not None and end is not None and end > start
        segments = []
        cursor = int(position)
        remaining = int(frames)
        while remaining > 0:
            if looping and cursor >= end:
                cursor = start
            span = remaining
            if looping and cursor < end:
                span = min(span, end - cursor)
            if span <= 0:
                break
            segments.append((cursor, span))
            cursor += span
            remaining -= span
        return segments, cursor

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        with self._lock:
            playing = self._playing
            if not playing and not self._voices:
                return
            frames = int(frames)
            if frames <= 0:
                return
            sounds_at = self._sounds_at(time_info)
            if playing:
                self._mix_transport(outdata, frames, sounds_at)
            voiced = bool(self._voices)
            if voiced:
                self._mix_voices(outdata, frames, sounds_at)
            if self._output_gain != 1.0:
                outdata *= self._output_gain
            if self._master_muted:
                outdata.fill(0)
            if voiced:
                self._limit(outdata, frames)
            else:
                self._limit_gain = 1.0
            np.clip(outdata, -1.0, 1.0, out=outdata)
            if frames:
                master_peak = float(np.max(np.abs(outdata)))
                if master_peak > self._master_peak:
                    self._master_peak = master_peak
            if self._capture_enabled:
                self._master_ring.push(outdata[:frames])

    def _mix_transport(self, outdata, frames: int, sounds_at: float) -> None:
        position = self._position
        clips = list(self._clips)
        soloed = any(clip.get("solo") for clip in clips)
        capture = self._capture_enabled and frames <= len(self._scratch)
        segments, next_position = self._playback_segments(position, frames)
        self._heard = (self._heard + ((sounds_at, tuple(segments)),))[-HEARD_BLOCKS:]
        for index, clip in enumerate(clips):
            heard = not clip["muted"] and not (soloed and not clip.get("solo"))
            if not heard:
                continue
            scratch = None
            if capture and index < len(self._track_rings):
                scratch = self._scratch[:frames]
                scratch.fill(0)
            peak = 0.0
            filled = 0
            for seg_position, span in segments:
                target = scratch[filled : filled + span] if scratch is not None else outdata[filled : filled + span]
                peak = max(peak, mix_clip_into(target, clip, seg_position))
                filled += span
            if scratch is not None:
                outdata[:frames] += scratch
            if index < len(self._track_peaks) and peak > self._track_peaks[index]:
                self._track_peaks[index] = peak
            if scratch is not None:
                self._track_rings[index].push(scratch)
        self._position = next_position

    def _mix_voices(self, outdata, frames: int, sounds_at: float) -> None:
        """Add every playing preview voice. Each reads the clip's audio that
        is already in memory; nothing here decodes or touches the disk.

        Stem voices read their group's clock, and every group moves on only
        after all its members are mixed, so stems that join late land on
        the same sample as the ones already playing. A looping group wraps
        mid-block where its loop ends, the same for every member."""
        clips = self._clips
        groups = self._groups
        finished = []
        for group in groups.values():
            group.running = False
            group.plan(frames)
        if any(voice.playing for voice in self._voices.values()):
            self._master += frames
        for key, voice in self._voices.items():
            if not (0 <= voice.clip < len(clips)):
                finished.append(key)
                continue
            clip = clips[voice.clip]
            if voice.group is not None:
                group = groups.get(voice.group)
                if group is None:
                    finished.append(key)
                    continue
                offset = int(clip.get("offset", 0))
                end = offset + int(clip.get("length", len(clip["audio"]) - offset))
                if group.loop is None and group.cursor + voice.shift >= end:
                    finished.append(key)
                    continue
                if not voice.playing:
                    continue
                group.running = True
                if voice.muted:
                    continue
                peak = self._mix_stem(outdata, clip, clip["start"] - offset + voice.shift, group, _own_length(voice, group.loop))
            elif voice.cycle is not None:
                if not voice.playing:
                    continue
                peak = self._mix_cycle(outdata, frames, clip, voice)
            else:
                if not voice.playing:
                    continue
                count = min(frames, voice.end - voice.cursor)
                peak = 0.0
                if count > 0:
                    peak = mix_clip_into(outdata[:count], clip, voice.cursor)
                    voice.cursor += count
                if voice.cursor >= voice.end:
                    finished.append(key)
            if voice.clip < len(self._track_peaks) and peak > self._track_peaks[voice.clip]:
                self._track_peaks[voice.clip] = peak
        for group in groups.values():
            if group.running:
                group.cursor = group.next_cursor
                group.fading = group.next_fading
        for key in finished:
            voice = self._voices.pop(key)
            self._completed[key] = voice.generation
        if finished:
            self._drop_empty_groups()
        self._publish_previews(sounds_at)

    def _mix_stem(self, outdata, clip, base: int, group, length=None) -> float:
        """One stem along its group's runs for this block. ``base`` turns a
        file frame into the clip's arrangement frame. ``length`` (frames,
        shorter than the group's loop, or None) is the stem's own loop: it
        wraps back to the loop start every ``length`` frames of the clock.

        A wrap crossfades the loop start with the audio just past the end.
        Where the clip has no audio past the end (a piece cut at its block)
        the end fades out instead, so the wrap never clicks."""
        loop = group.loop
        peak = 0.0
        filled = 0
        if loop is None:
            for frame, span, _fading in group.segments:
                peak = max(peak, mix_clip_into(outdata[filled : filled + span], clip, base + frame))
                filled += span
            return peak
        first = loop[0]
        if length is None:
            length = loop[1] - first
        fade = max(0, min(LOOP_XFADE, length // 4))
        # Where the stem had got to when the group's loop wrapped.
        wrapped_at = (loop[1] - first) % length or length
        clip_frames = int(clip.get("length", len(clip["audio"]) - int(clip.get("offset", 0))))
        room = clip["start"] + clip_frames - base  # the clock frame the clip's audio ends at
        fade_out = first + length + fade > room
        for frame, span, fading in group.segments:
            clock = frame
            end = frame + span
            while clock < end:
                if clock < first:  # a loop moved past the clock: it plays on
                    count = min(end, first) - clock
                    peak = max(peak, mix_clip_into(outdata[filled : filled + count], clip, base + clock))
                    clock += count
                    filled += count
                    continue
                phase = (clock - first) % length
                count = min(end - clock, length - phase)
                if fade_out and phase < length - fade:
                    count = min(count, length - fade - phase)
                target = outdata[filled : filled + count]
                if fade_out and phase >= length - fade:
                    self._mix_fade_out(target, clip, base + first + phase, length - phase)
                else:
                    faded = 0
                    if phase < fade:
                        tail = None
                        if fading and clock - first < fade:
                            tail = first + wrapped_at + phase  # the group just wrapped
                        elif clock - first >= length:
                            tail = first + length + phase  # the stem's own wrap
                        if tail is not None:
                            faded = min(count, fade - phase)
                            self._mix_blend(target[:faded], clip, base + first + phase, base + tail, phase, fade)
                    if count > faded:
                        peak = max(peak, mix_clip_into(target[faded:], clip, base + first + phase + faded))
                clock += count
                filled += count
        return peak

    def _mix_fade_out(self, target, clip, start: int, left: int) -> None:
        """The last ``left`` (at most LOOP_XFADE) frames before a stem
        block's loop end, fading to silence."""
        count = len(target)
        buffer = self._fade_tail[:count]
        buffer.fill(0)
        mix_clip_into(buffer, clip, start)
        buffer *= self._fade_ramp[left - count : left][::-1, None]
        target += buffer

    def _mix_blend(self, target, clip, head_at: int, tail_at: int, into: int, length: int) -> None:
        """A loop start (from arrangement frame ``head_at``) fading in over
        the audio it replaces (from ``tail_at``) fading out, ``into`` frames
        into a crossfade of ``length``. Equal gain: the two are the same
        take, so they add."""
        count = len(target)
        head = self._fade_head[:count]
        tail = self._fade_tail[:count]
        head.fill(0)
        tail.fill(0)
        mix_clip_into(head, clip, head_at)
        mix_clip_into(tail, clip, tail_at)
        rise = self._fade_ramp[into : into + count] * (LOOP_XFADE / max(1, length))
        head -= tail
        head *= rise[:, None]
        head += tail
        target += head

    def _drop_empty_groups(self) -> None:
        used = {voice.group for voice in self._voices.values()}
        for key in [key for key in self._groups if key not in used]:
            del self._groups[key]

    def _sharing(self, starting=()) -> bool:
        """Whether voices other than ``starting`` play now, so what starts
        joins the shared clock. With none playing it restarts from zero:
        what starts now sets the time."""
        if any(voice.playing for key, voice in self._voices.items() if key not in starting):
            return True
        self._master = 0
        return False

    def _join_group(self, group) -> None:
        """Put a looping group's clock on the shared clock's phase."""
        if group.loop is not None:
            first, last = group.loop
            group.cursor = group.next_cursor = first + self._master % (last - first)
            group.fading = group.next_fading = False

    def _limit(self, outdata, frames: int) -> None:
        """Peak limiter for stacked preview voices: instant attack, smooth
        release, so several loud stems together do not hard-clip."""
        peak = float(np.max(np.abs(outdata[:frames]))) if frames else 0.0
        target = 1.0 if peak <= LIMIT_CEILING else LIMIT_CEILING / peak
        previous = self._limit_gain
        if target < previous:
            gain = np.full(frames, target, dtype=np.float32)
        else:
            step = frames / (LIMIT_RELEASE_SECONDS * OUTPUT_RATE)
            reach = min(target, previous + step)
            gain = np.linspace(previous, reach, frames, dtype=np.float32)
            target = reach
        self._limit_gain = float(target)
        if target < 1.0 or previous < 1.0:
            outdata[:frames] *= gain[:, None]

    @staticmethod
    def _mix_cycle(outdata, frames: int, clip, voice) -> float:
        """One block of a combined loop: walk the shared cycle, wrapping at
        its end, and add the clip only inside the voice's own block."""
        first, last = voice.cycle
        own = voice.length or (last - first)
        cursor = voice.cursor
        filled = 0
        peak = 0.0
        while filled < frames:
            if cursor >= last:
                cursor = first
            local = (cursor - first) % own  # where its row's part is now
            span = min(frames - filled, last - cursor, own - local)
            low = max(local, voice.start)
            high = min(local + span, voice.end)
            if high > low:
                at = filled + (low - local)
                peak = max(peak, mix_clip_into(outdata[at : at + (high - low)], clip, low + voice.shift))
            cursor += span
            filled += span
        voice.cursor = cursor
        return peak

    def preview_loop(self, blocks) -> None:
        """Play blocks as one combined loop on one shared clock.

        ``blocks`` is a list of ``(key, clip index, start, end, row)``, with
        start and end in arrangement seconds. Every row starts at the same
        moment: a row's blocks play one after another, with the empty time
        between them left out. Each row loops on its own length with no gap,
        and is never cut or restarted when a longer row ends. Every row
        reads the shared clock, so rows stay in time and meet again where
        their lengths do. Voices already under these keys are replaced; if
        any of them were playing, the loop goes on from where the clock is
        (adding a track does not restart the others). Raises if the audio
        output cannot start.
        """
        rows: dict = {}
        for key, clip, start, end, row in blocks:
            first = max(0, int(round(float(start) * OUTPUT_RATE)))
            last = int(round(float(end) * OUTPUT_RATE))
            if last > first:
                rows.setdefault(row, []).append((first, last, key, clip))
        if not rows:
            return
        placed = []  # (key, clip, row start, row end, shift, row length)
        for members in rows.values():
            # A row's blocks one after another, the gaps between them dropped.
            members.sort()
            cursor = 0
            laid = []
            for first, last, key, clip in members:
                laid.append((key, clip, cursor, cursor + (last - first), first - cursor))
                cursor += last - first
            length = cursor
            placed.extend((key, clip, a, b, shift, length) for key, clip, a, b, shift in laid)
        with self._lock:
            keys = {item[0] for item in placed}
            going = any(self._voices[key].playing for key in keys if key in self._voices)
            # Joining loops already playing (or this loop growing while it
            # plays): each row starts at the shared clock's place in it.
            if not going:
                self._sharing(keys)  # with nothing else playing, the clock starts at zero
            for key, clip, a, b, shift, length in placed:
                voice = self._new_voice(key, clip, a, b)
                voice.shift = shift
                voice.length = length
                voice.cycle = (0, length)
                voice.cursor = self._master % length
                self._voices[key] = voice
            self._publish_previews()
        self._ensure_stream()

    def _publish_previews(self, sounds_at: float | None = None) -> None:
        """Swap in what the UI reads about each voice. From the callback,
        ``sounds_at`` is when the block just written starts to sound. From
        the UI thread (play, pause, resume) there is none: a voice whose
        cursor did not move keeps the stamp of the block that moved it, so
        pausing one voice does not shift the time shown for the others, and
        a voice that starts or resumes reads its cursor as heard now."""
        now = time.monotonic()
        previous = self._preview_view
        previous_groups = self._group_view
        fresh = now - BLOCK_SIZE / OUTPUT_RATE  # interpolates to the cursor itself, now

        def stamp(old, moving, cursor):
            if sounds_at is not None:
                return sounds_at
            if old is not None and old[0] and moving and old[1] == cursor:
                return old[2]
            return fresh if moving else now

        clips = self._clips
        groups = self._groups
        view = {}
        for key, voice in self._voices.items():
            old = previous.get(key)
            old = (old[1], old[4], old[6]) if old is not None else None
            if voice.group is None:
                when = stamp(old, voice.playing, voice.cursor)
                if voice.cycle is not None:
                    own = voice.length or None
                    view[key] = (voice.playing, voice.playing, voice.start, voice.end, voice.cursor, voice.shift, when, voice.cycle, own)
                else:
                    view[key] = (voice.playing, voice.playing, voice.start, voice.end, voice.cursor, 0, when, None, None)
                continue
            group = groups.get(voice.group)
            if group is None or not (0 <= voice.clip < len(clips)):
                continue
            clip = clips[voice.clip]
            offset = int(clip.get("offset", 0))
            end = offset + int(clip.get("length", len(clip["audio"]) - offset))
            # File frames, shown in arrangement time on this clip.
            # A paused stem shows where it was paused (``cursor``), still.
            moving = voice.playing and group.running
            cursor = group.cursor if voice.playing else voice.cursor
            view[key] = (
                voice.playing, moving, offset - voice.shift, end - voice.shift,
                cursor,
                clip["start"] - offset + voice.shift, stamp(old, moving, cursor), group.loop, _own_length(voice, group.loop),
            )
        self._preview_view = view
        states = {key: (COMPLETED, generation) for key, generation in self._completed.items()}
        for key, voice in self._voices.items():
            states[key] = (PLAYING if voice.playing else PAUSED, voice.generation)
        self._state_view = states
        group_view = {}
        for key, group in groups.items():
            old = previous_groups.get(key)
            old = (old[0], old[1], old[2]) if old is not None else None
            group_view[key] = (group.running, group.cursor, stamp(old, group.running, group.cursor), group.loop)
        self._group_view = group_view

    def _sounds_at(self, time_info) -> float:
        """Monotonic time at which the block being written starts to sound."""
        latency = 0.0
        try:
            latency = float(time_info.outputBufferDacTime) - float(time_info.currentTime)
        except (AttributeError, TypeError, ValueError):
            pass
        if not 0.0 < latency < 1.0:
            stream_latency = getattr(self._stream, "latency", 0.0)
            latency = float(stream_latency) if isinstance(stream_latency, (int, float)) else 0.0
        return time.monotonic() + max(0.0, min(1.0, latency))

    def play(self):
        with self._lock:
            if self._playing:
                return
            self._heard = ()
            self._ensure_stream()
            self._playing = True

    def pause(self):
        with self._lock:
            self._playing = False
            self._heard = ()

    def stop(self):
        with self._lock:
            self._playing = False
            self._position = 0
            self._heard = ()

    def preview_play(self, blocks) -> None:
        """Start or resume preview voices, all on the same audio block.

        ``blocks`` is a list of ``(key, clip index, start seconds, end
        seconds)`` in arrangement time. A paused voice resumes where it
        stopped; a playing one carries on. The transport is not touched.
        Raises if the audio output cannot start.
        """
        with self._lock:
            for key, clip, start, end in blocks:
                voice = self._voices.get(key)
                if voice is not None:
                    voice.playing = True
                    continue
                first = max(0, int(round(float(start) * OUTPUT_RATE)))
                last = int(round(float(end) * OUTPUT_RATE))
                if last > first:
                    self._voices[key] = self._new_voice(key, clip, first, last)
            self._publish_previews()
        self._ensure_stream()

    def stem_play(self, group, members, start_seconds: float = 0.0, loop=None) -> None:
        """Start or resume stems of one split on the group's shared clock.

        ``members`` is a list of ``(key, clip index)``, ``(key, clip
        index, muted)``, ``(key, clip index, muted, shift seconds)`` or
        ``(key, clip index, muted, shift seconds, length seconds)``: a shift
        plays that much later in the stem's file on the same clock, and a
        length is how much of each loop cycle it plays (see ``_Voice``). If the group is not playing yet its clock starts at
        ``start_seconds`` of the source song (the stem files' own time);
        otherwise the stems join it where it is now, on the next audio
        block, sample-aligned with the stems already playing. A paused stem
        resumes at the shared position, not where it paused. Other groups
        and block voices are not touched. Raises if the audio output cannot
        start.

        ``loop`` is ``(start, end)`` seconds of the source song: the group
        then repeats that region, wrapping on the sample (see
        ``group_set_loop``). A new looping group starts at the loop start.
        """
        frames = _loop_frames(loop)
        with self._lock:
            if group not in self._groups:
                first = max(0, int(round(float(start_seconds) * OUTPUT_RATE)))
                if frames is not None:
                    first = frames[0]
                self._groups[group] = _Group(group, first, frames)
                if self._sharing({member[0] for member in members}):
                    self._join_group(self._groups[group])
            elif loop is not None:
                self._groups[group].loop = frames
            for member in members:
                key, clip = member[0], int(member[1])
                voice = self._voices.get(key)
                if voice is None or voice.group != group:
                    voice = self._new_voice(key, clip, 0, 0, group)
                    self._voices[key] = voice
                voice.clip = clip
                voice.playing = True
                if len(member) > 2:
                    voice.muted = bool(member[2])
                if len(member) > 3:
                    voice.shift = int(round(float(member[3]) * OUTPUT_RATE))
                if len(member) > 4:
                    voice.length = None if member[4] is None else int(round(float(member[4]) * OUTPUT_RATE))
            self._drop_empty_groups()
            self._publish_previews()
        self._ensure_stream()

    def group_set_loop(self, group, loop) -> None:
        """Change a playing group's loop to ``(start, end)`` seconds of the
        source song, or None to stop looping. Safe while it plays: the
        callback picks it up on its next block; a clock already past the
        new end wraps to the new start there, crossfaded. A group that is
        not playing is left alone."""
        frames = _loop_frames(loop)
        with self._lock:
            state = self._groups.get(group)
            if state is not None:
                state.loop = frames
                if frames is None:
                    state.fading = False
                self._publish_previews()

    def group_loop(self, group):
        """A playing group's loop as ``(start, end)`` seconds of the source
        song, or None. Lock-free."""
        state = self._group_view.get(group)
        if state is None or state[3] is None:
            return None
        return state[3][0] / OUTPUT_RATE, state[3][1] / OUTPUT_RATE

    def preview_set_muted(self, key, muted: bool) -> None:
        """Silence a stem voice without leaving its group's clock."""
        with self._lock:
            voice = self._voices.get(key)
            if voice is not None:
                voice.muted = bool(muted)

    def group_position(self, group, now: float | None = None) -> float | None:
        """Seconds of the source song a stem group's clock is heard at
        ``now``, or None when the group is not playing. Lock-free."""
        state = self._group_view.get(group)
        if state is None:
            return None
        running, cursor, sounds_at, loop = state
        frame = cursor
        if running:
            now = time.monotonic() if now is None else float(now)
            frame = max(0.0, cursor - BLOCK_SIZE + (now - sounds_at) * OUTPUT_RATE)
            if loop is not None and frame < loop[0] <= cursor:
                frame += loop[1] - loop[0]
            frame = _wrap(frame, loop)
        return frame / OUTPUT_RATE

    def preview_resume(self, keys, generations: dict | None = None) -> None:
        """Play paused voices on from where each one stopped, together.

        A voice keeps its own place: resuming never moves it to another
        voice's time. A stem whose split is still playing goes on at the
        split's shared position, on the same sample as its other stems."""
        with self._lock:
            for key in keys:
                voice = self._voices.get(key)
                if voice is not None and self._current(key, (generations or {}).get(key)):
                    voice.playing = True
            self._publish_previews()
        self._ensure_stream()

    def preview_pause(self, key, generation: int | None = None) -> None:
        """Pause one voice where it is heard now, not where the render head
        is (an audio block and the output latency ahead), so resuming plays
        on from the sample the listener stopped at. Other voices go on."""
        with self._lock:
            voice = self._voices.get(key)
            if voice is not None and voice.playing and self._current(key, generation):
                if voice.group is not None:
                    group = self._groups.get(voice.group)
                    heard = self.group_position(voice.group)
                    if group is not None and heard is not None:
                        frame = int(round(heard * OUTPUT_RATE))
                        voice.cursor = frame
                        own = _own_length(voice, group.loop)
                        if own is not None:
                            voice.cursor = group.loop[0] + (frame - group.loop[0]) % own
                        others = any(
                            other.playing for other in self._voices.values()
                            if other is not voice and other.group == voice.group
                        )
                        if not others:
                            # The split's clock holds where it was heard.
                            group.cursor = group.next_cursor = frame
                            group.fading = group.next_fading = False
                else:
                    frame = self._heard_frame(key)
                    if frame is not None:
                        voice.cursor = frame
                voice.playing = False
            self._publish_previews()

    def _heard_frame(self, key) -> int | None:
        """Where a block voice's own cursor is heard now (lock held)."""
        entry = self._preview_view.get(key)
        if entry is None or not entry[1]:
            return None
        _playing, _moving, _start, _end, cursor, _base, sounds_at, loop, _own = entry
        frame = cursor - BLOCK_SIZE + (time.monotonic() - sounds_at) * OUTPUT_RATE
        if loop is not None and frame < loop[0] <= cursor:
            frame += loop[1] - loop[0]
        frame = _wrap(frame, loop)
        return max(0, int(round(frame)))

    def preview_stop(self, key, generation: int | None = None) -> None:
        with self._lock:
            if not self._current(key, generation):
                return
            self._voices.pop(key, None)
            self._completed.pop(key, None)
            self._drop_empty_groups()
            self._publish_previews()

    def preview_retarget(self, changes: dict) -> None:
        """Clips were reloaded in a new order: ``old key -> (new key, clip
        index)`` keeps those voices playing from the same place."""
        with self._lock:
            moved = []
            for old, (new, clip) in changes.items():
                voice = self._voices.pop(old, None)
                if voice is not None:
                    voice.key = new
                    voice.clip = int(clip)
                    moved.append(voice)
            for voice in moved:
                self._voices[voice.key] = voice
            self._publish_previews()

    def preview_stop_all(self) -> None:
        with self._lock:
            self._voices.clear()
            self._groups.clear()
            self._completed.clear()
            self._publish_previews()

    def _new_voice(self, key, clip, start, end, group=None) -> _Voice:
        """A voice with a fresh generation (lock held)."""
        self._generations += 1
        voice = _Voice(key, clip, start, end, group)
        voice.generation = self._generations
        self._completed.pop(key, None)
        return voice

    def _current(self, key, generation) -> bool:
        """Whether a request for ``generation`` (None: any) is for the voice
        under ``key`` now (lock held). A stale one is ignored."""
        if generation is None:
            return True
        voice = self._voices.get(key)
        return voice is not None and voice.generation == int(generation)

    def voice_states(self) -> dict:
        """``key -> (state, generation)`` for every voice and every block
        that completed, from the engine's own state. Lock-free. A key that
        is absent is ``STOPPED``."""
        return dict(self._state_view)

    def voice_state(self, key) -> tuple[str, int]:
        return self._state_view.get(key, (STOPPED, 0))

    def previews(self, now: float | None = None) -> dict:
        """``key -> (playing, seconds)`` for every preview voice.

        Lock-free, like ``heard_position``: the seconds are what is heard at
        ``now``, interpolated between audio blocks and inside the block.
        A stem voice's seconds are in arrangement time on its own clip.
        A finished or stopped voice is absent.
        """
        return {key: (playing, seconds) for key, playing, seconds, _inside in self._heard_previews(now)}

    def sounding_time(self, now: float | None = None) -> float | None:
        """Arrangement seconds heard now from the block voices, or None when
        none plays. Taken from a voice whose own block is sounding: in a
        combined loop each block holds its time while its row plays another
        block, so that one would stand still. Lock-free."""
        heard = sorted(
            (not inside, key, seconds)
            for key, playing, seconds, inside in self._heard_previews(now)
            if playing
        )
        return heard[0][2] if heard else None

    def _heard_previews(self, now: float | None = None):
        """``(key, playing, seconds, inside)`` per voice; ``inside`` is
        whether the heard frame is within the voice's own block."""
        view = self._preview_view
        if not view:
            return []
        now = time.monotonic() if now is None else float(now)
        shown = []
        for key, (playing, moving, start, end, cursor, base, sounds_at, loop, own) in view.items():
            frame = cursor
            if moving:
                # ``cursor`` is the end of the last block written; that block
                # starts sounding at ``sounds_at``.
                frame = cursor - BLOCK_SIZE + (now - sounds_at) * OUTPUT_RATE
                if loop is not None and frame < loop[0] <= cursor:
                    frame += loop[1] - loop[0]  # still the last cycle
                frame = _wrap(frame, loop)
                if own is not None:
                    frame = loop[0] + (frame - loop[0]) % own  # inside its own block
            inside = start <= frame < end
            shown.append((key, playing, (base + min(end, max(start, frame))) / OUTPUT_RATE, inside))
        return shown

    def _ensure_stream(self) -> None:
        with self._lock:
            if self._stream is None:
                self._stream = sd.OutputStream(
                    samplerate=OUTPUT_RATE,
                    channels=2,
                    dtype="float32",
                    blocksize=BLOCK_SIZE,
                    callback=self._callback,
                )
                self._stream.start()
            elif not self._stream.active:
                self._stream.start()

    def close(self):
        with self._lock:
            self._playing = False
            self._voices.clear()
            self._groups.clear()
            self._publish_previews()
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    @staticmethod
    def resample(audio, source_rate):
        if source_rate == OUTPUT_RATE:
            return np.asarray(audio, dtype=np.float32)
        divisor = gcd(int(source_rate), OUTPUT_RATE)
        return resample_poly(audio, OUTPUT_RATE // divisor, int(source_rate) // divisor, axis=0).astype(np.float32)
