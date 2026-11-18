import os
import math
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
        "key", "clip", "start", "end", "cursor", "playing", "group", "muted", "shift", "length", "generation", "sealed",
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
        # Set by the engine when the voice is made: a key reused for a new
        # voice gets a new number, so a pause, resume or stop meant for the
        # old one (a late click, a stale callback) cannot reach the new one.
        self.generation = 0
        # The block ends at a split mark: the audio past it is another
        # block's, so its loop fades out at its end instead of crossfading
        # into that audio.
        self.sealed = False


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


def output_latency():
    """The output buffer PortAudio is asked for. "high" is the device's
    safe latency (35 ms on a typical ALSA device, against 9 ms for "low",
    which is less than one 512-frame block): a callback held up a few
    milliseconds by the GUI (a repaint holds Python's lock while it runs)
    still finds audio queued, so it does not drop out. Quality is the same;
    the playhead already accounts for the output latency.
    SONICLAB_AUDIO_LATENCY overrides it: "low", "high" or seconds."""
    value = os.environ.get("SONICLAB_AUDIO_LATENCY", "high").strip().lower()
    if value in ("low", "high"):
        return value
    try:
        return max(0.0, float(value))
    except ValueError:
        return "high"


def _frame(seconds) -> int:
    """Seconds as the nearest sample frame. Truncating would put a stem
    whose start or offset lands a hair under a frame one sample early."""
    return int(round(float(seconds) * OUTPUT_RATE))


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


def _explicit_span(voice, group):
    """The file frames a block voice loops, ``(start, end)``, or None when
    it has no length of its own and simply follows the group loop."""
    if group is None or group.loop is None or not voice.length or voice.length <= 0:
        return None
    start = int(group.loop[0]) + int(voice.shift)
    return start, start + int(voice.length)


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


def mix_clip_rate_into(out, clip, start: float, rate: float, end: int) -> float:
    """Like ``mix_clip_into``, read ``rate`` times as fast: ``len(out)``
    output frames from arrangement frame ``start`` (fractional), never past
    arrangement frame ``end``, by linear interpolation. Gain, pan and fades
    as the clip's own. Returns the peak it added."""
    frames = len(out)
    audio = clip["audio"]
    offset = int(clip.get("offset", 0))
    length = int(clip.get("length", max(0, len(audio) - offset)))
    gain = float(clip["gain"]) * float(clip.get("clip_gain", 1.0))
    pan = clip["pan"]
    left_gain = gain * (1.0 - max(0.0, pan))
    right_gain = gain * (1.0 + min(0.0, pan))
    # Positions in the file, one per output frame.
    pos = (float(start) - clip["start"] + offset) + float(rate) * np.arange(frames, dtype=np.float64)
    limit = min(len(audio) - 1, offset + max(0, length) - 1, int(end) - clip["start"] + offset - 1)
    inside = (pos >= offset) & (pos <= limit)
    if not inside.any():
        return 0.0
    pos = np.clip(pos, 0, max(0, len(audio) - 2))
    low = pos.astype(np.int64)
    frac = (pos - low).astype(np.float32)
    high = np.minimum(low + 1, len(audio) - 1)
    left = audio[low, 0] * (1.0 - frac) + audio[high, 0] * frac
    right = audio[low, 1] * (1.0 - frac) + audio[high, 1] * frac
    env = inside.astype(np.float32)
    fade_in = int(clip.get("fade_in", 0))
    fade_out = int(clip.get("fade_out", 0))
    if fade_in > 0 or fade_out > 0:
        rel = (pos - offset).astype(np.float32)
        if fade_in > 0:
            env *= np.clip(rel / float(fade_in), 0.0, 1.0)
        if fade_out > 0:
            env *= np.clip((length - rel) / float(fade_out), 0.0, 1.0)
    left = left * env * left_gain
    right = right * env * right_gain
    out[:, 0] += left
    out[:, 1] += right
    return float(max(np.max(np.abs(left)), np.max(np.abs(right))))


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
        # Voice generations (see _Voice.generation), and the keys of blocks
        # that played to their end by themselves, by generation.
        self._generations = 0
        self._completed: dict = {}
        self._state_view: dict = {}
        # Diagnostics, written by the callback without allocating: its last
        # durations (seconds), callbacks counted, deadline misses (a block
        # took longer than it lasts) and underruns PortAudio reported.
        # The transport plays only these stretches when set (see
        # set_transport_filter); built off the audio thread, read by it.
        self._transport_filter = None
        # Member playback (see members_begin): the transport frame it began
        # at, the members by key, swapped in whole on every change, the same
        # for the UI, and the members that played to their end.
        self._member_origin = None
        self._members: dict = {}
        self._member_view = None
        self._members_done = frozenset()
        self._timing = np.zeros(4096, dtype=np.float64)
        self._timed = 0
        self._late = 0
        self._underruns = 0
        self._limit_gain = 1.0
        # Seconds from writing a block to hearing it, as the last callback
        # measured it (see _sounds_at); read by output_delay.
        self._output_delay = 0.0

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
                "start": max(0, _frame(start_seconds)),
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
                self._clips[index]["start"] = max(0, _frame(seconds))

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
            clip["start"] = max(0, _frame(start))
            clip["offset"] = max(0, _frame(offset))
            clip["length"] = max(1, _frame(duration))
            clip["fade_in"] = max(0, _frame(fade_in))
            clip["fade_out"] = max(0, _frame(fade_out))
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
            self._position = max(0, _frame(seconds))
            # Show the new place at once and hold it there until the first
            # block from it reaches the speakers.
            self._heard = ((time.monotonic(), ((self._position, 0),)),) if self._playing else ()

    def set_transport_filter(self, ranges) -> None:
        """Let the transport play only some stretches: ``{clip index:
        [(start frame, end frame), ...]}`` in arrangement frames, or None for
        every clip. The selection plays through the one arrangement mixer and
        clock this way, never on a clock of its own."""
        prepared = None
        if ranges is not None:
            prepared = {
                int(index): tuple(sorted((int(a), int(b)) for a, b in spans if int(b) > int(a)))
                for index, spans in ranges.items()
            }
        with self._lock:
            self._transport_filter = prepared

    @property
    def transport_filter(self):
        return self._transport_filter

    # -- block members on the one transport clock --------------------------
    #
    # The transport's position is the one clock. While member playback is
    # on, the transport plays its members instead of the arrangement: each
    # member is a stretch of a clip with the transport sample it started at.
    # Its place is ``position - start`` frames into its own stretch (rate 1:
    # every file is resampled to OUTPUT_RATE on load), whatever the place of
    # the stretch on the arrangement and whatever the other members do. A
    # member joins at the clock's place now and plays from its own first
    # frame. Members are added, moved and removed one at a time; the audio
    # thread sees the whole new set or the old one, never half of it.

    def members_begin(self) -> None:
        """Member playback on, with no members yet: the transport stops
        playing the arrangement. Its position is where it was."""
        with self._lock:
            if self._member_origin is None:
                self._member_origin = self._position
                self._members = {}
                self._publish_members()

    def members_end(self) -> int | None:
        """Member playback off: every member goes and the transport plays
        the arrangement again. Returns the frame member playback began at."""
        with self._lock:
            origin = self._member_origin
            self._member_origin = None
            self._members = {}
            self._members_done = frozenset()
            self._member_view = None
            return origin

    # A member entry: [clip, first, end, start, loop, held, period, rate].
    # ``rate`` is source frames per output frame (1.0: as recorded; a little
    # above plays it a little faster). ``start``, ``held`` and ``period`` are
    # in output frames (the transport's), ``first`` and ``end`` in the clip's
    # arrangement frames, its own audio.

    @staticmethod
    def _sounding(entry) -> int:
        """Output frames one pass of a member lasts at its rate."""
        length = entry[2] - entry[1]
        rate = entry[7]
        return length if rate == 1.0 else int(math.ceil(length / rate))

    def member_add(
        self,
        key,
        clip: int,
        first: int,
        end: int,
        loop: bool = False,
        into: int = 0,
        period: int | None = None,
        sync: bool = False,
        rate: float = 1.0,
    ) -> None:
        """One more member, from now: ``first``..``end`` arrangement frames of
        clip ``clip``, from its first frame (or ``into`` frames in, for a
        block that carries on from another player; a negative ``into`` waits
        that long first, for a stretch that follows others of its track).
        ``period``: output frames from one start of it to the next when it
        loops (a sequence of stretches loops as a whole), else its own
        length at ``rate``. ``rate``: source frames per output frame, to fit
        a stretch to the lead's bars (a little faster or slower).
        Other members are not touched. A key already playing is left as it
        is.

        ``sync``: it keeps time with the members already playing. Its cycle
        (``period``, else its length) is laid on the grid that began with
        member playback, so it starts where it would be had it started with
        the first ones: looping, at that place in its cycle now; not
        looping, at the start of its next cycle. ``into`` (zero or a wait)
        then counts from that start."""
        with self._lock:
            if self._member_origin is None or key in self._members or int(end) <= int(first):
                return
            members = dict(self._members)
            rate = float(rate) if rate and float(rate) > 0 else 1.0
            entry = [int(clip), int(first), int(end), 0, bool(loop), None, None, rate]
            sounding = self._sounding(entry)
            entry[6] = None if period is None or int(period) <= sounding else int(period)
            start = self._position - int(into)
            if sync and int(into) <= 0:
                cycle = entry[6] or sounding
                phase = (self._position - self._member_origin) % cycle
                base = self._position - phase
                if not loop and phase > 0:
                    base += cycle  # the next cycle on the grid, from its start
                start = base - int(into)
            entry[3] = start
            members[key] = entry
            self._members = members
            self._publish_members()

    def member_place(self, key, clip: int, first: int, end: int) -> None:
        """A member's clip or stretch moved (tracks reloaded): its place in
        its own stretch is kept."""
        with self._lock:
            entry = self._members.get(key)
            if entry is None or int(end) <= int(first):
                return
            members = dict(self._members)
            members[key] = [int(clip), int(first), int(end)] + entry[3:]
            self._members = members
            self._publish_members()

    def member_remove(self, key) -> None:
        with self._lock:
            if key in self._members:
                members = dict(self._members)
                del members[key]
                self._members = members
                self._members_done = self._members_done - {key}
                self._publish_members()

    def member_hold(self, key, held: bool, sync: bool = True) -> None:
        """Pause one member where it is, or let it play again; the clock and
        the other members go on. ``sync``: it comes back in time with them,
        where it would be had it never paused (its start sample stays, so
        it lands on the beat the others are on); a member that does not
        loop and would be past its end waits for its next cycle instead.
        Without ``sync`` it plays on from where it paused."""
        with self._lock:
            entry = self._members.get(key)
            if entry is None or (entry[5] is not None) == bool(held):
                return
            members = dict(self._members)
            entry = list(entry)
            if held:
                entry[5] = self._position - entry[3]
            elif not sync:
                entry[3], entry[5] = self._position - entry[5], None
            else:
                sounding = self._sounding(entry)
                cycle = entry[6] or sounding
                local = self._position - entry[3]
                if local > 0:
                    laps = local // cycle
                    entry[3] += laps * cycle
                    if not entry[4] and local - laps * cycle >= sounding:
                        entry[3] += cycle  # past its turn in this cycle: the next one
                entry[5] = None
            members[key] = entry
            self._members = members
            self._publish_members()

    def member_set_loop(self, loop: bool, keys=None) -> None:
        """Members wrap within their own stretch (or stop at its end). Each
        keeps its place: one that has wrapped is counted from the start of
        the pass it is in, so with looping off it finishes that pass."""
        with self._lock:
            members = dict(self._members)
            for key, entry in members.items():
                if keys is None or key in keys:
                    entry = list(entry)
                    cycle = entry[6] or self._sounding(entry)
                    if entry[5] is not None:
                        entry[5] %= cycle
                    elif self._position - entry[3] >= cycle:
                        entry[3] = self._position - (self._position - entry[3]) % cycle
                    entry[4] = bool(loop)
                    members[key] = entry
            self._members = members
            self._members_done = frozenset()
            self._publish_members()

    def _publish_members(self) -> None:
        self._member_view = {key: tuple(entry) for key, entry in self._members.items()}

    @property
    def output_delay(self) -> float:
        """Seconds from mixing a block to hearing it: what the last audio
        callback measured (PortAudio's DAC time, else the stream's reported
        latency), or the open stream's latency before any callback, 0 with
        no stream (nothing is heard, so nothing is late). Lock-free."""
        if self._output_delay > 0.0:
            return self._output_delay
        latency = getattr(self._stream, "latency", 0.0)
        return float(latency) if isinstance(latency, (int, float)) else 0.0

    @property
    def member_origin(self) -> int | None:
        """The transport frame member playback began at: the start of the
        grid every member keeps time on. Lock-free."""
        return self._member_origin

    @property
    def members(self):
        """``{key: (clip, first, end, start sample, loop, held frames or
        None, loop period or None, rate)}`` while member playback is on,
        else None. Lock-free."""
        return self._member_view

    @property
    def members_completed(self) -> frozenset:
        """Members that played to their own end (not looping). Lock-free."""
        return self._members_done

    @staticmethod
    def member_offset(entry, position: int):
        """Source frames into its stretch member ``entry`` is at transport
        frame ``position``, or None while it is silent (ended, or waiting
        for its turn)."""
        _clip, first, end, start, loop, held, period, rate = entry
        sounding = AudioEngine._sounding(entry)
        local = held if held is not None else position - start
        if loop:
            local %= period or sounding
        if not 0 <= local < sounding:
            return None
        return min(end - first - 1, int(local * rate))

    def _mix_members(self, target, clip, entries, position: int) -> float:
        """One clip's members for ``len(target)`` frames from transport
        frame ``position``. Returns the peak added."""
        frames = len(target)
        peak = 0.0
        for entry in entries:
            _clip, first, end, start, loop, held, period, rate = entry
            if held is not None:
                continue
            sounding = self._sounding(entry)
            cycle = period or sounding
            filled = 0
            while filled < frames:
                local = position + filled - start
                if loop:
                    local %= cycle
                elif local < 0:
                    filled += min(frames - filled, -local)
                    continue
                if local >= sounding:
                    if not loop:
                        break
                    filled += min(frames - filled, cycle - local)  # its turn comes round again
                    continue
                span = min(frames - filled, sounding - local)
                piece = target[filled : filled + span]
                if rate == 1.0:
                    peak = max(peak, mix_clip_into(piece, clip, first + local))
                else:
                    peak = max(peak, mix_clip_rate_into(piece, clip, first + local * rate, rate, end))
                filled += span
        return peak

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
        """The audio thread's entry: render, and time it for diagnostics
        (two clock reads and a store into a preallocated array)."""
        began = time.perf_counter()
        try:
            self._render(outdata, frames, time_info)
        finally:
            took = time.perf_counter() - began
            timing = self._timing
            timing[self._timed % len(timing)] = took
            self._timed += 1
            if took > int(frames) / OUTPUT_RATE:
                self._late += 1
            if status is not None and getattr(status, "output_underflow", False):
                self._underruns += 1

    def _render(self, outdata, frames, time_info):
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
        selection = None
        if self._member_origin is not None:
            # The shared clock runs on: the arrangement loop does not apply.
            segments, next_position = [(position, frames)], position + frames
            selection = {}
            done = set()
            for key, entry in self._members.items():
                selection.setdefault(entry[0], []).append(entry)
                if not entry[4] and entry[5] is None and position - entry[3] >= self._sounding(entry):
                    done.add(key)
            if done != self._members_done:
                self._members_done = frozenset(done)
        else:
            segments, next_position = self._playback_segments(position, frames)
        self._heard = (self._heard + ((sounds_at, tuple(segments)),))[-HEARD_BLOCKS:]
        allowed = self._transport_filter
        for index, clip in enumerate(clips):
            heard = not clip["muted"] and not (soloed and not clip.get("solo"))
            if not heard:
                continue
            if selection is not None:
                spans = selection.get(index)
                if not spans:
                    continue  # not selected
                scratch = None
                if capture and index < len(self._track_rings):
                    scratch = self._scratch[:frames]
                    scratch.fill(0)
                peak = self._mix_members(scratch if scratch is not None else outdata[:frames], clip, spans, position)
                if scratch is not None:
                    outdata[:frames] += scratch
                    self._track_rings[index].push(scratch)
                if index < len(self._track_peaks) and peak > self._track_peaks[index]:
                    self._track_peaks[index] = peak
                continue
            ranges = None
            if allowed is not None:
                ranges = allowed.get(index)
                if not ranges:
                    continue  # not in the filtered selection
            scratch = None
            if capture and index < len(self._track_rings):
                scratch = self._scratch[:frames]
                scratch.fill(0)
            peak = 0.0
            filled = 0
            for seg_position, span in segments:
                target = scratch[filled : filled + span] if scratch is not None else outdata[filled : filled + span]
                if ranges is None:
                    peak = max(peak, mix_clip_into(target, clip, seg_position))
                else:
                    # Only the selected stretches of this clip, each where it
                    # sits on the arrangement: one clock, one mix.
                    for first, last in ranges:
                        low, high = max(first, seg_position), min(last, seg_position + span)
                        if high > low:
                            at = low - seg_position
                            peak = max(peak, mix_clip_into(target[at : at + (high - low)], clip, low))
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
        # Solo silences the other voices; like mute it never moves a clock.
        soloed = any(clip.get("solo") for clip in clips)
        for group in groups.values():
            group.running = False
            group.plan(frames)
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
                if voice.muted or (soloed and not clip.get("solo")):
                    continue
                peak = self._mix_stem(
                    outdata, clip, clip["start"] - offset + voice.shift, group, _own_length(voice, group.loop),
                    voice.length if voice.sealed else None,
                )
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

    def _mix_stem(self, outdata, clip, base: int, group, length=None, stretch=None) -> float:
        """One stem along its group's runs for this block. ``base`` turns a
        file frame into the clip's arrangement frame. ``length`` (frames,
        shorter than the group's loop, or None) is the stem's own loop: it
        wraps back to the loop start every ``length`` frames of the clock.

        A wrap crossfades the loop start with the audio just past the end.
        Where there is none of this voice's own (the clip ends, or
        ``stretch``, the block's length in frames, does) the end fades out
        instead, so the wrap never clicks and never plays the next block."""
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
        if stretch:
            # A block that ends at a split mark plays its own stretch only:
            # past it is the next block's audio, so the wrap fades out and in
            # rather than crossfading into it. A bar's loop does crossfade
            # into the bar after it, which is the same stem playing on.
            room = min(room, first + int(stretch))
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
                            if fade_out:
                                # Past this voice's end is not its audio: the
                                # end faded out, so the start fades in.
                                self._mix_fade_in(target[:faded], clip, base + first + phase, phase, fade)
                            else:
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

    def _mix_fade_in(self, target, clip, head_at: int, into: int, length: int) -> None:
        """A loop start (from arrangement frame ``head_at``) rising from
        silence, ``into`` frames into a fade of ``length``."""
        count = len(target)
        head = self._fade_head[:count]
        head.fill(0)
        mix_clip_into(head, clip, head_at)
        head *= (self._fade_ramp[into : into + count] * (LOOP_XFADE / max(1, length)))[:, None]
        target += head

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
        # A song's clock first: its stems all read their place from it, so
        # every stem of one song always shows the same position.
        group_view = {}
        for key, group in groups.items():
            old = previous_groups.get(key)
            old = (old[0], old[1], old[2]) if old is not None else None
            group_view[key] = (group.running, group.cursor, stamp(old, group.running, group.cursor), group.loop)
        view = {}
        for key, voice in self._voices.items():
            old = previous.get(key)
            old = (old[1], old[4], old[6]) if old is not None else None
            if voice.group is None:
                when = stamp(old, voice.playing, voice.cursor)
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
            when = group_view[voice.group][2] if moving else stamp(old, False, cursor)
            view[key] = (
                voice.playing, moving, offset - voice.shift, end - voice.shift,
                cursor,
                clip["start"] - offset + voice.shift, when, group.loop, _own_length(voice, group.loop),
            )
        self._preview_view = view
        states = {key: (COMPLETED, generation) for key, generation in self._completed.items()}
        for key, voice in self._voices.items():
            states[key] = (PLAYING if voice.playing else PAUSED, voice.generation)
        self._state_view = states
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
        latency = max(0.0, min(1.0, latency))
        self._output_delay = latency
        return time.monotonic() + latency

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
        ``(key, clip index, muted, shift seconds, length seconds)`` (and a sixth,
        ``sealed``: the block ends at a split mark, see _Voice): a shift
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
                if len(member) > 5:
                    voice.sealed = bool(member[5])
                # One instance of this block: another voice of the same clip
                # on this clock, covering the same stretch of the file, is
                # the individual voice this member just took over. Voices of
                # other clips, other groups, and other stretches stay.
                self._retire_overlapping(voice)
            self._drop_empty_groups()
            self._publish_previews()
        self._ensure_stream()

    def _retire_overlapping(self, voice) -> None:
        """Drop voices of ``voice``'s clip and group whose own file stretch
        overlaps ``voice`` (lock held). The group clock is left where it is."""
        group = self._groups.get(voice.group)
        span = _explicit_span(voice, group)
        if span is None:
            return
        for key, other in list(self._voices.items()):
            if other is voice or other.group != voice.group or other.clip != voice.clip:
                continue
            other_span = _explicit_span(other, group)
            if other_span is not None and other_span[0] < span[1] and span[0] < other_span[1]:
                self._voices.pop(key, None)
                self._completed.pop(key, None)

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

    def diagnostics(self) -> dict:
        """Audio health: callback durations (mean, p95, p99, max in ms over
        the last 4096 callbacks), deadline misses, PortAudio underruns, the
        load (time spent rendering / time available), voices and any voice
        that duplicates another, and the stream's real rate, block and
        latency. Lock-free reads; the stream fields need an open stream."""
        count = min(self._timed, len(self._timing))
        durations = np.sort(self._timing[:count]) * 1000.0 if count else np.zeros(1)
        budget = BLOCK_SIZE / OUTPUT_RATE * 1000.0
        report = self.voice_report()
        seen: dict = {}
        duplicates = []
        for row in report:
            if row["state"] != PLAYING or row["file_span"] is None:
                continue
            same = (row["clip"], row["group"], round(row["file_span"][0], 6), round(row["file_span"][1], 6))
            if same in seen:
                duplicates.append((seen[same], row["key"]))
            else:
                seen[same] = row["key"]
        stream = self._stream
        return {
            "callbacks": int(self._timed),
            "mean_ms": float(durations.mean()),
            "p95_ms": float(durations[int(0.95 * (len(durations) - 1))]),
            "p99_ms": float(durations[int(0.99 * (len(durations) - 1))]),
            "max_ms": float(durations[-1]),
            "budget_ms": budget,
            "load": float(durations.mean() / budget),
            "deadline_misses": int(self._late),
            "underruns": int(self._underruns),
            "voices": sum(1 for row in report if row["state"] in (PLAYING, PAUSED)),
            "mixed_voices": sum(1 for row in report if row["mixed"]),
            "duplicates": duplicates,
            "sample_rate": float(stream.samplerate) if stream is not None else float(OUTPUT_RATE),
            "block_size": int(BLOCK_SIZE),
            "latency_s": float(stream.latency) if stream is not None else None,
            "requested_latency": output_latency(),
        }

    def voice_report(self) -> list[dict]:
        """Every voice the mixer holds, for diagnostics and for following a
        block after an edit: key, generation, state, whether it is mixed into
        the output now, its clip and group, and ``file_span``: the stretch of
        the stem file it loops, ``(start, length)`` in seconds, for a split stem's
        block voice (None for others).
        A key that is not here is not mixed: STOPPED means silent."""
        with self._lock:
            rows = []
            for key, voice in self._voices.items():
                group = self._groups.get(voice.group) if voice.group is not None else None
                span = None
                if group is not None and group.loop is not None:
                    length = voice.length or (group.loop[1] - group.loop[0])
                    span = ((group.loop[0] + voice.shift) / OUTPUT_RATE, length / OUTPUT_RATE)
                rows.append({
                    "key": key,
                    "generation": voice.generation,
                    "state": PLAYING if voice.playing else PAUSED,
                    "mixed": bool(
                        voice.playing and not voice.muted and 0 <= voice.clip < len(self._clips)
                        and (voice.group is None or group is not None)
                    ),
                    "clip": voice.clip,
                    "group": voice.group,
                    "file_span": span,
                })
            for key, generation in self._completed.items():
                rows.append({"key": key, "generation": generation, "state": COMPLETED, "mixed": False,
                             "clip": None, "group": None, "file_span": None})
            return rows

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
        none plays. Taken from a voice whose own block is sounding: a block
        shorter than its song clock's loop holds its time while the clock is
        past it, so that one would stand still. Lock-free."""
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
                    latency=output_latency(),
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
