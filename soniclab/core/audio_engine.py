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


class _Voice:
    """One stem block played on its own, apart from the transport."""

    __slots__ = ("key", "clip", "start", "end", "cursor", "playing")

    def __init__(self, key, clip: int, start: int, end: int):
        self.key = key
        self.clip = int(clip)
        self.start = int(start)
        self.end = int(end)
        self.cursor = int(start)
        self.playing = True

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
        is already in memory; nothing here decodes or touches the disk."""
        clips = self._clips
        finished = []
        for key, voice in self._voices.items():
            if not voice.playing:
                continue
            if not (0 <= voice.clip < len(clips)):
                finished.append(key)
                continue
            count = min(frames, voice.end - voice.cursor)
            if count > 0:
                peak = mix_clip_into(outdata[:count], clips[voice.clip], voice.cursor)
                if voice.clip < len(self._track_peaks) and peak > self._track_peaks[voice.clip]:
                    self._track_peaks[voice.clip] = peak
                voice.cursor += count
            if voice.cursor >= voice.end:
                finished.append(key)
        for key in finished:
            del self._voices[key]
        self._publish_previews(sounds_at)

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
        when = time.monotonic() if sounds_at is None else sounds_at
        self._preview_view = {
            key: (voice.playing, voice.start, voice.end, voice.cursor, when)
            for key, voice in self._voices.items()
        }

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
                    self._voices[key] = _Voice(key, clip, first, last)
            self._publish_previews()
        self._ensure_stream()

    def preview_pause(self, key) -> None:
        with self._lock:
            voice = self._voices.get(key)
            if voice is not None:
                voice.playing = False
            self._publish_previews()

    def preview_stop(self, key) -> None:
        with self._lock:
            self._voices.pop(key, None)
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
            self._publish_previews()

    def previews(self, now: float | None = None) -> dict:
        """``key -> (playing, seconds)`` for every preview voice.

        Lock-free, like ``heard_position``: the seconds are what is heard at
        ``now``, interpolated between audio blocks and inside the block.
        A finished or stopped voice is absent.
        """
        view = self._preview_view
        if not view:
            return {}
        now = time.monotonic() if now is None else float(now)
        shown = {}
        for key, (playing, start, end, cursor, sounds_at) in view.items():
            frame = cursor
            if playing:
                # ``cursor`` is the end of the last block written; that block
                # starts sounding at ``sounds_at``.
                frame = cursor - BLOCK_SIZE + (now - sounds_at) * OUTPUT_RATE
            shown[key] = (playing, min(end, max(start, frame)) / OUTPUT_RATE)
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
