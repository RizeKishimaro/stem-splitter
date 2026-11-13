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
        audio = clip["audio"]
        offset = int(clip.get("offset", 0))
        length = int(clip.get("length", max(0, len(audio) - offset)))
        gain = float(clip["gain"]) * float(clip.get("clip_gain", 1.0))
        pan = clip["pan"]
        left_gain = gain * (1.0 - max(0.0, pan))
        right_gain = gain * (1.0 + min(0.0, pan))
        fade_in = int(clip.get("fade_in", 0))
        fade_out = int(clip.get("fade_out", 0))
        file_start = offset + start - clip["start"]
        overlap_start = max(offset, file_start)
        overlap_end = min(len(audio), offset + max(0, length), file_start + frames)
        if overlap_end <= overlap_start:
            continue
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

    def clear_clips(self) -> None:
        """Remove every clip. Transport position and loop are unchanged."""
        self.replace_clips([])

    def replace_clips(self, clips):
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
            if not self._playing:
                return
            frames = int(frames)
            if frames <= 0:
                return
            position = self._position
            clips = list(self._clips)
            soloed = any(clip.get("solo") for clip in clips)
            capture = self._capture_enabled and frames <= len(self._scratch)
            segments, next_position = self._playback_segments(position, frames)
            self._heard = (self._heard + ((self._sounds_at(time_info), tuple(segments)),))[-HEARD_BLOCKS:]
            for index, clip in enumerate(clips):
                heard = not clip["muted"] and not (soloed and not clip.get("solo"))
                if not heard:
                    continue
                audio = clip["audio"]
                offset = int(clip.get("offset", 0))
                length = int(clip.get("length", max(0, len(audio) - offset)))
                gain = float(clip["gain"]) * float(clip.get("clip_gain", 1.0))
                pan = clip["pan"]
                left_gain = gain * (1.0 - max(0.0, pan))
                right_gain = gain * (1.0 + min(0.0, pan))
                fade_in = int(clip.get("fade_in", 0))
                fade_out = int(clip.get("fade_out", 0))
                play_end = offset + max(0, length)
                scratch = None
                if capture and index < len(self._track_rings):
                    scratch = self._scratch[:frames]
                    scratch.fill(0)
                peak = 0.0
                filled = 0
                for seg_position, span in segments:
                    source_start = seg_position - clip["start"]
                    file_start = offset + source_start
                    overlap_start = max(offset, file_start)
                    overlap_end = min(len(audio), play_end, file_start + span)
                    if overlap_end > overlap_start:
                        dest_start = filled + (overlap_start - file_start)
                        dest_end = dest_start + (overlap_end - overlap_start)
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
                        outdata[dest_start:dest_end, 0] += piece_l
                        outdata[dest_start:dest_end, 1] += piece_r
                        if scratch is not None:
                            scratch[dest_start:dest_end, 0] = piece_l
                            scratch[dest_start:dest_end, 1] = piece_r
                        piece_peak = float(max(np.max(np.abs(piece_l)), np.max(np.abs(piece_r))))
                        if piece_peak > peak:
                            peak = piece_peak
                    filled += span
                if index < len(self._track_peaks) and peak > self._track_peaks[index]:
                    self._track_peaks[index] = peak
                if scratch is not None:
                    self._track_rings[index].push(scratch)
            if self._output_gain != 1.0:
                outdata *= self._output_gain
            if self._master_muted:
                outdata.fill(0)
            np.clip(outdata, -1.0, 1.0, out=outdata)
            if frames:
                master_peak = float(np.max(np.abs(outdata)))
                if master_peak > self._master_peak:
                    self._master_peak = master_peak
            if self._capture_enabled:
                self._master_ring.push(outdata[:frames])
            self._position = next_position

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

    def close(self):
        with self._lock:
            self._playing = False
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
