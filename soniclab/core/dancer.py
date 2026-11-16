"""Heart-dancer clock and independent accent layers.

The authored frame is a function of the playback position, the BPM and the
downbeats. A display timer may ask for it sixty times a second; it does not
advance the dance, and drum hits never move it. Seeking, looping, pausing
and tempo changes all show up as a new position or a new grid.

Kick, snare, hat, bass and percussion are separate layers. Each one adds an
offset or a short pose accent on top of the current frame. A missing
instrument leaves its layer at rest. Detection runs on a buffer the UI
already captured. It is not called from the audio callback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"
SHEET_PATH = ASSETS / "heart_dance_smooth_sheet.png"
BLOCKS_PATH = ASSETS / "heart_dance_blocks.json"

PHRASE_BEATS = 8
BEATS_PER_BAR = 4
SMOOTH_PER_BEAT = 12
PIXEL_PER_BEAT = 4
SMOOTH_FRAMES = 96
PIXEL_FRAMES = 32
SHEET_COLUMNS = 12
SHEET_ROWS = 8
FRAME_PX = 384
# One cell of the 24-block canvas, in sheet pixels.
BLOCK_PX = 16

# A snare accent this strong shows the key pose of the current sixteenth.
# It does not move the phrase.
SNARE_SNAP = 0.45
# Hits closer than this (seconds) are the same strike.
REFRACTORY = {"kick": 0.070, "snare": 0.080, "hat": 0.040, "perc": 0.045}
# How a hit blooms and fades, in seconds. The phrase clock does not use these.
ENVELOPE = {
    "kick": (0.004, 0.120),
    "snare": (0.003, 0.070),
    "hat": (0.002, 0.035),
    "perc": (0.003, 0.050),
}


@dataclass(frozen=True)
class Sensitivity:
    """Higher means a quieter onset still counts. 1 is the default."""

    amount: float = 1.0

    def scale(self) -> float:
        return max(0.4, min(2.5, float(self.amount)))


@dataclass(frozen=True)
class Hit:
    """One onset, stamped in arrangement seconds (the playback clock)."""

    kind: str
    time: float
    confidence: float
    attack: float = 0.0


@dataclass(frozen=True)
class Layers:
    """Each instrument's accent, 0..1. Missing instruments stay at 0."""

    kick: float = 0.0
    snare: float = 0.0
    hat: float = 0.0
    bass: float = 0.0
    perc: float = 0.0

    @property
    def label(self) -> str:
        names = []
        for name, value in (
            ("kick", self.kick),
            ("snare", self.snare),
            ("hat", self.hat),
            ("bass", self.bass),
            ("perc", self.perc),
        ):
            if value >= 0.35:
                names.append(name)
        return ", ".join(names)


@dataclass(frozen=True)
class Pose:
    """Authored frame for this beat, plus accent offsets that do not move it."""

    frame: int
    beat: float
    on_downbeat: bool
    mode: str = "smooth"
    layers: Layers = Layers()
    drop_blocks: float = 0.0
    sway_blocks: float = 0.0
    hat_blocks: float = 0.0

    @property
    def draw_frame(self) -> int:
        """What to paint. A snare shows the key of this same sixteenth."""
        if self.mode == "smooth" and self.layers.snare >= SNARE_SNAP:
            return sheet_frame(self.beat, "pixel")
        return self.frame


def load_blocks(path: Path | None = None) -> dict:
    path = BLOCKS_PATH if path is None else Path(path)
    text = path.read_text(encoding="utf-8")
    data, _end = json.JSONDecoder().raw_decode(text.lstrip())
    return data


def fold_loop(seconds: float, loop: tuple[float, float] | None) -> float:
    """The arrangement position the engine plays, after a loop wrap."""
    seconds = float(seconds)
    if loop is None:
        return max(0.0, seconds)
    start, end = float(loop[0]), float(loop[1])
    span = end - start
    if span <= 1e-9:
        return max(0.0, seconds)
    if seconds < start:
        return seconds
    if seconds >= end:
        return start + (seconds - start) % span
    return seconds


def _bar_length(downbeats: tuple[float, ...], index: int, bpm: float) -> float:
    if 0 <= index + 1 < len(downbeats):
        gap = float(downbeats[index + 1]) - float(downbeats[index])
        if gap > 1e-4:
            return gap
    if index > 0:
        gap = float(downbeats[index]) - float(downbeats[index - 1])
        if gap > 1e-4:
            return gap
    return BEATS_PER_BAR * 60.0 / max(float(bpm), 1.0)


def phrase_beat(
    seconds: float,
    bpm: float,
    downbeats: tuple[float, ...] | list[float] | None = None,
    loop: tuple[float, float] | None = None,
) -> float:
    """Beats into the 2-bar phrase, in ``[0, 8)``.

    Each detected downbeat starts a bar. Even downbeats start bar 1 of the
    phrase (the jack), odd ones start bar 2 (the hop). The beat inside the
    bar follows the gap to the next downbeat, so a tempo change in the
    analysis moves the dance. With no downbeats the project BPM is the grid,
    still measured from the playback position.
    """
    seconds = fold_loop(seconds, loop)
    marks = tuple(float(item) for item in (downbeats or ()))
    if not marks:
        return (seconds * max(float(bpm), 1.0) / 60.0) % PHRASE_BEATS
    # Latest downbeat at or before the playhead. Before the first one, step
    # back by whole bars so the phrase is already running when the first
    # downbeat arrives.
    index = _latest_at_or_before(marks, seconds)
    if index < 0:
        gap = _bar_length(marks, 0, bpm)
        bars_back = int((marks[0] - seconds) / gap) + 1
        index = -bars_back
        origin = marks[0] - bars_back * gap
        bar_len = gap
    else:
        origin = marks[index]
        bar_len = _bar_length(marks, index, bpm)
        typical = _typical_bar(marks, bpm)
        if index + 1 < len(marks) and bar_len > 1.6 * typical:
            # Downbeats were missed here (a break, or the drum track is cut):
            # the gap holds several bars, not one slow one.
            bar_len /= max(1, round(bar_len / typical))
        elif index + 1 >= len(marks):
            bar_len = typical  # past the last downbeat: bars go on
    # Bars keep counting past ``bar_len`` until the next downbeat, so the
    # dance never stops between downbeats that are far apart.
    elapsed = max(0.0, (seconds - origin) / bar_len)
    bars = int(elapsed)
    into = min(0.9999, elapsed - bars)
    bar_in_phrase = (index + bars) % 2
    if bar_in_phrase < 0:
        bar_in_phrase += 2
    return (bar_in_phrase * BEATS_PER_BAR + into * BEATS_PER_BAR) % PHRASE_BEATS


def _typical_bar(marks: tuple[float, ...], bpm: float) -> float:
    """The usual gap between downbeats (their median), or a bar at ``bpm``."""
    gaps = [b - a for a, b in zip(marks, marks[1:]) if b - a > 1e-4]
    if not gaps:
        return BEATS_PER_BAR * 60.0 / max(float(bpm), 1.0)
    gaps.sort()
    return gaps[len(gaps) // 2]


def _latest_at_or_before(marks: tuple[float, ...], seconds: float) -> int:
    lo, hi = 0, len(marks)
    while lo < hi:
        mid = (lo + hi) // 2
        if marks[mid] <= seconds:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def sheet_frame(beat: float, mode: str) -> int:
    """Index into the 96-frame sheet. Pixel mode keeps the sixteenth keys.

    Drum hits never enter here. Two songs at the same BPM share this frame.
    """
    beat = float(beat) % PHRASE_BEATS
    if mode == "pixel":
        step = int(beat * PIXEL_PER_BEAT) % PIXEL_FRAMES
        return (step * (SMOOTH_PER_BEAT // PIXEL_PER_BEAT)) % SMOOTH_FRAMES
    return int(beat * SMOOTH_PER_BEAT) % SMOOTH_FRAMES


def _envelope(now: float, hit_time: float, kind: str) -> float:
    attack, release = ENVELOPE[kind]
    elapsed = float(now) - float(hit_time)
    if elapsed < 0.0:
        return 0.0  # not out of the speakers yet
    if elapsed < attack:
        return elapsed / attack
    return float(np.exp(-(elapsed - attack) / release))


def layer_levels(seconds: float, hits=(), bass: float = 0.0) -> Layers:
    """Accent strengths at a playback position. The phrase is not an input."""
    best = {"kick": 0.0, "snare": 0.0, "hat": 0.0, "perc": 0.0}
    for hit in hits:
        if hit.kind not in best:
            continue
        level = _envelope(seconds, hit.time, hit.kind) * max(0.0, min(1.0, hit.confidence))
        if level > best[hit.kind]:
            best[hit.kind] = level
    return Layers(
        kick=best["kick"],
        snare=best["snare"],
        hat=best["hat"],
        bass=max(0.0, min(1.0, float(bass))),
        perc=best["perc"],
    )


def pose_at(
    seconds: float,
    bpm: float,
    downbeats=None,
    loop=None,
    mode: str = "smooth",
    hits=(),
    bass: float = 0.0,
) -> Pose:
    """Authored pose at this playback position. Hits only add accents."""
    beat = phrase_beat(seconds, bpm, downbeats, loop)
    on_downbeat = beat < 0.04 or abs(beat - 4.0) < 0.04
    frame = sheet_frame(beat, "pixel" if mode != "smooth" else "smooth")
    layers = layer_levels(fold_loop(seconds, loop), hits, bass)
    # Strong hits win, and the offsets stay small so they cannot pile into a jump.
    drop = min(1.15, layers.kick * 1.1 + layers.perc * 0.12)
    sway = max(-0.4, min(0.4, layers.bass * 0.4))
    hat_blocks = min(0.2, layers.hat * 0.2)
    return Pose(frame, beat, on_downbeat, mode, layers, drop, sway, hat_blocks)


def _mono(audio) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _bands(chunk: np.ndarray, rate: int) -> dict[str, float]:
    count = int(chunk.size)
    if count < 32:
        return {"low": 0.0, "body": 0.0, "mid": 0.0, "high": 0.0}
    # No edge window: an attack at the end of the buffer must still count.
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(count, 1.0 / float(rate))

    def energy(lo: float, hi: float) -> float:
        chosen = (freqs >= lo) & (freqs < hi)
        if not np.any(chosen):
            return 0.0
        return float(np.mean(spectrum[chosen]))

    return {
        "low": energy(30.0, 140.0),
        "body": energy(40.0, 220.0),
        "mid": energy(250.0, 2800.0),
        "high": energy(5500.0, 14000.0),
    }


def _classify(low: float, mid: float, high: float, sensitivity: float) -> tuple[str, float] | None:
    """Band shape of one onset. A steady tone never reaches this.

    A low-only attack can be a kick. A high-only attack can be a hat.
    Energy in the low band, or a high-band spike with no attack test,
    is not enough on its own: the caller already required an RMS jump.
    Broadband mid-and-high is a snare, not a hat. Anything else that
    still jumped is percussion.
    """
    total = low + mid + high
    if total <= 1e-8:
        return None
    low_s, mid_s, high_s = low / total, mid / total, high / total
    sense = max(0.4, min(2.5, float(sensitivity)))
    kick_need = 0.60 / min(sense, 1.5)
    hat_need = 0.48 / min(sense, 1.5)
    if low_s >= kick_need and high_s < 0.18 and low_s > high_s * 3.0:
        return "kick", low_s
    if mid_s >= 0.18 and high_s >= 0.18 and low_s < 0.50:
        return "snare", min(1.0, mid_s + high_s)
    if high_s >= hat_need and low_s < 0.22 and mid_s < high_s:
        return "hat", high_s
    if low_s < 0.85:
        return "perc", max(mid_s, high_s, 0.3)
    return None


def detect_hits(
    audio,
    rate: int = 44100,
    origin: float = 0.0,
    sensitivity: float = 1.0,
) -> list[Hit]:
    """Onsets in this buffer, timed from ``origin`` (arrangement seconds).

    ``origin`` is the playback position of the first sample. The UI passes
    the render head minus the buffer length, then animates against the
    heard position, so a hit waits out the output latency. This is not
    called from the audio callback.
    """
    samples = _mono(audio)
    rate = int(rate)
    if rate <= 0 or samples.size < 1024:
        return []
    hop = 256
    window = 512
    sense = max(0.4, min(2.5, float(sensitivity)))
    onset_ratio = 1.85 / sense
    floor = 0.025 / sense
    hits: list[Hit] = []
    last = {kind: -1e9 for kind in REFRACTORY}
    previous = None
    previous_rms = 0.0
    for start in range(0, samples.size - window, hop):
        chunk = samples[start : start + window]
        bands = _bands(chunk, rate)
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        if previous is None:
            previous = bands
            previous_rms = rms
            continue
        jumped = rms >= floor and rms > previous_rms * onset_ratio
        previous_rms = rms
        if not jumped:
            previous = bands
            continue
        low = max(0.0, bands["low"] - previous["low"])
        mid = max(0.0, bands["mid"] - previous["mid"])
        high = max(0.0, bands["high"] - previous["high"])
        previous = bands
        found = _classify(low, mid, high, sense)
        if found is None:
            continue
        kind, confidence = found
        when = float(origin) + (start + window) / rate
        if when < 0.0 or when - last[kind] < REFRACTORY[kind]:
            continue
        # One strike, one layer. The tail of a snare is not a second drum.
        if kind == "perc" and any(when - last[name] < 0.030 for name in ("kick", "snare", "hat")):
            continue
        # A low note that is still loud a tenth of a second later is bass,
        # not a kick. A short buffer has no later audio, so the hit stands
        # and the animation is not held back.
        if kind == "kick" and _low_note_holds(samples, start + window, rate, rms):
            continue
        last[kind] = when
        hits.append(Hit(kind, when, confidence, rms))
    return hits


def _low_note_holds(samples: np.ndarray, onset_end: int, rate: int, attack_rms: float) -> bool:
    """True when the low attack is still going after the strike would have died."""
    begin = int(onset_end) + int(0.10 * rate)
    end = begin + int(0.04 * rate)
    if end > int(samples.size):
        return False
    tail = samples[begin:end]
    tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
    return tail_rms > max(0.045, float(attack_rms) * 0.50)


def sustain_bass(audio, rate: int = 44100) -> float:
    """Low energy that is held, not an attack. Silence and kicks score low."""
    samples = _mono(audio)
    if samples.size < 1024 or int(rate) <= 0:
        return 0.0
    if samples.size > 4096:
        samples = samples[-4096:]
    half = samples.size // 2
    first = float(np.sqrt(np.mean(samples[:half] * samples[:half]) + 1e-12))
    second = float(np.sqrt(np.mean(samples[half:] * samples[half:]) + 1e-12))
    if first < 0.02 or second < 0.02 or max(first, second) > min(first, second) * 1.8:
        return 0.0
    bands = _bands(samples, int(rate))
    other = bands["mid"] + bands["high"] + 1e-8
    if bands["body"] < other * 1.4:
        return 0.0
    return min(1.0, bands["body"] / (bands["body"] + other))


class HitDeck:
    """Hits accepted so far, with duplicates across buffers removed.

    A seek backwards clears it. Pause does not: the playhead stays put, so
    the accents stay with it.
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = float(sensitivity)
        self.hits: list[Hit] = []
        self.bass = 0.0
        self._last = {kind: -1e9 for kind in REFRACTORY}

    def reset(self) -> None:
        self.hits.clear()
        self.bass = 0.0
        self._last = {kind: -1e9 for kind in REFRACTORY}

    def update(self, audio, rate: int, origin: float) -> list[Hit]:
        found = detect_hits(audio, rate, origin, self.sensitivity)
        accepted = []
        for hit in found:
            if hit.time - self._last.get(hit.kind, -1e9) < REFRACTORY[hit.kind]:
                continue
            self._last[hit.kind] = hit.time
            accepted.append(hit)
        if accepted:
            self.hits.extend(accepted)
            horizon = max(hit.time for hit in self.hits) - 1.5
            self.hits = [hit for hit in self.hits if hit.time >= horizon]
        self._drop_kicks_that_ring(_mono(audio), int(rate), float(origin))
        self.bass = sustain_bass(audio, rate)
        return accepted

    def _drop_kicks_that_ring(self, samples: np.ndarray, rate: int, origin: float) -> None:
        """A low attack that is still loud later was a bass note, not a kick.

        The live buffer is too short to know this when the attack arrives, so
        the drop starts on time and is taken back once the note keeps ringing.
        """
        if not self.hits or rate <= 0:
            return
        end = origin + samples.size / rate
        kept = []
        for hit in self.hits:
            if hit.kind != "kick" or hit.attack <= 0.0 or end < hit.time + 0.12:
                kept.append(hit)
                continue
            index = int(round((hit.time + 0.12 - origin) * rate))
            if index < 0 or index + 256 > samples.size:
                kept.append(hit)
                continue
            tail = samples[index : index + 256]
            tail_rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
            if tail_rms > max(0.05, hit.attack * 0.45):
                continue
            kept.append(hit)
        self.hits = kept
