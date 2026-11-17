"""The session's musical timeline: the one authority for bars and tempo.

Musical positions are integer ticks (``ppq`` ticks per quarter note, 960 by
default). Bars are explicit intervals in ticks with their own meter; their
numbers come from their order, never from a stored index. Seconds are
derived from ticks through a piecewise tempo map, so a tempo change never
moves a bar boundary; it moves where that boundary falls in time.

Imported audio does not define bars. Detected beats live in the source
file's sample frames (``DetectedBeat``); ``soniclab.core.audio_alignment``
compares them with this grid and suggests offsets. Nothing here reads audio.

Past the last stored bar the grid carries on with whole bars of the meter in
force there, so the ruler never runs out. With no stored bars at all, the
grid is regular from tick 0.
"""

from __future__ import annotations

import bisect
import uuid
from dataclasses import asdict, dataclass, field

PPQ = 960
DEFAULT_BPM = 120.0
FORMAT = "soniclab.session-timeline"
VERSION = 1

# Where a bar came from. Only "grid" bars are made by the timeline itself.
ORIGIN_GRID = "grid"
ORIGIN_MANUAL = "manual"  # a boundary the user placed, moved or removed
ORIGIN_PICKUP = "pickup"  # an intentional partial bar before bar 1 (numbered 0)
ORIGIN_MIGRATED = "migrated"  # converted from an old project's bar lines
ORIGIN_MIGRATED_DOWNBEATS = "migrated-downbeats"  # an old project's bars that followed the drums
ORIGIN_DOWNBEATS = "downbeats"  # applied from detected downbeats, by a user command


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Session:
    id: str = field(default_factory=_new_id)
    ppq: int = PPQ
    bpm: float = DEFAULT_BPM  # the tempo before the first tempo event
    numerator: int = 4  # the meter before the first meter event
    denominator: int = 4


@dataclass
class Bar:
    id: str
    start_tick: int
    end_tick: int
    numerator: int
    denominator: int
    origin: str = ORIGIN_GRID

    @property
    def length(self) -> int:
        return self.end_tick - self.start_tick


@dataclass
class TempoEvent:
    id: str
    tick: int
    bpm: float


@dataclass
class MeterEvent:
    id: str
    tick: int
    numerator: int
    denominator: int


@dataclass
class DetectedBeat:
    """A beat found in audio, in the source file's sample frames. Analysis
    data: it is compared with the grid, never written into it."""

    id: str
    frame: int
    confidence: float = 1.0
    downbeat: bool = False


@dataclass(frozen=True)
class Position:
    """A musical position: bar number (bar 1 is the first full bar; a
    pickup is bar 0), beat from 1, and ticks into that beat."""

    bar: int
    beat: int
    tick: int


class SessionTimeline:
    """Bars, tempo and meter for one session, and every conversion between
    ticks, seconds and musical positions."""

    def __init__(self, session: Session | None = None, tempos=(), meters=(), bars=()):
        self.session = session or Session()
        self.tempos: list[TempoEvent] = sorted(tempos, key=lambda event: event.tick)
        self.meters: list[MeterEvent] = sorted(meters, key=lambda event: event.tick)
        self.bars: list[Bar] = sorted(bars, key=lambda bar: bar.start_tick)
        self._segments = None  # (ticks, seconds, bpm) per tempo segment, rebuilt on change

    # ---------------------------------------------------------------- tempo

    @property
    def ppq(self) -> int:
        return int(self.session.ppq)

    def _tempo_segments(self):
        if self._segments is None:
            ticks = [0]
            bpms = [float(self.session.bpm)]
            for event in self.tempos:
                if event.tick <= 0:
                    bpms[0] = float(event.bpm)  # an event at 0 replaces the default
                    continue
                if event.tick == ticks[-1]:
                    bpms[-1] = float(event.bpm)
                    continue
                ticks.append(int(event.tick))
                bpms.append(float(event.bpm))
            seconds = [0.0]
            for index in range(1, len(ticks)):
                seconds.append(seconds[-1] + self._span_seconds(ticks[index] - ticks[index - 1], bpms[index - 1]))
            self._segments = (ticks, seconds, bpms)
        return self._segments

    def _span_seconds(self, ticks: float, bpm: float) -> float:
        return float(ticks) * 60.0 / (max(1e-6, float(bpm)) * self.ppq)

    def _changed(self) -> None:
        self._segments = None

    def tempo_at(self, tick: float) -> float:
        ticks, _seconds, bpms = self._tempo_segments()
        return bpms[max(0, bisect.bisect_right(ticks, tick) - 1)]

    def tick_to_seconds(self, tick: float) -> float:
        """Seconds at ``tick``, integrating every tempo segment crossed.
        Before tick 0 the first tempo carries on backwards."""
        ticks, seconds, bpms = self._tempo_segments()
        index = max(0, bisect.bisect_right(ticks, tick) - 1)
        return seconds[index] + self._span_seconds(float(tick) - ticks[index], bpms[index])

    def seconds_to_tick(self, seconds: float) -> float:
        """The exact inverse of ``tick_to_seconds`` (a float; round it to
        store it)."""
        ticks, starts, bpms = self._tempo_segments()
        index = max(0, bisect.bisect_right(starts, seconds) - 1)
        return ticks[index] + (float(seconds) - starts[index]) * bpms[index] * self.ppq / 60.0

    def set_tempo(self, tick: int, bpm: float) -> TempoEvent:
        """A tempo from ``tick`` on. Bars keep their ticks."""
        tick = max(0, int(tick))
        for event in self.tempos:
            if event.tick == tick:
                event.bpm = float(bpm)
                self._changed()
                return event
        event = TempoEvent(_new_id(), tick, float(bpm))
        self.tempos.append(event)
        self.tempos.sort(key=lambda item: item.tick)
        self._changed()
        return event

    def scale_tempo(self, factor: float) -> None:
        """Every tempo times ``factor`` (the audio was stretched by 1/factor).
        Bars stay on the same ticks, so they follow the stretched audio."""
        factor = float(factor)
        self.session.bpm *= factor
        for event in self.tempos:
            event.bpm *= factor
        self._changed()

    def reset_tempo(self, bpm: float) -> None:
        """One tempo for the whole session. Bars keep their ticks."""
        self.session.bpm = float(bpm)
        self.tempos.clear()
        self._changed()

    # ---------------------------------------------------------------- meter

    def meter_at(self, tick: float) -> tuple[int, int]:
        meter = (int(self.session.numerator), int(self.session.denominator))
        for event in self.meters:
            if event.tick > tick:
                break
            meter = (int(event.numerator), int(event.denominator))
        return meter

    def set_meter(self, tick: int, numerator: int, denominator: int) -> MeterEvent:
        """A meter from ``tick`` on (normally a bar boundary). Stored bars
        keep their boundaries and their own meter; only bars the grid makes
        past them, and the default before any event, follow it."""
        tick = max(0, int(tick))
        if tick == 0:
            self.session.numerator, self.session.denominator = int(numerator), int(denominator)
        for event in self.meters:
            if event.tick == tick:
                event.numerator, event.denominator = int(numerator), int(denominator)
                return event
        event = MeterEvent(_new_id(), tick, int(numerator), int(denominator))
        self.meters.append(event)
        self.meters.sort(key=lambda item: item.tick)
        return event

    def bar_ticks(self, numerator: int, denominator: int) -> int:
        """Ticks in one whole bar of this meter."""
        return int(round(int(numerator) * self.ppq * 4 / max(1, int(denominator))))

    def beat_ticks(self, denominator: int) -> int:
        return int(round(self.ppq * 4 / max(1, int(denominator))))

    # ----------------------------------------------------------------- bars

    def _generated(self, start_tick: int, until_tick: float) -> list[Bar]:
        """Whole bars of the meter in force, from ``start_tick`` through
        ``until_tick``; made on the fly, not stored."""
        bars = []
        tick = int(start_tick)
        while tick <= until_tick or not bars:
            numerator, denominator = self.meter_at(tick)
            length = max(1, self.bar_ticks(numerator, denominator))
            following = next((event.tick for event in self.meters if event.tick > tick), None)
            end = tick + length
            if following is not None and end > following:
                end = following  # a meter change cuts the bar before it
            bars.append(Bar(f"grid:{tick}", tick, end, numerator, denominator, ORIGIN_GRID))
            tick = end
        return bars

    def bars_until(self, until_tick: float) -> list[Bar]:
        """Every bar from the first through the one holding ``until_tick``:
        the stored bars, then the grid's carrying on past them."""
        stored = list(self.bars)
        if not stored:
            return self._generated(0, until_tick)
        last = stored[-1]
        if last.end_tick > until_tick:
            return stored
        return stored + self._generated(last.end_tick, until_tick)

    def _materialize(self, until_tick: float) -> None:
        """Store the bars through ``until_tick`` so they can be edited."""
        self.bars = [
            Bar(_new_id(), bar.start_tick, bar.end_tick, bar.numerator, bar.denominator, bar.origin)
            if bar.id.startswith("grid:") else bar
            for bar in self.bars_until(until_tick)
        ]

    def _first_numbered(self, bars) -> int:
        """How many pickup bars come before bar 1."""
        count = 0
        for bar in bars:
            if bar.origin != ORIGIN_PICKUP:
                break
            count += 1
        return count

    def bar_number(self, index: int) -> int:
        """The displayed number of the bar at ``index`` (pickups are 0)."""
        pickups = self._first_numbered(self.bars)
        return 0 if index < pickups else index - pickups + 1

    def bar_at_tick(self, tick: float) -> tuple[int, Bar]:
        """``(index, bar)`` of the bar holding ``tick``; before the first
        bar, the first one."""
        bars = self.bars_until(max(0.0, tick))
        starts = [bar.start_tick for bar in bars]
        index = max(0, bisect.bisect_right(starts, tick) - 1)
        return index, bars[index]

    def bar_at_seconds(self, seconds: float) -> tuple[int, Bar]:
        return self.bar_at_tick(self.seconds_to_tick(seconds))

    def tick_to_position(self, tick: float) -> Position:
        index, bar = self.bar_at_tick(tick)
        into = max(0.0, float(tick) - bar.start_tick)
        beat = self.beat_ticks(bar.denominator)
        return Position(self.bar_number(index), int(into // beat) + 1, int(round(into % beat)))

    def position_to_tick(self, bar: int, beat: int = 1, tick: int = 0) -> int:
        """The tick of a musical position (bar numbers as shown)."""
        pickups = self._first_numbered(self.bars)
        index = max(0, int(bar) - 1 + pickups) if int(bar) > 0 else 0
        bars = self.bars
        while len(bars) <= index:
            bars = self.bars_until((bars[-1].end_tick if bars else 0) + self.bar_ticks(*self.meter_at(0)) * (index + 1))
        found = bars[index]
        return found.start_tick + (max(1, int(beat)) - 1) * self.beat_ticks(found.denominator) + int(tick)

    def get_bar_lines(self, until_seconds: float) -> list[float]:
        """Bar starts in seconds through ``until_seconds`` (and one past)."""
        until = self.seconds_to_tick(until_seconds)
        bars = self.bars_until(until)
        lines = [self.tick_to_seconds(bar.start_tick) for bar in bars]
        lines.append(self.tick_to_seconds(bars[-1].end_tick))
        return lines

    def get_beat_lines(self, until_seconds: float) -> list[tuple[float, bool]]:
        """``(seconds, is_bar_start)`` for every beat through
        ``until_seconds``. A bar shorter than its meter ends on a short beat."""
        until = self.seconds_to_tick(until_seconds)
        marks = []
        for bar in self.bars_until(until):
            step = self.beat_ticks(bar.denominator)
            tick = bar.start_tick
            while tick < bar.end_tick:
                marks.append((self.tick_to_seconds(tick), tick == bar.start_tick))
                tick += step
        return marks

    def snap_tick(self, tick: float, division: str | int = "beat") -> int:
        """The nearest bar line (``"bar"``), beat (``"beat"``) or multiple of
        ``division`` ticks, counted from the bar holding ``tick``."""
        _index, bar = self.bar_at_tick(tick)
        if division == "bar":
            return bar.start_tick if tick - bar.start_tick < bar.end_tick - tick else bar.end_tick
        step = self.beat_ticks(bar.denominator) if division == "beat" else max(1, int(division))
        snapped = bar.start_tick + round((float(tick) - bar.start_tick) / step) * step
        return int(min(max(snapped, bar.start_tick), bar.end_tick))

    # ---------------------------------------------------------------- edits

    def insert_bar(self, tick: int) -> int:
        """A new boundary at ``tick``: the bar holding it becomes two.
        Returns the new bar's index."""
        tick = int(tick)
        self._materialize(tick)
        index, bar = self.bar_at_tick(tick)
        if tick <= bar.start_tick or tick >= bar.end_tick:
            return index
        second = Bar(_new_id(), tick, bar.end_tick, bar.numerator, bar.denominator, ORIGIN_MANUAL)
        bar.end_tick = tick
        bar.origin = ORIGIN_MANUAL if bar.origin == ORIGIN_GRID else bar.origin
        self.bars.insert(index + 1, second)
        return index + 1

    def remove_bar(self, index: int) -> None:
        """Remove the boundary at the start of bar ``index``: it joins the
        bar before it. The first bar cannot be removed."""
        self._materialize(0)
        if not (0 < index < len(self.bars)):
            return
        gone = self.bars.pop(index)
        before = self.bars[index - 1]
        before.end_tick = gone.end_tick
        before.origin = ORIGIN_MANUAL

    def move_boundary(self, index: int, tick: int, ripple: bool = False) -> None:
        """Move the start of bar ``index`` (the end of the one before) to
        ``tick``. Without ``ripple`` it stays between its neighbours' other
        boundaries; with it, every later bar moves by the same ticks."""
        self._materialize(0)
        if not (0 <= index < len(self.bars)):
            return
        bar = self.bars[index]
        tick = int(tick)
        if ripple:
            shift = tick - bar.start_tick
            if index > 0:
                shift = max(shift, self.bars[index - 1].start_tick + 1 - bar.start_tick)
            else:
                shift = max(shift, -bar.start_tick)
            for later in self.bars[index:]:
                later.start_tick += shift
                later.end_tick += shift
            if index > 0:
                self.bars[index - 1].end_tick = bar.start_tick
                self.bars[index - 1].origin = ORIGIN_MANUAL
            bar.origin = ORIGIN_MANUAL
            return
        low = self.bars[index - 1].start_tick + 1 if index > 0 else 0
        tick = min(max(tick, low), bar.end_tick - 1)
        bar.start_tick = tick
        bar.origin = ORIGIN_MANUAL
        if index > 0:
            self.bars[index - 1].end_tick = tick
            self.bars[index - 1].origin = ORIGIN_MANUAL

    def clear_bars(self) -> None:
        """Back to the regular grid from the meter."""
        self.bars.clear()

    def set_bar_lines_seconds(self, lines, origin: str = ORIGIN_MANUAL) -> None:
        """Bar boundaries from bar-line times (the ruler's edits), through
        the current tempo map. Empty clears to the regular grid."""
        ticks = sorted({int(round(self.seconds_to_tick(max(0.0, float(line))))) for line in lines or ()})
        self.bars = self._bars_between(ticks, origin)

    def _bars_between(self, ticks, origin: str) -> list[Bar]:
        bars = []
        if len(ticks) < 2:
            return bars
        if ticks[0] > 0:
            numerator, denominator = self.meter_at(0)
            bars.append(Bar(_new_id(), 0, ticks[0], numerator, denominator, ORIGIN_PICKUP))
        for start, end in zip(ticks, ticks[1:]):
            numerator, denominator = self.meter_at(start)
            bars.append(Bar(_new_id(), start, end, numerator, denominator, origin))
        return bars

    def legacy_lines(self) -> list[float]:
        """The stored bars as old-style bar-line seconds (bar 1 on), for the
        code not yet on this service and for older SonicLab builds. Empty
        when the grid is regular."""
        bars = [bar for bar in self.bars if bar.origin != ORIGIN_PICKUP]
        if not bars:
            return []
        lines = [round(self.tick_to_seconds(bar.start_tick), 6) for bar in bars]
        lines.append(round(self.tick_to_seconds(bars[-1].end_tick), 6))
        return lines

    # ------------------------------------------------------------ migration

    @classmethod
    def from_legacy_lines(
        cls, lines, bpm: float, numerator: int, denominator: int, origin: str = ORIGIN_MIGRATED, ppq: int = PPQ
    ) -> SessionTimeline:
        """An old project's bar lines (arrangement seconds) as bars in ticks,
        keeping every boundary on the same second.

        Each bar gets its meter's length in ticks, and a tempo event at its
        start makes it last exactly as long as it did; irregular bars become
        tempo changes, not odd bars, and no line moves. A lead-in before the
        first line is a pickup bar (numbered 0) at the first bar's tempo.
        With fewer than two lines the grid is regular at ``bpm``."""
        session = Session(ppq=int(ppq), bpm=float(bpm), numerator=int(numerator), denominator=int(denominator))
        timeline = cls(session)
        clean = sorted({round(max(0.0, float(line)), 9) for line in lines or ()})
        if len(clean) < 2:
            return timeline
        whole = timeline.bar_ticks(numerator, denominator)
        quarters_per_bar = whole / ppq
        tempos = []
        bars = []
        first_bpm = 60.0 * quarters_per_bar / max(1e-6, clean[1] - clean[0])
        start = 0
        if clean[0] > 1e-9:
            pickup = int(round(clean[0] * first_bpm * ppq / 60.0))
            if pickup > 0:
                bars.append(Bar(_new_id(), 0, pickup, numerator, denominator, ORIGIN_PICKUP))
                tempos.append(TempoEvent(_new_id(), 0, 60.0 * pickup / ppq / clean[0]))
                start = pickup
        for begin, end in zip(clean, clean[1:]):
            bpm_here = 60.0 * quarters_per_bar / max(1e-6, end - begin)
            if not tempos or abs(tempos[-1].bpm - bpm_here) > 1e-9:
                tempos.append(TempoEvent(_new_id(), start, bpm_here))
            bars.append(Bar(_new_id(), start, start + whole, numerator, denominator, origin))
            start += whole
        timeline.tempos = tempos
        timeline.bars = bars
        timeline._changed()
        return timeline

    # -------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "session": asdict(self.session),
            "tempos": [asdict(event) for event in self.tempos],
            "meters": [asdict(event) for event in self.meters],
            "bars": [asdict(bar) for bar in self.bars],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionTimeline:
        data = data or {}
        raw = data.get("session") or {}
        session = Session(
            id=str(raw.get("id") or _new_id()),
            ppq=int(raw.get("ppq") or PPQ),
            bpm=float(raw.get("bpm") or DEFAULT_BPM),
            numerator=int(raw.get("numerator") or 4),
            denominator=int(raw.get("denominator") or 4),
        )
        tempos = [TempoEvent(str(e.get("id") or _new_id()), int(e["tick"]), float(e["bpm"])) for e in data.get("tempos") or ()]
        meters = [
            MeterEvent(str(e.get("id") or _new_id()), int(e["tick"]), int(e["numerator"]), int(e["denominator"]))
            for e in data.get("meters") or ()
        ]
        bars = [
            Bar(
                str(b.get("id") or _new_id()), int(b["start_tick"]), int(b["end_tick"]),
                int(b["numerator"]), int(b["denominator"]), str(b.get("origin") or ORIGIN_GRID),
            )
            for b in data.get("bars") or ()
        ]
        return cls(session, tempos, meters, bars)

    def copy(self) -> SessionTimeline:
        return SessionTimeline.from_dict(self.to_dict())
