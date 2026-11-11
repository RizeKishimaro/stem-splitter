"""Bar lines on the arrangement: where each bar starts and ends.

A bar map is a sorted list of bar-line times in arrangement seconds. Bar
``n`` (counted from 1) runs from line ``n - 1`` to line ``n``. Past the last
line the map repeats the last bar length, so the ruler never runs out. Beats
split each bar evenly, so they always land on the bar lines, even when bars
have different lengths.
"""

from __future__ import annotations

import bisect
import re

# Two bar lines closer than this are treated as one.
MIN_BAR_SECONDS = 0.1


def bar_seconds(bpm: float, beats_per_bar: int, beat_unit: int) -> float:
    beat = (4.0 / max(1, int(beat_unit))) * 60.0 / max(1.0, float(bpm))
    return max(MIN_BAR_SECONDS, beat * max(1, int(beats_per_bar)))


def grid_bars(bpm: float, beats_per_bar: int, beat_unit: int, until: float) -> list[float]:
    """Evenly spaced bar lines from 0 through ``until``."""
    length = bar_seconds(bpm, beats_per_bar, beat_unit)
    count = int(max(0.0, float(until)) / length) + 2
    return [round(index * length, 6) for index in range(count)]


def clean(lines) -> list[float]:
    """Sorted, non-negative, with lines closer than MIN_BAR_SECONDS merged."""
    result: list[float] = []
    for value in sorted(max(0.0, float(line)) for line in lines or ()):
        if not result or value - result[-1] >= MIN_BAR_SECONDS:
            result.append(value)
    return result


def bars_from_downbeats(events, source_start: float = 0.0, source_offset: float = 0.0) -> list[float]:
    """Bar lines at the downbeats (position 1) of a clip, in arrangement time.

    ``events`` are ``(time, position)`` pairs in the clip's source time.
    Downbeats before the clip's offset are not heard, so they are dropped.
    """
    lines = []
    for time, position in events or ():
        if int(position) != 1 or float(time) < float(source_offset):
            continue
        lines.append(float(time) - float(source_offset) + float(source_start))
    return clean(lines)


def last_length(lines, fallback: float) -> float:
    if len(lines) >= 2:
        return max(MIN_BAR_SECONDS, lines[-1] - lines[-2])
    return max(MIN_BAR_SECONDS, float(fallback))


def extend(lines, until: float, fallback: float) -> list[float]:
    """The map's lines plus repeats of the last bar length through ``until``."""
    result = list(lines) or [0.0]
    length = last_length(result, fallback)
    while result[-1] < until:
        result.append(round(result[-1] + length, 6))
    return result


def beat_lines(lines, beats_per_bar: int) -> list[tuple[float, bool]]:
    """``(time, is_bar_line)`` for every beat between the given bar lines."""
    beats = max(1, int(beats_per_bar))
    marks = []
    for start, end in zip(lines, lines[1:]):
        step = (end - start) / beats
        marks.append((start, True))
        for beat in range(1, beats):
            marks.append((start + beat * step, False))
    if lines:
        marks.append((lines[-1], True))
    return marks


def bar_index(lines, seconds: float, fallback: float) -> int:
    """Zero-based bar under ``seconds``; -1 before the first line."""
    if not lines or seconds < lines[0]:
        return -1
    position = bisect.bisect_right(lines, seconds) - 1
    if position < len(lines) - 1:
        return position
    length = last_length(lines, fallback)
    return position + int((seconds - lines[-1]) / length)


def bar_span(lines, index: int, fallback: float) -> tuple[float, float]:
    """Start and end of bar ``index`` (zero-based), extending past the map."""
    full = extend(lines, 0.0, fallback)
    while len(full) <= index + 1:
        full.append(round(full[-1] + last_length(full, fallback), 6))
    return full[index], full[index + 1]


def position(lines, seconds: float, beats_per_bar: int, fallback: float) -> tuple[int, int]:
    """1-based ``(bar, beat)`` at ``seconds``. Bar 0 is the lead-in before bar 1."""
    index = bar_index(lines, seconds, fallback)
    if index < 0:
        if not lines:
            return 0, 1
        # The lead-in is as long as the first bar.
        length = last_length(lines[:2], fallback)
        start, end = lines[0] - length, lines[0]
    else:
        start, end = bar_span(lines, index, fallback)
    beats = max(1, int(beats_per_bar))
    beat = int((seconds - start) / max(1e-9, end - start) * beats)
    return index + 1, min(beats, max(0, beat) + 1)


def move_line(lines, index: int, seconds: float, ripple: bool = False) -> list[float]:
    """Move one bar line, kept between its neighbours.

    With ``ripple`` every later line moves by the same amount, so the bars
    after it keep their lengths.
    """
    result = list(lines)
    if not (0 <= index < len(result)):
        return result
    low = result[index - 1] + MIN_BAR_SECONDS if index > 0 else 0.0
    if ripple:
        target = max(low, float(seconds))
        shift = target - result[index]
        return [value + shift if position >= index else value for position, value in enumerate(result)]
    high = result[index + 1] - MIN_BAR_SECONDS if index + 1 < len(result) else float("inf")
    result[index] = min(high, max(low, float(seconds)))
    return result


def set_span(lines, index: int, start: float, end: float, fallback: float) -> list[float]:
    """Make bar ``index`` (zero-based) run from ``start`` to ``end``.

    Lines past the map are filled in first. The bars on either side stretch
    or shrink to meet the new edges.
    """
    if end - start < MIN_BAR_SECONDS:
        raise ValueError("A bar must end after it starts")
    result = extend(lines, 0.0, fallback)
    while len(result) <= index + 1:
        result.append(round(result[-1] + last_length(result, fallback), 6))
    low = result[index - 1] + MIN_BAR_SECONDS if index > 0 else 0.0
    high = result[index + 2] - MIN_BAR_SECONDS if index + 2 < len(result) else float("inf")
    if start < low or end > high:
        raise ValueError("The bar would overlap the bar before or after it")
    result[index] = float(start)
    result[index + 1] = float(end)
    return result


def insert_line(lines, seconds: float) -> list[float]:
    return clean(list(lines) + [float(seconds)])


def remove_line(lines, index: int) -> list[float]:
    return [value for position, value in enumerate(lines) if position != index]


def nearest_line(lines, seconds: float) -> int:
    if not lines:
        return -1
    return min(range(len(lines)), key=lambda position: abs(lines[position] - seconds))


def format_clock(seconds: float) -> str:
    """``HH:MM:SS.cc``"""
    hundredths = int(round(max(0.0, float(seconds)) * 100))
    hours, rest = divmod(hundredths, 360000)
    minutes, rest = divmod(rest, 6000)
    whole, cents = divmod(rest, 100)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{cents:02d}"


def parse_clock(text: str) -> float:
    """Read ``HH:MM:SS.cc``, ``MM:SS.cc`` or plain seconds."""
    value = str(text or "").strip()
    if not re.fullmatch(r"\d+(:\d+){0,2}(\.\d+)?", value):
        raise ValueError(f"Not a time: {text!r}")
    total = 0.0
    for part in value.split(":"):
        total = total * 60.0 + float(part)
    return total
