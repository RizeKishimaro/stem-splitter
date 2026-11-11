"""Compare source onsets with a Madmom beat reference.

An onset is not a beat. It is classified only when it sits close enough to
one reference beat of any position (1, 2, 3, or 4). Other onsets stay
unmatched. ``alignment_offset`` shifts the comparison without moving the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

ON_TIME_SECONDS = 0.020
MAX_MATCH_SECONDS = 0.080

KIND_EARLY = "early"
KIND_ON_TIME = "on_time"
KIND_LATE = "late"
KIND_UNMATCHED = "unmatched"


@dataclass(frozen=True)
class OnsetMark:
    """One onset painted on a clip, plus its relationship to the reference."""

    source_seconds: float
    local_seconds: float
    kind: str
    error_seconds: float | None = None


def arrangement_time(source_seconds: float, clip_start: float, clip_offset: float, alignment: float) -> float:
    """Map a source timestamp onto the arrangement without moving the clip."""
    return float(clip_start) + (float(source_seconds) - float(clip_offset)) + float(alignment)


def reference_beats(events, clip_start: float, clip_offset: float, alignment: float) -> tuple[tuple[float, int], ...]:
    """Arrangement times for every Madmom beat position, not only downbeats."""
    beats = []
    for time_seconds, position in events:
        beat_position = int(position)
        if beat_position < 1:
            continue
        beats.append(
            (
                arrangement_time(time_seconds, clip_start, clip_offset, alignment),
                beat_position,
            )
        )
    beats.sort()
    return tuple(beats)


def classify_onsets(
    onsets,
    reference_events,
    onset_start: float,
    onset_offset: float,
    onset_alignment: float,
    onset_duration: float,
    reference_start: float,
    reference_offset: float,
    reference_alignment: float,
) -> tuple[OnsetMark, ...]:
    """Classify onsets. Unmatched events stay in the result and carry no error."""
    beats = reference_beats(reference_events, reference_start, reference_offset, reference_alignment)
    duration = float(onset_duration)
    origin = float(onset_offset)
    marks = []
    for source_seconds in onsets:
        source = float(source_seconds)
        local = source - origin
        if local < 0.0 or local >= duration:
            continue
        when = arrangement_time(source, onset_start, onset_offset, onset_alignment)
        error, kind = _classify_one(when, beats)
        marks.append(OnsetMark(source, local, kind, error))
    return tuple(marks)


def matched_errors(marks) -> tuple[float, ...]:
    return tuple(mark.error_seconds for mark in marks if mark.error_seconds is not None)


def nudged_alignment(alignment: float, errors) -> float | None:
    """Return the alignment that cancels the median confident error.

    ``None`` means there is no confident match. The clip start is not involved.
    """
    values = [float(error) for error in errors]
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return float(alignment) - median


def _classify_one(when: float, beats: tuple[tuple[float, int], ...]) -> tuple[float | None, str]:
    if not beats:
        return None, KIND_UNMATCHED
    nearest = 0
    nearest_distance = abs(beats[0][0] - when)
    for index, (beat_time, _position) in enumerate(beats[1:], start=1):
        distance = abs(beat_time - when)
        if distance < nearest_distance:
            nearest = index
            nearest_distance = distance
    error = when - beats[nearest][0]
    if abs(error) > _gate(beats, nearest):
        return None, KIND_UNMATCHED
    if abs(error) <= ON_TIME_SECONDS:
        return error, KIND_ON_TIME
    if error < 0.0:
        return error, KIND_EARLY
    return error, KIND_LATE


def _gate(beats: tuple[tuple[float, int], ...], index: int) -> float:
    """Half the neighboring reference gap, and never wider than 80 ms."""
    if len(beats) < 2:
        return MAX_MATCH_SECONDS
    if index <= 0:
        gap = beats[1][0] - beats[0][0]
    elif index >= len(beats) - 1:
        gap = beats[-1][0] - beats[-2][0]
    else:
        gap = min(beats[index][0] - beats[index - 1][0], beats[index + 1][0] - beats[index][0])
    if gap <= 0.0:
        return MAX_MATCH_SECONDS
    return min(gap * 0.5, MAX_MATCH_SECONDS)
