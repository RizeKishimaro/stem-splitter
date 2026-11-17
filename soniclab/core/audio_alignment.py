"""Compare detected beats with the session grid. Suggestions only.

Detected beats and downbeats stay in their source file's sample frames
(``DetectedBeat``). Placing a clip maps them to arrangement seconds, where
they can be measured against the session's bar or beat lines. Nothing here
changes the session timeline or a clip: the caller decides what to do with
a suggestion (move the clip, or nothing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soniclab.core.session_timeline import DetectedBeat, SessionTimeline


def detected_beats(events, rate: int) -> list[DetectedBeat]:
    """``(source seconds, beat position)`` events, as stored on a track, as
    beats in source sample frames. Position 1 is a downbeat."""
    beats = []
    for number, (time, position) in enumerate(events or ()):
        beats.append(DetectedBeat(
            id=f"beat:{number}",
            frame=int(round(float(time) * int(rate))),
            confidence=1.0,
            downbeat=int(position) == 1,
        ))
    return beats


def arrangement_seconds(beat: DetectedBeat, rate: int, clip_start: float, clip_offset: float) -> float:
    """Where a detected beat sounds on the arrangement, for a clip placed at
    ``clip_start`` that plays its file from ``clip_offset`` (seconds)."""
    return float(clip_start) + beat.frame / float(rate) - float(clip_offset)


@dataclass(frozen=True)
class AlignmentSuggestion:
    """How far the clip would move (seconds, positive = later) so its
    detected downbeats land on the session's bar lines; the spread of what
    is left over (seconds), and how many beats were compared."""

    offset: float
    spread: float
    compared: int


def suggest_offset(
    timeline: SessionTimeline,
    beats,
    rate: int,
    clip_start: float,
    clip_offset: float,
    clip_duration: float,
    downbeats_only: bool = True,
) -> AlignmentSuggestion | None:
    """Suggest a clip offset that lines its detected (down)beats up with the
    session's bar lines (or beat lines when ``downbeats_only`` is False).
    Beats outside the clip's audible stretch are not counted. None when no
    beat can be compared."""
    end = float(clip_start) + float(clip_duration)
    times = [
        arrangement_seconds(beat, rate, clip_start, clip_offset)
        for beat in beats
        if (beat.downbeat or not downbeats_only)
    ]
    times = [time for time in times if float(clip_start) <= time < end]
    if not times:
        return None
    until = max(times) + 60.0
    if downbeats_only:
        grid = timeline.get_bar_lines(until)
    else:
        grid = [time for time, _bar in timeline.get_beat_lines(until)]
    grid = np.asarray(grid, dtype=np.float64)
    residuals = []
    for time in times:
        nearest = grid[int(np.argmin(np.abs(grid - time)))]
        residuals.append(nearest - time)
    residuals = np.asarray(residuals)
    offset = float(np.median(residuals))
    spread = float(np.median(np.abs(residuals - offset)))
    return AlignmentSuggestion(offset, spread, len(times))
