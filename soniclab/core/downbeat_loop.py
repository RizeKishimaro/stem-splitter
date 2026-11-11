"""Loop bounds from detected downbeat timestamps.

Arrangement time is ``clip_start + (time_seconds - clip_offset) + alignment``.
``alignment`` is zero until a downbeat-alignment control exists. Both edges
move by the same amount, and the source timestamps are left unchanged.
"""

from __future__ import annotations

DEFAULT_LOOP_SHORTCUTS = {
    1: "Ctrl+Alt+1",
    2: "Ctrl+Alt+2",
    3: "Ctrl+Alt+3",
    4: "Ctrl+Alt+4",
}


def loop_shortcut(intervals: int, settings=None) -> str:
    """Shortcut for a downbeat-loop length. A stored value overrides the default."""
    default = DEFAULT_LOOP_SHORTCUTS[int(intervals)]
    if settings is None:
        return default
    value = settings.value(f"shortcuts/downbeatLoop{int(intervals)}", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def arrangement_downbeats(
    events, clip_start: float, clip_offset: float, alignment: float = 0.0
) -> tuple[float, ...]:
    """Map source downbeats (beat position 1) onto the arrangement timeline."""
    origin = float(clip_start) - float(clip_offset) + float(alignment)
    times = [
        origin + float(time_seconds)
        for time_seconds, position in events
        if int(position) == 1
    ]
    times.sort()
    return tuple(times)


def downbeat_loop_region(
    events,
    playhead: float,
    intervals: int,
    clip_start: float,
    clip_offset: float,
    alignment: float = 0.0,
) -> tuple[float, float] | None:
    """Return ``(start, end)`` in arrangement seconds, or None when it cannot be formed.

    The start is the downbeat nearest the playhead. An equal distance keeps the
    earlier one. The end is that downbeat plus ``intervals`` later downbeats.
    """
    count = int(intervals)
    if count < 1:
        return None
    times = arrangement_downbeats(events, clip_start, clip_offset, alignment)
    if not times:
        return None
    playhead = float(playhead)
    nearest = 0
    nearest_distance = abs(times[0] - playhead)
    for index, time in enumerate(times[1:], start=1):
        distance = abs(time - playhead)
        if distance < nearest_distance:
            nearest = index
            nearest_distance = distance
    end_index = nearest + count
    if end_index >= len(times):
        return None
    return (times[nearest], times[end_index])
