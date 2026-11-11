"""Copy detected beats between clips, and to and from downbeat files.

A track may carry its own ``downbeats`` list of ``[time_seconds,
beat_position]`` pairs in that track's source time. When present it replaces
the Madmom result for the track's file, so applied downbeats survive a save
and do not need the source audio to be analyzed again.

Transfer between clips keeps arrangement time: a downbeat that sits at 12.0 s
on the timeline under the source clip sits at 12.0 s under the target clip.

A downbeat file stores clip-local seconds (``0`` is the left edge of the clip
it came from). Importing lines that origin up with the target clip's left edge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FILE_FORMAT = "soniclab-downbeats"
FILE_VERSION = 1
FILE_SUFFIX = ".downbeats.json"


class DownbeatFileError(ValueError):
    """The file is not a downbeat document this version can read."""


def track_events(track: dict, detected=()) -> tuple[tuple[float, int], ...]:
    """Beats for a track: its applied list when it has one, else ``detected``."""
    stored = track.get("downbeats")
    if stored:
        return tuple((float(time), int(position)) for time, position in stored)
    return tuple((float(time), int(position)) for time, position in detected)


def transfer_events(
    events,
    source_start: float,
    source_offset: float,
    target_start: float,
    target_offset: float,
) -> list[list]:
    """Re-express source beats in the target's source time.

    ``target_time = source_time - source_offset + source_start - target_start
    + target_offset``. Beats that would land before the target file starts are
    dropped. The result is sorted and strictly increasing.
    """
    shift = (
        float(source_start)
        - float(source_offset)
        - float(target_start)
        + float(target_offset)
    )
    return _clean((float(time) + shift, int(position)) for time, position in events)


def default_folder(project_path: str | None = None) -> Path:
    """``Downbeats`` beside a saved project, else a shared user folder."""
    if project_path:
        return Path(project_path).expanduser().resolve().parent / "Downbeats"
    return Path.home() / ".local" / "share" / "soniclab" / "downbeats"


def export_document(
    events,
    clip_offset: float,
    clip_duration: float,
    *,
    name: str = "",
    tempo_bpm: float | None = None,
    time_signature=None,
) -> dict:
    """Clip-local beats inside the clip, ready to write as JSON."""
    origin = float(clip_offset)
    length = float(clip_duration)
    local = [
        (float(time) - origin, int(position))
        for time, position in events
        if 0.0 <= float(time) - origin < length
    ]
    return {
        "format": FILE_FORMAT,
        "version": FILE_VERSION,
        "name": str(name),
        "time_unit": "seconds",
        "beat_position_semantics": "1_is_downbeat",
        "clip_duration": length,
        "tempo_bpm": None if tempo_bpm is None else float(tempo_bpm),
        "time_signature": list(time_signature) if time_signature else None,
        "events": [
            {"local_seconds": round(time, 6), "beat_position": position}
            for time, position in _clean(local)
        ],
    }


def write_document(path, document: dict) -> Path:
    target = Path(path).expanduser()
    if not target.name.endswith(FILE_SUFFIX):
        target = target.with_name(target.name.removesuffix(".json") + FILE_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def read_document(path) -> dict:
    try:
        document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownbeatFileError(str(exc)) from exc
    if not isinstance(document, dict) or document.get("format") != FILE_FORMAT:
        raise DownbeatFileError("not a SonicLab downbeat file")
    if document.get("version") != FILE_VERSION:
        raise DownbeatFileError(f"unsupported downbeat file version {document.get('version')!r}")
    raw = document.get("events")
    if not isinstance(raw, list):
        raise DownbeatFileError("events must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise DownbeatFileError("each event must be an object")
        seconds = item.get("local_seconds")
        position = item.get("beat_position")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds != seconds:
            raise DownbeatFileError("local_seconds must be a number")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise DownbeatFileError("beat_position must be an integer >= 1")
    return document


def import_events(document: dict, target_offset: float) -> list[list]:
    """Place file beats so local ``0`` is the target clip's left edge."""
    origin = float(target_offset)
    return _clean(
        (float(item["local_seconds"]) + origin, int(item["beat_position"]))
        for item in document.get("events") or ()
    )


def _clean(pairs) -> list[list]:
    ordered = sorted((time, position) for time, position in pairs if time >= 0.0)
    result: list[list] = []
    for time, position in ordered:
        if result and time <= result[-1][0]:
            continue
        result.append([float(time), int(position)])
    return result
