"""Sprite animation projects for the animation editor.

A project describes a sheet, the frames cut from it, and named animations.
The base animation is timed by BPM alone. Kick, snare, hat, bass and
percussion animations play on top of that clock and never change its speed.

The JSON format is ``soniclab.sprite`` version 1. Paths are relative to the
project file when they are not already absolute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "soniclab.sprite"
VERSION = 1

LAYERS = (
    "body",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "beat_accents",
    "instrument",
)
# Clock events drive the base dance. Drum events are overlays.
CLOCK_TRIGGERS = ("", "beat", "downbeat")
OVERLAY_TRIGGERS = ("kick", "snare", "hat", "bass", "perc")
TRIGGERS = ("downbeat", "beat") + OVERLAY_TRIGGERS
LOOP_MODES = ("loop", "once", "reverse", "pingpong")
POSE_CATEGORIES = ("unknown", "idle", "dance", "arms", "kick", "lean", "spin", "turn")
GRIDS = ("beat", "eighth", "sixteenth")
GRID_BEATS = {"beat": 1.0, "eighth": 0.5, "sixteenth": 0.25}
OVERLAY_PRIORITY = ("kick", "snare", "hat", "bass", "perc")


@dataclass
class SheetSpec:
    """How a PNG is cut into frames. Spacing is the gap between cells."""

    file: str
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    spacing_x: int = 0
    spacing_y: int = 0
    margin_x: int = 0
    margin_y: int = 0
    image_width: int = 0
    image_height: int = 0

    def count(self) -> int:
        return max(0, int(self.columns)) * max(0, int(self.rows))

    def rect(self, source: int) -> tuple[int, int, int, int]:
        columns = max(1, int(self.columns))
        source = int(source)
        column = source % columns
        row = source // columns
        x = int(self.margin_x) + column * (int(self.frame_width) + int(self.spacing_x))
        y = int(self.margin_y) + row * (int(self.frame_height) + int(self.spacing_y))
        return x, y, int(self.frame_width), int(self.frame_height)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "columns": int(self.columns),
            "rows": int(self.rows),
            "spacing_x": int(self.spacing_x),
            "spacing_y": int(self.spacing_y),
            "margin_x": int(self.margin_x),
            "margin_y": int(self.margin_y),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SheetSpec:
        data = data or {}
        width = int(data.get("frame_width", data.get("frameWidth", 0)) or 0)
        height = int(data.get("frame_height", data.get("frameHeight", width)) or 0)
        return cls(
            file=str(data.get("file", "")),
            frame_width=width,
            frame_height=height,
            columns=max(1, int(data.get("columns", 1) or 1)),
            rows=max(1, int(data.get("rows", 1) or 1)),
            spacing_x=int(data.get("spacing_x", 0) or 0),
            spacing_y=int(data.get("spacing_y", 0) or 0),
            margin_x=int(data.get("margin_x", 0) or 0),
            margin_y=int(data.get("margin_y", 0) or 0),
            image_width=int(data.get("image_width", 0) or 0),
            image_height=int(data.get("image_height", 0) or 0),
        )


@dataclass
class FrameInfo:
    """One cell. ``source`` is the sheet cell; ``id`` stays put when reordered."""

    id: int
    name: str
    source: int
    category: str = ""
    confidence: str = ""
    note: str = ""
    rect: tuple[int, int, int, int] | None = None
    content_rect: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict:
        data = {"id": int(self.id), "name": self.name, "source": int(self.source)}
        if self.category:
            data["category"] = self.category
        if self.confidence:
            data["confidence"] = self.confidence
        if self.note:
            data["note"] = self.note
        if self.rect is not None:
            data["rect"] = [int(value) for value in self.rect]
        if self.content_rect is not None:
            data["content_rect"] = [int(value) for value in self.content_rect]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> FrameInfo:
        source = int(data.get("source", data.get("id", 0)))
        rect = data.get("rect")
        content = data.get("content_rect")
        return cls(
            id=int(data.get("id", source)),
            name=str(data.get("name", f"Frame {source}")),
            source=source,
            category=str(data.get("category", "") or ""),
            confidence=str(data.get("confidence", "") or ""),
            note=str(data.get("note", "") or ""),
            rect=tuple(int(value) for value in rect) if rect else None,
            content_rect=tuple(int(value) for value in content) if content else None,
        )


@dataclass
class Sync:
    type: str = "beat"
    grid: str = "beat"
    beats_per_cycle: float = 1.0
    speed_multiplier: float = 1.0
    intensity: float = 1.0
    sensitivity: float = 0.5
    cooldown: float = 0.0

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "grid": self.grid,
            "beats_per_cycle": float(self.beats_per_cycle),
            "speed_multiplier": float(self.speed_multiplier),
            "intensity": float(self.intensity),
            "sensitivity": float(self.sensitivity),
            "cooldown": float(self.cooldown),
        }

    @classmethod
    def from_dict(cls, data: dict | None, trigger: str = "") -> Sync:
        data = data or {}
        kind = str(data.get("type", trigger or "beat"))
        grid = str(data.get("grid", "beat"))
        if grid not in GRIDS:
            grid = "beat"
        return cls(
            type=kind,
            grid=grid,
            beats_per_cycle=float(data.get("beats_per_cycle", 1.0) or 1.0),
            speed_multiplier=float(data.get("speed_multiplier", 1.0) or 1.0),
            intensity=float(data.get("intensity", 1.0) or 0.0),
            sensitivity=float(data.get("sensitivity", 0.5) or 0.0),
            cooldown=float(data.get("cooldown", 0.0) or 0.0),
        )


@dataclass
class Animation:
    name: str
    layer: str = "body"
    frames: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    fps: float = 12.0
    loop: str = "loop"
    base: bool = False
    trigger: str = ""
    sync: Sync = field(default_factory=Sync)
    note: str = ""
    paint: str = ""

    def to_dict(self) -> dict:
        payload = {
            "layer": self.layer,
            "frames": [int(frame) for frame in self.frames],
            "durations": [float(duration) for duration in self.durations],
            "fps": float(self.fps),
            "loop": self.loop,
            "base": bool(self.base),
            "speed_multiplier": float(self.sync.speed_multiplier),
            "sync": self.sync.to_dict(),
        }
        if self.trigger:
            payload["trigger"] = self.trigger
        if self.note:
            payload["note"] = self.note
        if self.paint:
            payload["paint"] = self.paint
        return payload

    @classmethod
    def from_dict(cls, name: str, data: dict) -> Animation:
        data = data or {}
        trigger = str(data.get("trigger", "") or "")
        sync = Sync.from_dict(data.get("sync"), trigger)
        if "speed_multiplier" in data and "sync" not in data:
            sync.speed_multiplier = float(data.get("speed_multiplier") or 1.0)
        elif "speed_multiplier" in data and not data.get("sync", {}).get("speed_multiplier"):
            sync.speed_multiplier = float(data.get("speed_multiplier") or sync.speed_multiplier)
        loop = data.get("loop", "loop")
        if loop is True:
            loop = "loop"
        elif loop is False:
            loop = "once"
        loop = str(loop)
        if loop not in LOOP_MODES:
            loop = "loop"
        if not trigger and sync.type in OVERLAY_TRIGGERS:
            trigger = sync.type
        layer = str(data.get("layer", "body") or "body")
        frames = [int(frame) for frame in data.get("frames", [])]
        durations = [float(duration) for duration in data.get("durations", [])]
        base = bool(data.get("base", False))
        if "base" not in data and not trigger:
            base = True
        return cls(
            name=str(name),
            layer=layer,
            frames=frames,
            durations=durations,
            fps=float(data.get("fps", 12.0) or 12.0),
            loop=loop,
            base=base,
            trigger=trigger,
            sync=sync,
            note=str(data.get("note", "") or ""),
            paint=str(data.get("paint", "") or ""),
        )


@dataclass
class SpriteProject:
    name: str = "character"
    bpm: float = 120.0
    sheet: SheetSpec = field(default_factory=lambda: SheetSpec("", 32, 32, 1, 1))
    frames: list[FrameInfo] = field(default_factory=list)
    animations: dict[str, Animation] = field(default_factory=dict)
    root: str = ""

    def frame(self, frame_id: int) -> FrameInfo | None:
        for item in self.frames:
            if item.id == int(frame_id):
                return item
        return None

    def source_of(self, frame_id: int) -> int:
        item = self.frame(frame_id)
        return item.source if item is not None else int(frame_id)

    def rect_of(self, frame_id: int) -> tuple[int, int, int, int]:
        item = self.frame(frame_id)
        if item is None:
            return self.sheet.rect(int(frame_id))
        if item.rect is not None:
            return tuple(int(value) for value in item.rect)
        return self.sheet.rect(item.source)

    def rect_of_source(self, source: int) -> tuple[int, int, int, int]:
        source = int(source)
        for item in self.frames:
            if item.source == source and item.rect is not None:
                return tuple(int(value) for value in item.rect)
        return self.sheet.rect(source)

    def frame_at_point(self, x: int, y: int) -> FrameInfo | None:
        hits: list[tuple[int, FrameInfo]] = []
        for item in self.frames:
            left, top, width, height = self.rect_of(item.id)
            if left <= x < left + width and top <= y < top + height:
                hits.append((width * height, item))
        if not hits:
            return None
        hits.sort(key=lambda pair: pair[0])
        return hits[0][1]

    def to_frame_map(self) -> dict:
        """Reusable frame-map schema. Animations stay in their own file."""
        frames = {}
        for item in sorted(self.frames, key=lambda frame: frame.id):
            rect = [int(value) for value in self.rect_of(item.id)]
            column = item.source % max(1, self.sheet.columns)
            row = item.source // max(1, self.sheet.columns)
            payload = {
                "rect": rect,
                "category": item.category or "unknown",
                "column": column,
                "row": row,
                "sheet_index": int(item.source),
            }
            if item.confidence:
                payload["confidence"] = item.confidence
            if item.note:
                payload["note"] = item.note
            if item.content_rect is not None:
                payload["content_rect"] = [int(value) for value in item.content_rect]
            frames[f"frame_{item.id:03d}"] = payload
        return {
            "format": "soniclab.sprite-map",
            "version": 1,
            "character": self.name,
            "source": Path(self.sheet.file).name if self.sheet.file else "",
            "image": {"width": int(self.sheet.image_width), "height": int(self.sheet.image_height)},
            "grid": {
                "columns": int(self.sheet.columns),
                "rows": int(self.sheet.rows),
                "cell_width": int(self.sheet.frame_width),
                "cell_height": int(self.sheet.frame_height),
                "spacing_x": int(self.sheet.spacing_x),
                "spacing_y": int(self.sheet.spacing_y),
                "margin_x": int(self.sheet.margin_x),
                "margin_y": int(self.sheet.margin_y),
            },
            "frames": frames,
            "animations": {},
        }

    def next_id(self) -> int:
        if not self.frames:
            return 0
        return max(item.id for item in self.frames) + 1

    def base_animation(self) -> Animation | None:
        for animation in self.animations.values():
            if animation.base and not animation.trigger:
                return animation
        for animation in self.animations.values():
            if not animation.trigger:
                return animation
        return None

    def animation_for_trigger(self, trigger: str) -> Animation | None:
        for animation in self.animations.values():
            if animation.trigger == trigger:
                return animation
        return None

    def animations_on_layer(self, layer: str) -> list[Animation]:
        return [animation for animation in self.animations.values() if animation.layer == layer]

    def resolve_sheet(self) -> Path:
        path = Path(self.sheet.file)
        if path.is_file():
            return path
        if self.root:
            candidate = Path(self.root) / path
            if candidate.is_file():
                return candidate
        assets = Path(__file__).resolve().parent.parent / "assets"
        candidate = assets / path.name
        return candidate

    def duplicate_frames(self, ids: list[int]) -> list[int]:
        created: list[int] = []
        for frame_id in ids:
            original = self.frame(frame_id)
            if original is None:
                continue
            copy = FrameInfo(
                self.next_id(),
                f"{original.name} copy",
                original.source,
                category=original.category,
                confidence=original.confidence,
                note=original.note,
                rect=original.rect,
                content_rect=original.content_rect,
            )
            index = next(i for i, item in enumerate(self.frames) if item.id == original.id)
            self.frames.insert(index + 1, copy)
            created.append(copy.id)
        return created

    def delete_frames(self, ids: list[int]) -> None:
        gone = {int(frame_id) for frame_id in ids}
        self.frames = [item for item in self.frames if item.id not in gone]
        for animation in self.animations.values():
            durations = _sequence_durations(animation) if animation.durations else []
            kept_frames = []
            kept_durations = []
            for index, frame in enumerate(animation.frames):
                if frame in gone:
                    continue
                kept_frames.append(frame)
                if durations:
                    kept_durations.append(durations[index])
            animation.frames = kept_frames
            animation.durations = kept_durations

    def move_frames(self, ids: list[int], index: int) -> None:
        """Move ``ids`` so they begin at ``index`` in the list that remains."""
        by_id = {item.id: item for item in self.frames}
        moving = [by_id[int(frame_id)] for frame_id in ids if int(frame_id) in by_id]
        if not moving:
            return
        moving_ids = {item.id for item in moving}
        rest = [item for item in self.frames if item.id not in moving_ids]
        index = max(0, min(len(rest), int(index)))
        self.frames = rest[:index] + moving + rest[index:]

    def rename_frame(self, frame_id: int, name: str) -> None:
        item = self.frame(frame_id)
        if item is None:
            return
        cleaned = " ".join(str(name).split())
        if cleaned:
            item.name = cleaned

    def add_to_animation(self, name: str, ids: list[int], index: int | None = None) -> None:
        animation = self.animations.get(name)
        if animation is None:
            return
        ids = [int(frame_id) for frame_id in ids if self.frame(frame_id) is not None]
        durations = _sequence_durations(animation)
        if index is None or index >= len(animation.frames):
            animation.frames.extend(ids)
            if durations:
                step = durations[-1] if durations else _default_step(animation)
                animation.durations = durations + [step] * len(ids)
            return
        index = max(0, int(index))
        animation.frames[index:index] = ids
        if durations:
            step = durations[min(index, len(durations) - 1)]
            durations[index:index] = [step] * len(ids)
            animation.durations = durations

    def reorder_animation(self, name: str, positions: list[int], index: int) -> None:
        animation = self.animations.get(name)
        if animation is None:
            return
        positions = sorted({int(position) for position in positions if 0 <= int(position) < len(animation.frames)})
        if not positions:
            return
        durations = _sequence_durations(animation)
        moving_frames = [animation.frames[position] for position in positions]
        moving_durations = [durations[position] for position in positions]
        rest_frames = [frame for position, frame in enumerate(animation.frames) if position not in positions]
        rest_durations = [duration for position, duration in enumerate(durations) if position not in positions]
        removed_before = sum(1 for position in positions if position < index)
        dest = max(0, min(len(rest_frames), int(index) - removed_before))
        animation.frames = rest_frames[:dest] + moving_frames + rest_frames[dest:]
        animation.durations = rest_durations[:dest] + moving_durations + rest_durations[dest:]

    def remove_animation_positions(self, name: str, positions: list[int]) -> None:
        animation = self.animations.get(name)
        if animation is None:
            return
        gone = {int(position) for position in positions}
        durations = _sequence_durations(animation)
        animation.frames = [frame for position, frame in enumerate(animation.frames) if position not in gone]
        animation.durations = [duration for position, duration in enumerate(durations) if position not in gone]

    def set_durations(self, name: str, positions: list[int], beats: float) -> None:
        animation = self.animations.get(name)
        if animation is None:
            return
        beats = max(0.01, float(beats))
        durations = _sequence_durations(animation)
        for position in positions:
            if 0 <= int(position) < len(durations):
                durations[int(position)] = beats
        animation.durations = durations

    def apply_fps(self, name: str, fps: float) -> None:
        """Even frame lengths so ``fps`` is the rate at this project's BPM."""
        animation = self.animations.get(name)
        if animation is None or not animation.frames:
            return
        fps = max(0.1, float(fps))
        animation.fps = fps
        step = (float(self.bpm) / 60.0) / fps
        animation.durations = [step] * len(animation.frames)
        animation.sync.beats_per_cycle = step * len(animation.frames)

    def variation(self, source: str, new_name: str, stride: int = 1) -> str:
        animation = self.animations[source]
        stride = max(1, int(stride))
        name = unique_name(new_name, self.animations)
        frames = animation.frames[::stride] or list(animation.frames)
        durations = _sequence_durations(animation)[::stride]
        self.animations[name] = Animation(
            name=name,
            layer=animation.layer,
            frames=frames,
            durations=durations[: len(frames)],
            fps=animation.fps,
            loop=animation.loop,
            base=False,
            trigger=animation.trigger,
            sync=Sync.from_dict(animation.sync.to_dict(), animation.trigger),
        )
        return name

    def rename_animation(self, old: str, new: str) -> str:
        if old not in self.animations:
            return old
        cleaned = " ".join(str(new).split()).replace(" ", "_")
        if not cleaned or cleaned == old:
            return old
        cleaned = unique_name(cleaned, self.animations)
        animation = self.animations.pop(old)
        animation.name = cleaned
        self.animations[cleaned] = animation
        return cleaned

    def fill_cells(self) -> None:
        """One frame per sheet cell, keeping names for sources that already exist."""
        by_source = {item.source: item for item in self.frames}
        frames: list[FrameInfo] = []
        next_id = 0
        for source in range(self.sheet.count()):
            existing = by_source.get(source)
            if existing is not None and existing.id not in {item.id for item in frames}:
                frames.append(existing)
                next_id = max(next_id, existing.id + 1)
                continue
            frames.append(FrameInfo(next_id, f"Frame {source}", source))
            next_id += 1
        self.frames = frames

    def normalize(self) -> None:
        self.bpm = max(1.0, float(self.bpm or 120.0))
        self.sheet.columns = max(1, int(self.sheet.columns))
        self.sheet.rows = max(1, int(self.sheet.rows))
        self.sheet.frame_width = max(1, int(self.sheet.frame_width))
        self.sheet.frame_height = max(1, int(self.sheet.frame_height))
        seen: set[int] = set()
        unique: list[FrameInfo] = []
        for item in self.frames:
            if item.id in seen:
                continue
            seen.add(item.id)
            item.name = item.name.strip() or f"Frame {item.source}"
            unique.append(item)
        self.frames = unique
        known = {item.id for item in self.frames}
        for animation in self.animations.values():
            for frame_id in animation.frames:
                if int(frame_id) not in known:
                    known.add(int(frame_id))
                    self.frames.append(FrameInfo(int(frame_id), f"Frame {int(frame_id)}", int(frame_id)))
        base_named = False
        for animation in self.animations.values():
            animation.frames = [int(frame) for frame in animation.frames if int(frame) in known]
            animation.durations = _fit_durations(animation.durations, animation.frames)
            animation.sync.speed_multiplier = max(0.05, float(animation.sync.speed_multiplier or 1.0))
            animation.sync.sensitivity = min(1.0, max(0.0, float(animation.sync.sensitivity)))
            animation.sync.intensity = min(1.0, max(0.0, float(animation.sync.intensity)))
            animation.sync.cooldown = max(0.0, float(animation.sync.cooldown))
            animation.sync.beats_per_cycle = max(0.0, float(animation.sync.beats_per_cycle))
            if animation.loop not in LOOP_MODES:
                animation.loop = "loop"
            if animation.sync.grid not in GRIDS:
                animation.sync.grid = "beat"
            if animation.base and animation.trigger:
                animation.base = False
            if animation.base and not animation.trigger:
                if base_named:
                    animation.base = False
                else:
                    base_named = True
        if not base_named:
            for animation in self.animations.values():
                if not animation.trigger:
                    animation.base = True
                    break

    def to_dict(self) -> dict:
        instruments = {
            animation.trigger: name
            for name, animation in self.animations.items()
            if animation.trigger
        }
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "bpm": float(self.bpm),
            "spritesheet": self.sheet.to_dict(),
            "frames": [item.to_dict() for item in self.frames],
            "layers": [{"id": layer, "name": layer.replace("_", " ")} for layer in self.layer_ids()],
            "instruments": instruments,
            "animations": {name: animation.to_dict() for name, animation in self.animations.items()},
        }

    def layer_ids(self) -> list[str]:
        layers = list(LAYERS)
        for animation in self.animations.values():
            if animation.layer not in layers:
                layers.append(animation.layer)
        return layers

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent) + "\n"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        self.root = str(path.parent)

    @classmethod
    def from_dict(cls, data: dict, root: str = "") -> SpriteProject:
        data = data or {}
        if isinstance(data.get("frames"), dict):
            return project_from_frame_map(data, root)
        sheet = SheetSpec.from_dict(data.get("spritesheet", data.get("sheet", {})))
        raw_frames = data.get("frames")
        if raw_frames:
            frames = [FrameInfo.from_dict(item) if isinstance(item, dict) else FrameInfo(int(item), f"Frame {item}", int(item)) for item in raw_frames]
        else:
            frames = [FrameInfo(index, f"Frame {index}", index) for index in range(sheet.count())]
        animations = {}
        for name, payload in (data.get("animations") or {}).items():
            animations[str(name)] = Animation.from_dict(str(name), payload)
        project = cls(
            name=str(data.get("name", "character")),
            bpm=float(data.get("bpm", data.get("sync", {}).get("bpm", 120.0)) or 120.0),
            sheet=sheet,
            frames=frames,
            animations=animations,
            root=root,
        )
        project.normalize()
        return project

    @classmethod
    def load(cls, path: Path) -> SpriteProject:
        path = Path(path)
        project = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), root=str(path.parent))
        return project


def unique_name(name: str, taken: dict) -> str:
    if name not in taken:
        return name
    number = 2
    while f"{name}_{number}" in taken:
        number += 1
    return f"{name}_{number}"


def _default_step(animation: Animation) -> float:
    if animation.sync.beats_per_cycle > 0 and animation.frames:
        return animation.sync.beats_per_cycle / len(animation.frames)
    return GRID_BEATS.get(animation.sync.grid, 1.0)


def _fit_durations(durations: list[float], frames: list[int]) -> list[float]:
    if not frames:
        return []
    if not durations:
        return []
    cleaned = [max(0.01, float(duration)) for duration in durations[: len(frames)]]
    if len(cleaned) < len(frames):
        pad = cleaned[-1] if cleaned else 1.0
        cleaned.extend([pad] * (len(frames) - len(cleaned)))
    return cleaned


def _sequence_durations(animation: Animation) -> list[float]:
    if len(animation.durations) == len(animation.frames) and animation.frames:
        return [max(0.01, float(duration)) for duration in animation.durations]
    if not animation.frames:
        return []
    step = _default_step(animation)
    return [step] * len(animation.frames)


def guess_frame_size(width: int, height: int) -> tuple[int, int, int, int]:
    """A square cell size that cuts the image into a sensible grid."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return 1, 1, 1, 1
    best = (width, height, 1, 1)
    best_score = 10**9
    limit = min(width, height)
    for size in range(16, limit + 1):
        if width % size or height % size:
            continue
        columns, rows = width // size, height // size
        count = columns * rows
        if not 1 <= count <= 512:
            continue
        score = abs(count - 96) + (0 if columns >= rows else 40)
        if score < best_score:
            best_score = score
            best = (size, size, columns, rows)
    return best


def _frame_key(value) -> int:
    if isinstance(value, str) and value.startswith("frame_"):
        return int(value.split("_")[-1])
    return int(value)


def project_from_frame_map(data: dict, root: str = "") -> SpriteProject:
    """Load a sprite-map file. The whole sheet is not one animation."""
    data = data or {}
    grid = data.get("grid") or {}
    image = data.get("image") or {}
    raw = data.get("frames") or {}
    frames: list[FrameInfo] = []
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        index = int(payload.get("sheet_index", _frame_key(key)))
        rect = payload.get("rect") or [0, 0, int(grid.get("cell_width") or 1), int(grid.get("cell_height") or 1)]
        content = payload.get("content_rect")
        frames.append(
            FrameInfo(
                id=index,
                name=f"{index:03d}",
                source=index,
                category=str(payload.get("category") or "unknown"),
                confidence=str(payload.get("confidence") or ""),
                note=str(payload.get("note") or ""),
                rect=tuple(int(value) for value in rect),
                content_rect=tuple(int(value) for value in content) if content else None,
            )
        )
    frames.sort(key=lambda item: item.id)
    cell_w = int(grid.get("cell_width") or (frames[0].rect[2] if frames and frames[0].rect else 1))
    cell_h = int(grid.get("cell_height") or (frames[0].rect[3] if frames and frames[0].rect else 1))
    sheet = SheetSpec(
        file=str(data.get("source") or ""),
        frame_width=cell_w,
        frame_height=cell_h,
        columns=max(1, int(grid.get("columns") or 1)),
        rows=max(1, int(grid.get("rows") or 1)),
        spacing_x=int(grid.get("spacing_x") or 0),
        spacing_y=int(grid.get("spacing_y") or 0),
        margin_x=int(grid.get("margin_x") or 0),
        margin_y=int(grid.get("margin_y") or 0),
        image_width=int(image.get("width") or 0),
        image_height=int(image.get("height") or 0),
    )
    first = frames[0].id if frames else 0
    animations = {
        "standing": Animation(
            name="standing",
            layer="body",
            frames=[first],
            durations=[1.0],
            loop="once",
            base=True,
            sync=Sync("beat", "beat", 1.0, 1, 1, 0.5, 0),
            note="Rest pose. One frame. This is not a dance cycle.",
        )
    }
    for name, payload in (data.get("animations") or {}).items():
        if not isinstance(payload, dict) or not payload.get("frames") or payload.get("sequence") is False:
            continue
        ids = [_frame_key(value) for value in payload["frames"]]
        animations[str(name)] = Animation(
            name=str(name),
            layer=str(payload.get("layer") or "body"),
            frames=ids,
            durations=[],
            loop=str(payload.get("loop") or "once"),
            base=False,
            sync=Sync("beat", "beat", float(len(ids) or 1), 1, 1, 0.5, 0),
            note=str(payload.get("note") or ""),
        )
    project = SpriteProject(
        name=str(data.get("character") or data.get("name") or "character"),
        bpm=float(data.get("bpm") or 120),
        sheet=sheet,
        frames=frames,
        animations=animations,
        root=root,
    )
    project.normalize()
    return project


def validate(data: dict) -> list[str]:
    """Problems in a project dict. An empty list means the file can be used."""
    if isinstance(data, dict) and isinstance(data.get("frames"), dict):
        return _validate_frame_map(data)
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["The project is not a JSON object."]
    version = data.get("version", VERSION)
    if version != VERSION and "animations" not in data:
        errors.append(f"Unsupported version {version}.")
    sheet = data.get("spritesheet", data.get("sheet", {}))
    if not isinstance(sheet, dict):
        errors.append("spritesheet must be an object.")
        sheet = {}
    width = int(sheet.get("frame_width", 0) or 0)
    height = int(sheet.get("frame_height", 0) or 0)
    if width <= 0 or height <= 0:
        errors.append("Frame width and height must be greater than zero.")
    columns = int(sheet.get("columns", 1) or 1)
    rows = int(sheet.get("rows", 1) or 1)
    if columns <= 0 or rows <= 0:
        errors.append("Columns and rows must be greater than zero.")
    image_w = int(sheet.get("image_width", 0) or 0)
    image_h = int(sheet.get("image_height", 0) or 0)
    if image_w and image_h and width > 0 and height > 0:
        spec = SheetSpec.from_dict(sheet)
        if spec.count():
            last = spec.rect(spec.count() - 1)
            if last[0] + last[2] > image_w or last[1] + last[3] > image_h:
                errors.append("The grid extends past the sprite sheet.")
    frames = data.get("frames", [])
    ids = set()
    if isinstance(frames, list):
        for item in frames:
            if isinstance(item, dict):
                frame_id = item.get("id")
                if frame_id in ids:
                    errors.append(f"Frame id {frame_id} is repeated.")
                ids.add(frame_id)
    animations = data.get("animations", {})
    if not isinstance(animations, dict) or not animations:
        errors.append("At least one animation is required.")
        animations = {}
    bases = 0
    for name, animation in animations.items():
        if not isinstance(animation, dict):
            errors.append(f"Animation {name} is not an object.")
            continue
        if not str(animation.get("layer", "")).strip():
            errors.append(f"Animation {name} has no layer.")
        loop = animation.get("loop", "loop")
        if loop not in (True, False, *LOOP_MODES):
            errors.append(f"Animation {name} has an unknown loop mode.")
        trigger = str(animation.get("trigger", "") or "")
        if trigger and trigger not in TRIGGERS:
            errors.append(f"Animation {name} has an unknown trigger {trigger}.")
        sync = animation.get("sync") or {}
        multiplier = sync.get("speed_multiplier", animation.get("speed_multiplier", 1.0))
        try:
            if float(multiplier) <= 0:
                errors.append(f"Animation {name} has a speed multiplier that is not positive.")
        except (TypeError, ValueError):
            errors.append(f"Animation {name} has a speed multiplier that is not a number.")
        for frame in animation.get("frames", []):
            if ids and frame not in ids:
                errors.append(f"Animation {name} uses missing frame {frame}.")
                break
        durations = animation.get("durations", [])
        sequence = animation.get("frames", [])
        if durations and len(durations) != len(sequence):
            errors.append(f"Animation {name} durations do not match its frames.")
        if animation.get("base") and not trigger:
            bases += 1
    if bases > 1:
        errors.append("Only one animation can be the base dance.")
    return errors


def _validate_frame_map(data: dict) -> list[str]:
    errors: list[str] = []
    image = data.get("image") or {}
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    frames = data.get("frames") or {}
    if not frames:
        return ["The frame map has no frames."]
    rects: list[tuple[str, int, int, int, int]] = []
    for key, payload in frames.items():
        if not isinstance(payload, dict):
            errors.append(f"{key} is not an object.")
            continue
        rect = payload.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            errors.append(f"{key} needs a rect of x, y, width, height.")
            continue
        x, y, rect_w, rect_h = (int(value) for value in rect)
        if rect_w <= 0 or rect_h <= 0:
            errors.append(f"{key} has an empty rect.")
        if width and height and (x < 0 or y < 0 or x + rect_w > width or y + rect_h > height):
            errors.append(f"{key} is outside the image.")
        category = str(payload.get("category") or "unknown")
        if category not in POSE_CATEGORIES:
            errors.append(f"{key} has a category that the editor cannot store.")
        content = payload.get("content_rect")
        if content:
            cx, cy, cw, ch = (int(value) for value in content)
            if cx < x or cy < y or cx + cw > x + rect_w or cy + ch > y + rect_h:
                errors.append(f"{key} content is not inside its rect.")
        rects.append((str(key), x, y, rect_w, rect_h))
    for index, (key, x, y, rect_w, rect_h) in enumerate(rects):
        for other, ox, oy, ow, oh in rects[index + 1 :]:
            if x < ox + ow and ox < x + rect_w and y < oy + oh and oy < y + rect_h:
                errors.append(f"{key} overlaps {other}.")
                return errors
    return errors


def _cursor(beats: float, frames: list[int], durations: list[float], loop: str) -> int | None:
    count = len(frames)
    if count == 0:
        return None
    if count == 1 or loop == "once" and beats < 0:
        return frames[0]
    if loop == "pingpong" and count > 1:
        order = list(range(count)) + list(range(count - 2, 0, -1))
        spans = list(durations) + list(durations[-2:0:-1])
    elif loop == "reverse":
        order = list(range(count - 1, -1, -1))
        spans = list(reversed(durations))
    else:
        order = list(range(count))
        spans = list(durations)
    total = sum(spans)
    if total <= 0:
        return frames[order[0]]
    if loop == "once":
        if beats >= total:
            return frames[order[-1]]
        local = max(0.0, beats)
    else:
        local = beats % total
    walked = 0.0
    for index, span in zip(order, spans):
        if local < walked + span:
            return frames[index]
        walked += span
    return frames[order[-1]]


def frame_id_at(animation: Animation, seconds: float, bpm: float) -> int | None:
    """The frame this animation shows. BPM and its own multiplier set the speed."""
    if not animation.frames:
        return None
    multiplier = max(0.05, float(animation.sync.speed_multiplier))
    beats = max(0.0, float(seconds)) * (max(float(bpm), 1.0) / 60.0) * multiplier
    return _cursor(beats, list(animation.frames), _sequence_durations(animation), animation.loop)


def base_frame_id(project: SpriteProject, seconds: float, bpm: float | None = None) -> int | None:
    """The dance frame. Drum events are not an input, so they cannot change it."""
    animation = project.base_animation()
    if animation is None:
        return None
    return frame_id_at(animation, seconds, project.bpm if bpm is None else bpm)


def overlay_frame(
    project: SpriteProject,
    seconds: float,
    events: list[dict],
    bpm: float | None = None,
) -> int | None:
    """The strongest drum animation still playing, or None to keep the base."""
    bpm = project.bpm if bpm is None else bpm
    accepted: dict[str, float] = {}
    best: tuple[int, int] | None = None
    for event in events:
        kind = str(event.get("kind", ""))
        animation = project.animation_for_trigger(kind)
        if animation is None or not animation.frames:
            continue
        when = float(event.get("time", 0.0))
        strength = float(event.get("confidence", 1.0))
        if strength + animation.sync.sensitivity < 1.0:
            continue
        previous = accepted.get(kind)
        if previous is not None and when - previous < animation.sync.cooldown:
            continue
        accepted[kind] = when
        if when > seconds:
            continue
        local = seconds - when
        durations = _sequence_durations(animation)
        span = sum(durations)
        multiplier = max(0.05, animation.sync.speed_multiplier)
        length = span * 60.0 / max(float(bpm), 1.0) / multiplier
        if local > length:
            continue
        frame = frame_id_at(animation, local, bpm)
        if frame is None:
            continue
        rank = OVERLAY_PRIORITY.index(kind) if kind in OVERLAY_PRIORITY else len(OVERLAY_PRIORITY)
        if best is None or rank < best[0]:
            best = (rank, frame)
    return None if best is None else best[1]


def shown_frame_id(
    project: SpriteProject,
    seconds: float,
    events: list[dict] | None = None,
    bpm: float | None = None,
) -> int | None:
    """Base frame, replaced only while an overlay animation is playing."""
    overlay = overlay_frame(project, seconds, events or [], bpm)
    if overlay is not None:
        return overlay
    return base_frame_id(project, seconds, bpm)


def display_source(project: SpriteProject, clock_frame: int, accents: dict | None = None) -> int:
    """Sheet cell for a dancer clock frame. Accents do not move ``clock_frame``."""
    accents = accents or {}
    for kind in ("kick", "snare", "hat", "bass", "perc"):
        level = float(accents.get(kind, 0.0) or 0.0)
        if level < 0.45:
            continue
        animation = project.animation_for_trigger(kind)
        if animation is None or not animation.frames:
            continue
        # Bass stays a sway unless this animation asks to paint its frames.
        if kind == "bass" and animation.paint != "frame":
            continue
        count = len(animation.frames)
        if kind == "bass":
            index = (int(clock_frame) // 12) % count
        else:
            index = min(count - 1, int((1.0 - min(1.0, level)) * count))
        return project.source_of(animation.frames[index])
    animation = project.base_animation()
    if animation and animation.frames:
        return project.source_of(animation.frames[int(clock_frame) % len(animation.frames)])
    return int(clock_frame)


class EditorSession:
    """Undo history for one project. Snapshots are the JSON objects."""

    def __init__(self, project: SpriteProject):
        self.project = project
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.clipboard: list[int] = []

    def change(self, fn) -> None:
        self.undo_stack.append(self.project.to_dict())
        if len(self.undo_stack) > 50:
            del self.undo_stack[0]
        self.redo_stack.clear()
        fn(self.project)
        self.project.normalize()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.project.to_dict())
        self.project = SpriteProject.from_dict(self.undo_stack.pop(), root=self.project.root)
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.project.to_dict())
        self.project = SpriteProject.from_dict(self.redo_stack.pop(), root=self.project.root)
        return True


def heart_dancer_project() -> SpriteProject:
    """The authored heart dance as an editable project. The sheet is not redrawn."""
    from soniclab.core.dancer import BLOCKS_PATH, SHEET_PATH, load_blocks

    blocks = load_blocks()
    smooth = blocks["smooth"]
    counts = ("1", "e", "&", "a")
    frames = []
    for item in smooth["frames"]:
        beat = int(item["beat"])
        step = int(item["step_in_beat"])
        count = counts[min(3, step // 3)]
        extra = step % 3
        name = f"Beat {beat} {count}" if extra == 0 else f"Beat {beat} {count} +{extra}"
        frames.append(FrameInfo(int(item["frame"]), name, int(item["frame"])))
    sheet = SheetSpec(
        file=SHEET_PATH.name,
        frame_width=int(smooth["frame_px"]),
        frame_height=int(smooth["frame_px"]),
        columns=int(smooth["sheet"]["columns"]),
        rows=int(smooth["sheet"]["rows"]),
        image_width=int(smooth["frame_px"]) * int(smooth["sheet"]["columns"]),
        image_height=int(smooth["frame_px"]) * int(smooth["sheet"]["rows"]),
    )
    ids = [item.id for item in frames]

    def clock(name: str, layer: str, sequence: list[int], beats: float, grid: str, base: bool = False) -> Animation:
        step = beats / max(1, len(sequence))
        return Animation(
            name=name,
            layer=layer,
            frames=sequence,
            durations=[step] * len(sequence),
            fps=float(blocks["meta"]["bpm"]) / 60.0 / step,
            loop="loop" if base else "once",
            base=base,
            trigger="",
            sync=Sync("beat" if base else "downbeat", grid, beats, 1.0, 1.0, 0.5, 0.0),
        )

    def overlay(name: str, layer: str, sequence: list[int], trigger: str, cooldown: float, loop: str = "once") -> Animation:
        step = GRID_BEATS["sixteenth"] / 3.0
        return Animation(
            name=name,
            layer=layer,
            frames=sequence,
            durations=[step] * len(sequence),
            fps=25.0,
            loop=loop,
            base=False,
            trigger=trigger,
            sync=Sync(trigger, "sixteenth", step * len(sequence), 1.0, 1.0, 0.5, cooldown),
        )

    beat_hits = [item.id for item in frames if "+" not in item.name and item.name.rsplit(" ", 1)[-1] == "1"]
    project = SpriteProject(
        name="heart_dancer",
        bpm=float(blocks["meta"]["bpm"]),
        sheet=sheet,
        frames=frames,
        root=str(BLOCKS_PATH.parent),
        animations={
            "body_bounce": clock("body_bounce", "body", ids, 8.0, "sixteenth", base=True),
            "downbeat_transition": overlay("downbeat_transition", "beat_accents", [0, 48], "downbeat", 0.4),
            "beat_step": overlay("beat_step", "body", beat_hits, "beat", 0.2),
            "kick_drop": overlay("kick_drop", "body", [0, 1, 2], "kick", 0.07),
            "snare_clap": overlay("snare_clap", "left_arm", [48, 49, 50], "snare", 0.08),
            "hat_gesture": overlay("hat_gesture", "right_arm", [36, 37, 38], "hat", 0.04),
            "bass_sway": overlay("bass_sway", "body", [60, 61, 62, 63], "bass", 0.1, loop="pingpong"),
            "perc_accent": overlay("perc_accent", "instrument", [84, 85], "perc", 0.05),
        },
    )
    # downbeat and beat are clock language in the table, stored as triggers so
    # they can be previewed. They are not overlays on the dancer clock.
    project.animations["downbeat_transition"].trigger = "downbeat"
    project.animations["beat_step"].trigger = "beat"
    pixel = ids[::3]
    project.animations["body_pixel"] = Animation(
        name="body_pixel",
        layer="body",
        frames=pixel,
        durations=[8.0 / len(pixel)] * len(pixel),
        fps=125.0 / 60.0 / (8.0 / len(pixel)),
        loop="loop",
        base=False,
        trigger="",
        sync=Sync("beat", "sixteenth", 8.0, 1.0, 1.0, 0.5, 0.0),
    )
    project.normalize()
    # normalize would mark body_pixel as a second clock animation and clear
    # base on it already. body_bounce stays the one base. Beat and downbeat
    # keep their triggers, so they are not elected base.
    return project
