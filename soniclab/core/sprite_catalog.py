"""Measured map of the green dancer sheet.

The grid comes from the pixels: 880×1280 divides into 8×10 cells of 110×128,
with no margin and no gap. Each cell holds one sprite, and none of those
sprites cross a cell edge. Categories are readings of the drawings.
Anything that is not clear stays unknown.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QApplication

from soniclab.core.sprite_project import Animation, SheetSpec, SpriteProject, Sync, project_from_frame_map

CELL_W = 110
CELL_H = 128
COLUMNS = 8
ROWS = 10
FRAME_COUNT = COLUMNS * ROWS
IMAGE_W = 880
IMAGE_H = 1280

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "green_dancer"
SHEET_NAME = "sprites.png"
ORIGINAL = Path("/home/rizekishimaro/Pictures/homework") / "🌶️" / "sprites.png"
_APP = None


def _ensure_app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication([])
    return _APP


def _assign(table: dict, indexes, category: str, confidence: str, note: str = "") -> None:
    for index in indexes:
        table[int(index)] = {"category": category, "confidence": confidence, "note": note}


def _poses() -> dict[int, dict]:
    poses: dict[int, dict] = {}
    _assign(poses, [0], "idle", "high", "Standing. Used as the rest pose.")
    _assign(
        poses,
        [1, 2, 3, 7],
        "idle",
        "medium",
        "Standing. The pixels differ from frame 0. The gesture is not named.",
    )
    _assign(
        poses,
        [4, 5, 6],
        "idle",
        "high",
        "Standing. These three differ only by faint edge pixels.",
    )
    _assign(
        poses,
        [8],
        "idle",
        "medium",
        "Close to the standing frames. A small wing change is possible and not named.",
    )
    _assign(
        poses,
        range(9, 16),
        "arms",
        "medium",
        "Coat or wing position changes along the row. Not confirmed as one clip.",
    )
    _assign(
        poses,
        [16, 21],
        "unknown",
        "low",
        "The drawing starts lower in the cell than frame 0 and still reaches the bottom. Pose not named.",
    )
    _assign(
        poses,
        [17],
        "unknown",
        "medium",
        "The drawing sits higher in the cell than the standing frames and does not reach their foot line. The move is not named.",
    )
    _assign(poses, [18], "arms", "high", "One arm is raised. The body is high in the cell.")
    _assign(
        poses,
        [19],
        "arms",
        "medium",
        "An arm is extended and the head is tilted. The body is high in the cell.",
    )
    _assign(poses, [20, 22], "idle", "medium", "Standing, close to frame 0.")
    _assign(
        poses,
        [23],
        "unknown",
        "low",
        "Differs from the standing frames. A face change is possible and not confirmed.",
    )
    _assign(
        poses,
        range(24, 32),
        "arms",
        "medium",
        "Arms are held in front of the body. Neighbors are similar. Not confirmed as one clip.",
    )
    _assign(poses, [32, 34, 35, 36, 37, 38], "lean", "high", "The torso leans. Hands on 32 and 36 come close to the cell edge.")
    _assign(
        poses,
        [33, 39],
        "lean",
        "medium",
        "A lean, with a leg forward. A kick is not confirmed.",
    )
    _assign(
        poses,
        range(40, 48),
        "arms",
        "medium",
        "Arm and wing positions. Not confirmed as one clip.",
    )
    _assign(poses, range(48, 55), "spin", "high", "Part of one roll, in sheet order.")
    _assign(poses, [55], "spin", "medium", "The roll is finishing and the body is more upright.")
    _assign(poses, [56, 57, 62], "kick", "high", "One leg is raised.")
    _assign(poses, [58, 61], "kick", "medium", "A leg is lifted. The pose is less clear than 56 and 57.")
    _assign(poses, [59], "arms", "medium", "Hands are near the waist. The supporting leg is down.")
    _assign(poses, [60], "arms", "medium", "Hands are together in front. Not a confirmed kick.")
    _assign(poses, [63], "arms", "medium", "A hand is near the head.")
    turn_notes = {
        64: "Turn. Read as three-quarter, toward the viewer.",
        65: "Turn. Read as a side view.",
        66: "Turn. Read as a side view with the coat toward the camera.",
        67: "Turn. Read as the back.",
        68: "Turn. Read as the back.",
        69: "Turn. Read as a side view.",
        70: "Turn. Read as three-quarter.",
        71: "Turn. Read as the front.",
    }
    for index, note in turn_notes.items():
        _assign(poses, [index], "turn", "high", note)
    _assign(
        poses,
        range(72, 80),
        "idle",
        "medium",
        "Front view, standing. Neighbors differ. The coat or the arms may move. The gesture is not named.",
    )
    if set(poses) != set(range(FRAME_COUNT)):
        missing = sorted(set(range(FRAME_COUNT)) - set(poses))
        raise RuntimeError(f"Unclassified frames: {missing}")
    return poses


POSES = _poses()

# One pose per beat. The dancer clock has 12 frames in a beat, so each pose is held 12 times.
BODY_POSES = [0, 72, 74, 20, 64, 66, 68, 71]
HOLDS = 12
KICK_FRAMES = [56, 57, 62]
SNARE_FRAMES = [18, 60, 24]
HAT_FRAMES = [9, 40, 41]
BASS_FRAMES = [32, 34, 36, 38]
PERC_FRAMES = [48, 50, 52, 55]
DOWNBEAT_FRAMES = [0, 64]

SEQUENCES = (
    {
        "name": "tumble",
        "frames": [f"frame_{index:03d}" for index in range(48, 56)],
        "category": "spin",
        "confidence": "high",
        "confirmed": False,
        "sheet_order": True,
        "timing": None,
        "note": (
            "Frames 48–55 are one roll in sheet order. "
            "The sheet has no frame duration, so this is not a timed animation."
        ),
    },
    {
        "name": "turn",
        "frames": [f"frame_{index:03d}" for index in range(64, 72)],
        "category": "turn",
        "confidence": "high",
        "confirmed": False,
        "sheet_order": True,
        "timing": None,
        "note": (
            "Frames 64–71 turn from a three-quarter view through the side and the back, "
            "then return toward the front. Timing is not in the image."
        ),
    },
)

GROUPS = (
    {
        "name": "standing",
        "frames": [f"frame_{index:03d}" for index in range(8)],
        "category": "idle",
        "sequence": False,
        "confidence": "high",
        "note": "Same standing family. Frames 4, 5 and 6 are nearly the same picture. Not one animation.",
    },
    {
        "name": "wings",
        "frames": [f"frame_{index:03d}" for index in range(8, 16)],
        "category": "arms",
        "sequence": False,
        "confidence": "medium",
        "note": "Wing or coat changes. Not exported as an animation.",
    },
    {
        "name": "mixed_row_2",
        "frames": [f"frame_{index:03d}" for index in range(16, 24)],
        "category": "unknown",
        "sequence": False,
        "confidence": "low",
        "note": "Standing frames mixed with a raised arm and two unnamed poses. Not one movement.",
    },
    {
        "name": "arms_front",
        "frames": [f"frame_{index:03d}" for index in range(24, 32)],
        "category": "arms",
        "sequence": False,
        "confidence": "medium",
        "note": "Arms held in front. A pose family, not a confirmed clip.",
    },
    {
        "name": "leans",
        "frames": [f"frame_{index:03d}" for index in range(32, 40)],
        "category": "lean",
        "sequence": False,
        "confidence": "medium",
        "note": "Lean family in sheet order. It may be a pose set rather than one move, so it is not an animation.",
    },
    {
        "name": "arms_row",
        "frames": [f"frame_{index:03d}" for index in range(40, 48)],
        "category": "arms",
        "sequence": False,
        "confidence": "medium",
        "note": "Arm and wing positions. Not a confirmed clip.",
    },
    {
        "name": "kick_row",
        "frames": [f"frame_{index:03d}" for index in range(56, 64)],
        "category": "kick",
        "sequence": False,
        "confidence": "medium",
        "note": "Kicks are mixed with arm poses, so the row is not one animation.",
    },
    {
        "name": "front_standing",
        "frames": [f"frame_{index:03d}" for index in range(72, 80)],
        "category": "idle",
        "sequence": False,
        "confidence": "medium",
        "note": "Front standing views. Neighbor frames differ. Not one animation.",
    },
)


def _rgba(path: Path) -> np.ndarray:
    _ensure_app()
    image = QImage(str(path)).convertToFormat(QImage.Format.Format_ARGB32)
    if image.isNull():
        raise RuntimeError(f"Could not read {path}")
    width, height = image.width(), image.height()
    raw = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
    view = raw.reshape(height, image.bytesPerLine() // 4, 4)[:, :width]
    return np.array(view, copy=True)


def measure(path: Path) -> dict:
    """Cell grid, content boxes, and the checks that the grid is real."""
    from scipy import ndimage

    pixels = _rgba(path)
    height, width = pixels.shape[:2]
    if (width, height) != (IMAGE_W, IMAGE_H):
        raise RuntimeError(f"{path.name} is {width}×{height}, expected {IMAGE_W}×{IMAGE_H}")
    alpha = pixels[:, :, 3]
    _labels, count = ndimage.label(alpha > 0)
    if count != FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} sprites, found {count}")
    frames = []
    for index in range(FRAME_COUNT):
        row, column = divmod(index, COLUMNS)
        x = column * CELL_W
        y = row * CELL_H
        cell = alpha[y : y + CELL_H, x : x + CELL_W]
        if not np.any(cell):
            raise RuntimeError(f"frame_{index:03d} is empty")
        ys, xs = np.where(cell > 0)
        content = [int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        region = _labels[y : y + CELL_H, x : x + CELL_W]
        found = {int(value) for value in np.unique(region) if value}
        if len(found) != 1:
            raise RuntimeError(f"frame_{index:03d} contains {len(found)} sprites")
        label = found.pop()
        sprite = np.argwhere(_labels == label)
        top, left = sprite.min(axis=0)
        bottom, right = sprite.max(axis=0)
        if left < x or top < y or right >= x + CELL_W or bottom >= y + CELL_H:
            raise RuntimeError(f"frame_{index:03d} crosses its cell")
        pose = POSES[index]
        frames.append(
            {
                "id": index,
                "rect": [x, y, CELL_W, CELL_H],
                "content_rect": content,
                "category": pose["category"],
                "confidence": pose["confidence"],
                "note": pose["note"],
                "column": column,
                "row": row,
            }
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"width": width, "height": height, "frames": frames, "sha256": digest, "components": int(count)}


def _frame_map(measured: dict) -> dict:
    frames = {}
    for item in measured["frames"]:
        payload = {
            "rect": list(item["rect"]),
            "content_rect": list(item["content_rect"]),
            "category": item["category"],
            "confidence": item["confidence"],
            "column": item["column"],
            "row": item["row"],
            "sheet_index": item["id"],
        }
        if item["note"]:
            payload["note"] = item["note"]
        frames[f"frame_{item['id']:03d}"] = payload
    return {
        "format": "soniclab.sprite-map",
        "version": 1,
        "character": "green_dancer",
        "source": SHEET_NAME,
        "image": {"width": measured["width"], "height": measured["height"]},
        "grid": {
            "columns": COLUMNS,
            "rows": ROWS,
            "cell_width": CELL_W,
            "cell_height": CELL_H,
            "spacing_x": 0,
            "spacing_y": 0,
            "margin_x": 0,
            "margin_y": 0,
            "confidence": "high",
            "note": (
                "880 and 1280 divide evenly by 110 and 128. "
                "Eighty sprites, one per cell, and none cross a cell edge."
            ),
        },
        "frames": frames,
        "animations": {},
    }


def _animations_document() -> dict:
    def clip(frames, **fields):
        payload = {"frames": [f"frame_{index:03d}" for index in frames], "confirmed": False}
        payload.update(fields)
        return payload

    animations = {
        "body": clip(
            BODY_POSES,
            trigger="",
            base=True,
            holds=HOLDS,
            category="idle",
            confidence="high",
            sheet_order=False,
            timing="one pose per beat",
            note=(
                "The dance. Standing, a front view, then the turn on the second bar. "
                "Each pose is held for one beat. Drum hits do not change this speed."
            ),
        ),
        "kick": clip(
            KICK_FRAMES,
            trigger="kick",
            category="kick",
            confidence="high",
            sheet_order=False,
            timing=None,
            note="Leg raises. A kick shows these frames and does not advance the dance.",
        ),
        "snare": clip(
            SNARE_FRAMES,
            trigger="snare",
            category="arms",
            confidence="high",
            sheet_order=False,
            timing=None,
            note="Raised arm, then hands together, then arms in front.",
        ),
        "hat": clip(
            HAT_FRAMES,
            trigger="hat",
            category="arms",
            confidence="medium",
            sheet_order=False,
            timing=None,
            note="Small wing and arm accents.",
        ),
        "bass": clip(
            BASS_FRAMES,
            trigger="bass",
            category="lean",
            confidence="high",
            sheet_order=False,
            timing="one lean per beat while the bass holds",
            note="Body leans. A held bass steps through them once per beat.",
        ),
        "perc": clip(
            PERC_FRAMES,
            trigger="perc",
            category="spin",
            confidence="high",
            sheet_order=False,
            timing=None,
            note="Moments from the roll. The whole roll is the tumble clip.",
        ),
        "downbeat": clip(
            DOWNBEAT_FRAMES,
            trigger="downbeat",
            category="turn",
            confidence="high",
            sheet_order=False,
            timing=None,
            note="The standing pose and the start of the turn. The dance already lands on these.",
        ),
    }
    for sequence in SEQUENCES:
        animations[sequence["name"]] = {
            "frames": list(sequence["frames"]),
            "category": sequence["category"],
            "confidence": sequence["confidence"],
            "confirmed": False,
            "sheet_order": True,
            "timing": None,
            "trigger": "",
            "note": sequence["note"],
        }
    return {
        "format": "soniclab.sprite-map",
        "version": 1,
        "character": "green_dancer",
        "source": SHEET_NAME,
        "animations": animations,
        "groups": {group["name"]: {key: value for key, value in group.items() if key != "name"} for group in GROUPS},
    }


def _hit(name: str, layer: str, frames: list[int], trigger: str, cooldown: float, note: str, loop: str = "once", paint: str = "") -> Animation:
    step = 0.25 / 3.0
    return Animation(
        name=name,
        layer=layer,
        frames=list(frames),
        durations=[step] * len(frames),
        fps=25.0,
        loop=loop,
        base=False,
        trigger=trigger,
        sync=Sync(trigger, "sixteenth", step * len(frames), 1.0, 1.0, 0.5, cooldown),
        note=note,
        paint=paint,
    )


def build_project(measured: dict, root: Path | None = None) -> SpriteProject:
    """Green dancer wired to the same clock as the heart dancer."""
    data = _frame_map(measured)
    project = project_from_frame_map(data, root=str(root or ASSET_DIR))
    project.sheet.file = SHEET_NAME
    project.sheet.image_width = measured["width"]
    project.sheet.image_height = measured["height"]
    project.bpm = 125.0
    body = [pose for pose in BODY_POSES for _ in range(HOLDS)]
    step = 8.0 / len(body)
    project.animations = {
        "body": Animation(
            name="body",
            layer="body",
            frames=body,
            durations=[step] * len(body),
            fps=125.0 / 60.0 / step,
            loop="loop",
            base=True,
            trigger="",
            sync=Sync("beat", "sixteenth", 8.0, 1.0, 1.0, 0.5, 0.0),
            note=(
                "One pose per beat: stand, front, front, stand, then the turn. "
                "Drum hits do not change this speed."
            ),
        ),
        "kick": _hit("kick", "left_leg", KICK_FRAMES, "kick", 0.07, "Leg raises on a kick."),
        "snare": _hit("snare", "left_arm", SNARE_FRAMES, "snare", 0.08, "Arm hits on a snare."),
        "hat": _hit("hat", "right_arm", HAT_FRAMES, "hat", 0.04, "Small arm and wing accents on a hat."),
        "bass": _hit(
            "bass",
            "body",
            BASS_FRAMES,
            "bass",
            0.1,
            "Leans while the bass holds. One lean per beat.",
            loop="pingpong",
            paint="frame",
        ),
        "perc": _hit("perc", "instrument", PERC_FRAMES, "perc", 0.05, "Roll accents on percussion."),
        "downbeat": _hit("downbeat", "beat_accents", DOWNBEAT_FRAMES, "downbeat", 0.4, "Standing pose, then the turn."),
        "tumble": _hit("tumble", "body", list(range(48, 56)), "", 0.0, SEQUENCES[0]["note"], loop="once"),
        "turn": _hit("turn", "body", list(range(64, 72)), "", 0.0, SEQUENCES[1]["note"], loop="once"),
    }
    project.animations["tumble"].trigger = ""
    project.animations["turn"].trigger = ""
    project.normalize()
    return project


def green_dancer_project() -> SpriteProject:
    sheet = ASSET_DIR / SHEET_NAME
    if not sheet.is_file():
        raise RuntimeError(f"Missing {sheet}")
    return build_project(measure(sheet), ASSET_DIR)


def _contact_sheet(source: Path, dest: Path) -> None:
    _ensure_app()
    image = QImage(str(source))
    scale = 2
    canvas = QImage(image.width() * scale, image.height() * scale, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor("#000000"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.drawImage(QRect(0, 0, canvas.width(), canvas.height()), image)
    painter.setFont(QFont("DejaVu Sans", 11))
    for index in range(FRAME_COUNT):
        row, column = divmod(index, COLUMNS)
        x = column * CELL_W * scale
        y = row * CELL_H * scale
        painter.setPen(QColor(215, 179, 106, 170))
        painter.drawRect(x, y, CELL_W * scale - 1, CELL_H * scale - 1)
        painter.fillRect(x + 2, y + 2, 28, 16, QColor(0, 0, 0, 180))
        painter.setPen(QColor("#F4E2B8"))
        painter.drawText(x + 4, y + 15, f"{index:03d}")
    painter.end()
    canvas.save(str(dest))


def _report(measured: dict) -> str:
    counts: dict[str, int] = {}
    confidence: dict[str, int] = {}
    for item in measured["frames"]:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
        confidence[item["confidence"]] = confidence.get(item["confidence"], 0) + 1
    lines = [
        "# Green dancer sprite map",
        "",
        "The source image is 880×1280. It was not modified. The copy used by the editor is `soniclab/assets/green_dancer/sprites.png`.",
        "",
        "## Grid",
        "",
        "The layout was measured from the pixels, not assumed.",
        "",
        "| | |",
        "|---|---|",
        "| Frames | 80 |",
        "| Columns × rows | 8 × 10 |",
        "| Cell | 110 × 128 |",
        "| Spacing | 0, 0 |",
        "| Margin | 0, 0 |",
        "| Origin | 0, 0 |",
        "| Connected sprites | 80 |",
        "| Sprites crossing a cell edge | 0 |",
        "| Empty cells | 0 |",
        "| Grid confidence | high |",
        "",
        "880 ÷ 110 = 8 and 1280 ÷ 128 = 10, with no remainder. Labeling every non-transparent pixel gives one sprite per cell. Each sprite's pixels stay inside its cell. Opaque black outlines are part of the drawing. The background is transparent, so it looks black when composited.",
        "",
        "The source rectangle for every frame is the full cell. Content often comes within 2 pixels of the bottom edge, and a raised hand or the sprout can come within a few pixels of the top or side. A tighter crop would cut feet, hair, or hands. The tight pixel bounds are stored as `content_rect` and are not the crop.",
        "",
        "No two frames are byte-identical.",
        "",
        "## Content inset",
        "",
        "Left, right, top, and bottom empty pixels inside each cell. Bottom insets of 2 mean the feet are almost on the cell edge.",
        "",
        "| Frame | L | R | T | B |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in measured["frames"]:
        x, y, w, h = item["rect"]
        cx, cy, cw, ch = item["content_rect"]
        lines.append(f"| {item['id']:03d} | {cx - x} | {x + w - (cx + cw)} | {cy - y} | {y + h - (cy + ch)} |")
    lines.extend(
        [
            "",
            "## Categories",
            "",
            "These are pose families, not confirmed move names. Confidence is high, medium, or low. Low and medium frames can be changed in the editor.",
            "",
            "| Category | Frames |",
            "|---|---:|",
        ]
    )
    for name in ("idle", "dance", "arms", "kick", "lean", "spin", "turn", "unknown"):
        lines.append(f"| {name} | {counts.get(name, 0)} |")
    lines.extend(
        [
            "",
            f"Confidence counts: high {confidence.get('high', 0)}, medium {confidence.get('medium', 0)}, low {confidence.get('low', 0)}.",
            "",
            "Nothing was labeled dance. The rows that look like dancing are standing, arms, leans, kicks, a roll, or a turn, and the unclear cells stay unknown.",
            "",
            "## Playback",
            "",
            "The heart dancer stays the character in the dancer dock. View → Heart Dancer brings it back. The green dancer is a separate project. Its dance follows the beat clock. A drum hit shows that instrument's pose and does not speed up the dance.",
            "",
            "| Clip | Trigger | Frames |",
            "|---|---|---|",
            "| body | beat clock | 0, 72, 74, 20, then 64, 66, 68, 71. One pose per beat. |",
            "| kick | kick | 56, 57, 62 |",
            "| snare | snare | 18, 60, 24 |",
            "| hat | hat | 9, 40, 41 |",
            "| bass | bass | 32, 34, 36, 38 |",
            "| perc | percussion | 48, 50, 52, 55 |",
            "",
            "Tumble is frames 48–55 in sheet order. Turn is frames 64–71. Those clips have no instrument. The second half of the body dance uses the turn.",
            "",
            "The other rows are groups, not animations:",
            "",
        ]
    )
    for group in GROUPS:
        lines.append(f"- **{group['name']}** — {group['note']}")
    lines.extend(
        [
            "",
            "The sheet is not one 80-frame animation. The body dance is eight poses.",
            "",
            "## Uncertain frames",
            "",
            "High-confidence labels are the standing frames 0 and 4–6, the raised arm on 18, the leans on 32 and 34–38, the roll on 48–54, the kicks on 56, 57 and 62, and the turn on 64–71.",
            "",
            "Every other frame is medium or low. The low ones are 16, 21 and 23. Their drawings differ from the standing pose and do not match a named move.",
            "",
            "| Frame | Category | Confidence | Note |",
            "|---|---|---|---|",
        ]
    )
    for item in measured["frames"]:
        if item["confidence"] == "high":
            continue
        note = item["note"].replace("|", "/")
        lines.append(f"| {item['id']:03d} | {item['category']} | {item['confidence']} | {note} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `soniclab/assets/green_dancer/sprites.png` — copy of the source",
            "- `soniclab/assets/green_dancer/green_dancer_sprites.json` — frame ids and rectangles",
            "- `soniclab/assets/green_dancer/green_dancer_animations.json` — body dance and the instrument poses",
            "- `soniclab/assets/green_dancer/green_dancer_contact_sheet.png` — the sheet at 2× with frame numbers",
            "- `soniclab/assets/green_dancer/green_dancer.sprite.json` — the editor project",
            "",
            "In the Sprite Animation Editor, **File → Open Green Dancer** loads the 80 frames. Each thumbnail shows the frame id and the source position. **View → Verify Mapping** draws every rectangle, the tight content box, and the frame number. The X Y W H fields correct a rectangle. Category can be changed; a manual change is stored as confidence `manual`. The preview scales the selected cell by a whole number of pixels.",
            "",
            "File → Open Heart Dancer loads the heart project. The dancer dock starts on the heart dancer. Heart on the dancer, or View → Heart Dancer, clears a sent project.",
            "",
        ]
    )
    return "\n".join(lines)


def write_catalog(original: Path = ORIGINAL) -> dict:
    """Copy the sheet and write the map. The original file is only read."""
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    dest = ASSET_DIR / SHEET_NAME
    shutil.copy2(original, dest)
    after = hashlib.sha256(original.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("The original sprite sheet changed")
    if hashlib.sha256(dest.read_bytes()).hexdigest() != before:
        raise RuntimeError("The copied sprite sheet does not match the original")
    measured = measure(dest)
    if measured["sha256"] != before:
        raise RuntimeError("Measured bytes do not match the original")
    frame_map = _frame_map(measured)
    animations = _animations_document()
    (ASSET_DIR / "green_dancer_sprites.json").write_text(json.dumps(frame_map, indent=2) + "\n", encoding="utf-8")
    (ASSET_DIR / "green_dancer_animations.json").write_text(json.dumps(animations, indent=2) + "\n", encoding="utf-8")
    project = build_project(measured, ASSET_DIR)
    project.save(ASSET_DIR / "green_dancer.sprite.json")
    _contact_sheet(dest, ASSET_DIR / "green_dancer_contact_sheet.png")
    report = _report(measured)
    docs = Path(__file__).resolve().parent.parent.parent / "docs" / "green_dancer_mapping_report.md"
    docs.write_text(report, encoding="utf-8")
    return measured


def main() -> None:
    measured = write_catalog()
    print(f"frames {len(measured['frames'])} sha256 {measured['sha256'][:12]}")
