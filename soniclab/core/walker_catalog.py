"""Measured map of the walker sheet, and its dance on the downbeat.

The sheet is 1474×2000: a 4×4 grid of 368.5×500 cells, read left to right,
top to bottom, with the last cell empty, so 15 frames. Labeling the opaque
pixels gives 15 sprites (plus a 2-pixel speck inside frame 13's box, part
of that drawing). Every sprite stays inside its cell, and the drawings are
already registered: the torso sits near x 166–190 of each cell and the feet
on y 481–491, which is the walk's own sway and bounce. So each frame's
rectangle is its cell, 368×500 from the cell's left edge rounded down
(0, 368, 737, 1105), and nothing is re-centered.

The 15 frames are one walk that turns from the front through the back and
round again. Frame 14 leads back into frame 0, so the cycle loops. It plays
every two beats, twice a bar: frame 0 lands on every downbeat and on
beat 3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from soniclab.core.dancer import PHRASE_BEATS, SMOOTH_PER_BEAT
from soniclab.core.sprite_project import Animation, FrameInfo, SheetSpec, SpriteProject, Sync

IMAGE_W = 1474
IMAGE_H = 2000
COLUMNS = 4
ROWS = 4
PITCH_X = IMAGE_W / COLUMNS  # 368.5
CELL_W = 368
CELL_H = 500
FRAME_COUNT = 15
WALK_BEATS = 2  # beats per walk cycle: two cycles a bar

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "walker"
SHEET_NAME = "walker_sheet.png"
PROJECT_NAME = "walker.sprite.json"
_APP = None


def _ensure_app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication([])
    return _APP


def cell_rect(index: int) -> tuple[int, int, int, int]:
    row, column = divmod(int(index), COLUMNS)
    return int(column * PITCH_X), row * CELL_H, CELL_W, CELL_H


def _alpha(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    width, height = image.width(), image.height()
    raw = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
    return np.array(raw.reshape(height, image.bytesPerLine() // 4, 4)[:, :width, 3], copy=True)


def measure(path: Path) -> dict:
    """Frame rectangles and tight content boxes, with the checks that the
    grid is real: each cell holds one sprite and no sprite crosses it."""
    from scipy import ndimage

    _ensure_app()
    image = QImage(str(path))
    if image.isNull():
        raise RuntimeError(f"Could not read {path}")
    if (image.width(), image.height()) != (IMAGE_W, IMAGE_H):
        raise RuntimeError(f"{path.name} is {image.width()}×{image.height()}, expected {IMAGE_W}×{IMAGE_H}")
    alpha = _alpha(image)
    labels, _count = ndimage.label(alpha > 0)
    frames = []
    for index in range(COLUMNS * ROWS):
        x, y, width, height = cell_rect(index)
        cell = labels[y : y + height, x : x + width]
        found = {int(value) for value in np.unique(cell) if value}
        big = [label for label in found if (labels == label).sum() > 200]
        if index >= FRAME_COUNT:
            if big:
                raise RuntimeError(f"Cell {index} should be empty")
            continue
        if len(big) != 1:
            raise RuntimeError(f"frame_{index:03d} holds {len(big)} sprites")
        for label in found:
            ys, xs = np.where(labels == label)
            if xs.min() < x or ys.min() < y or xs.max() >= x + width or ys.max() >= y + height:
                raise RuntimeError(f"frame_{index:03d} crosses its cell")
        ys, xs = np.where(cell > 0)
        content = [int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        frames.append({"id": index, "rect": [x, y, width, height], "content_rect": content})
    return {
        "width": IMAGE_W,
        "height": IMAGE_H,
        "frames": frames,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def body_frames() -> list[int]:
    """The dancer clock's 96 frames (two bars, 12 a beat) as walker frames:
    the 15-frame walk every ``WALK_BEATS`` beats, starting on the downbeat."""
    per_cycle = WALK_BEATS * SMOOTH_PER_BEAT
    slots = PHRASE_BEATS * SMOOTH_PER_BEAT
    return [(slot % per_cycle) * FRAME_COUNT // per_cycle for slot in range(slots)]


def build_project(measured: dict, root: Path | None = None) -> SpriteProject:
    frames = [
        FrameInfo(
            id=item["id"],
            name=f"{item['id']:03d}",
            source=item["id"],
            category="dance",
            confidence="high",
            note="The walk turns front, side, back, side, front.",
            rect=tuple(item["rect"]),
            content_rect=tuple(item["content_rect"]),
        )
        for item in measured["frames"]
    ]
    body = body_frames()
    step = PHRASE_BEATS / len(body)
    project = SpriteProject(
        name="walker",
        bpm=125.0,
        sheet=SheetSpec(SHEET_NAME, CELL_W, CELL_H, COLUMNS, ROWS, 0, 0, 0, 0, measured["width"], measured["height"]),
        frames=frames,
        animations={
            "walk": Animation(
                name="walk",
                layer="body",
                frames=body,
                durations=[step] * len(body),
                fps=125.0 / 60.0 / step,
                loop="loop",
                base=True,
                trigger="",
                sync=Sync("downbeat", "beat", float(WALK_BEATS), 1.0, 1.0, 0.5, 0.0),
                note="The 15-frame walk every two beats. Frame 0 lands on every downbeat and on beat 3.",
            ),
        },
        root=str(root or ASSET_DIR),
    )
    project.normalize()
    return project


def walker_project() -> SpriteProject:
    return SpriteProject.load(ASSET_DIR / PROJECT_NAME)


def write_catalog(original: Path) -> dict:
    """Save the sheet as PNG (same pixels), measure it, and write the map
    and the project. The original file is only read."""
    _ensure_app()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    source = QImage(str(original))
    if source.isNull():
        raise RuntimeError(f"Could not read {original}")
    dest = ASSET_DIR / SHEET_NAME
    if not source.save(str(dest), "PNG"):
        raise RuntimeError(f"Could not write {dest}")
    if not np.array_equal(_alpha(source), _alpha(QImage(str(dest)))):
        raise RuntimeError("The PNG copy does not match the original")
    measured = measure(dest)
    frame_map = {
        "format": "soniclab.sprite-map",
        "version": 1,
        "character": "walker",
        "source": SHEET_NAME,
        "image": {"width": measured["width"], "height": measured["height"]},
        "grid": {
            "columns": COLUMNS,
            "rows": ROWS,
            "cell_width": CELL_W,
            "cell_height": CELL_H,
            "pitch_x": PITCH_X,
            "pitch_y": CELL_H,
            "frames": FRAME_COUNT,
            "note": "Cells are 368.5 wide; each rect starts at the cell edge rounded down. The last cell is empty.",
        },
        "frames": {
            f"frame_{item['id']:03d}": {"rect": item["rect"], "content_rect": item["content_rect"], "sheet_index": item["id"]}
            for item in measured["frames"]
        },
    }
    (ASSET_DIR / "walker_sprites.json").write_text(json.dumps(frame_map, indent=2) + "\n", encoding="utf-8")
    build_project(measured, ASSET_DIR).save(ASSET_DIR / PROJECT_NAME)
    return measured


if __name__ == "__main__":
    import sys

    result = write_catalog(Path(sys.argv[1]))
    print(f"frames {len(result['frames'])} sha256 {result['sha256'][:12]}")
