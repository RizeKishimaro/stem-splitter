#!/usr/bin/env python3
"""Draw SonicLab's icon as a PNG. The Mac installer turns it into an .icns file."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(pixels[y * width * 4 : (y + 1) * width * 4]) for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    size = 1024
    pixels = bytearray(size * size * 4)
    radius = 220
    left, top, right, bottom = 64, 64, size - 64, size - 64
    fill = (28, 31, 38, 255)
    gold = (242, 193, 78, 255)
    bars = (
        (250, 390, 340, 634),
        (390, 280, 480, 744),
        (530, 340, 620, 684),
        (670, 430, 760, 594),
    )
    for y in range(size):
        for x in range(size):
            inside = left + radius <= x < right - radius or top + radius <= y < bottom - radius
            if not inside:
                cx = min(max(x, left + radius), right - radius - 1)
                cy = min(max(y, top + radius), bottom - radius - 1)
                inside = (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius
            if left <= x < right and top <= y < bottom and inside:
                color = fill
                for bar in bars:
                    if bar[0] <= x < bar[2] and bar[1] <= y < bar[3]:
                        color = gold
                        break
                offset = (y * size + x) * 4
                pixels[offset : offset + 4] = bytes(color)
    destination = Path(__file__).resolve().parents[1] / "SonicLab.png"
    write_png(destination, size, size, pixels)


if __name__ == "__main__":
    main()
