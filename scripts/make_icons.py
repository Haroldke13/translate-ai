#!/usr/bin/env python3
"""Generate the PWA icons with no image library, so the build stays offline.

    python scripts/make_icons.py

Writes 192px, 512px and a maskable 512px icon into kikuyu_ai/web/static/icons.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

BRAND = (15, 118, 110)
BRAND_DARK = (11, 92, 86)
INK = (255, 255, 255)
OUTPUT = Path(__file__).resolve().parents[1] / "kikuyu_ai" / "web" / "static" / "icons"

# A speech waveform: relative bar heights across the middle of the icon.
BARS = (0.30, 0.55, 0.86, 1.00, 0.72, 0.44, 0.66, 0.92, 0.58, 0.34)


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height, width = len(pixels), len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # PNG filter type 0 (None) for each scanline
        for red, green, blue in row:
            raw += bytes((red, green, blue))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def blend(bottom: tuple[int, int, int], top: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    alpha = max(0.0, min(1.0, alpha))
    return tuple(round(bottom[i] * (1 - alpha) + top[i] * alpha) for i in range(3))


def rounded_coverage(x: int, y: int, size: int, radius: float, inset: float) -> float:
    """Anti-aliased coverage of a rounded square at one pixel."""
    left = top = inset
    right = bottom = size - inset
    if not (left <= x + 0.5 <= right and top <= y + 0.5 <= bottom):
        return 0.0
    corners = (
        (left + radius, top + radius, x + 0.5 < left + radius and y + 0.5 < top + radius),
        (right - radius, top + radius, x + 0.5 > right - radius and y + 0.5 < top + radius),
        (left + radius, bottom - radius, x + 0.5 < left + radius and y + 0.5 > bottom - radius),
        (right - radius, bottom - radius, x + 0.5 > right - radius and y + 0.5 > bottom - radius),
    )
    for cx, cy, inside_corner in corners:
        if inside_corner:
            distance = ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) ** 0.5
            return max(0.0, min(1.0, radius - distance + 0.5))
    return 1.0


def build(size: int, maskable: bool) -> list[list[tuple[int, int, int]]]:
    # Maskable icons get a safe zone: the launcher may crop to a circle.
    inset = size * 0.0 if maskable else size * 0.045
    radius = size * (0.5 if maskable else 0.22)
    background = INK if not maskable else BRAND
    pixels = [[background if not maskable else BRAND for _ in range(size)] for _ in range(size)]

    if not maskable:
        for y in range(size):
            for x in range(size):
                coverage = rounded_coverage(x, y, size, radius, inset)
                if coverage:
                    # Vertical gradient reads as depth at small sizes.
                    tint = blend(BRAND, BRAND_DARK, y / size)
                    pixels[y][x] = blend(pixels[y][x], tint, coverage)
    else:
        for y in range(size):
            tint = blend(BRAND, BRAND_DARK, y / size)
            for x in range(size):
                pixels[y][x] = tint

    scale = 0.62 if maskable else 0.74
    span = size * scale
    left = (size - span) / 2
    centre = size / 2
    bar_width = span / (len(BARS) * 2 - 1)
    for index, height_ratio in enumerate(BARS):
        bar_left = left + index * bar_width * 2
        half = (span * 0.5 * height_ratio) / 2
        cap = bar_width / 2
        for y in range(size):
            for x in range(int(bar_left) - 1, int(bar_left + bar_width) + 2):
                if not 0 <= x < size:
                    continue
                dx = x + 0.5 - (bar_left + bar_width / 2)
                dy = abs(y + 0.5 - centre)
                # A capsule: a rectangle with semicircular ends.
                if dy <= half - cap:
                    distance = abs(dx)
                else:
                    distance = ((dx) ** 2 + (dy - (half - cap)) ** 2) ** 0.5
                coverage = max(0.0, min(1.0, cap - distance + 0.5))
                if coverage:
                    pixels[y][x] = blend(pixels[y][x], INK, coverage)
    return pixels


def main() -> None:
    for name, size, maskable in (
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("icon-maskable-512.png", 512, True),
    ):
        write_png(OUTPUT / name, build(size, maskable))
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
