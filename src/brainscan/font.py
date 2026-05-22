"""Font atlases for GPU text rendering.

Two atlases are generated, both packed into uint32 arrays for upload to wgpu
storage buffers:

* ``generate_font_atlas`` --- a 1-bit 8x16 bitmap atlas covering all 256 byte
  values. Used for the small dim-grey chrome labels (section and matrix names)
  in the weight region, where glyphs are drawn at native size.
* ``generate_lane_font_atlas`` --- a 32x64 px, 8-bit antialiased *coverage*
  atlas rendered from IBM Plex Mono on a shared baseline. Used for the two
  conversation lanes in the bottom strip, sampled 1:1 by the fragment shader
  so the text is smooth rather than a blocky upscale of a tiny bitmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_W = 8
GLYPH_H = 16
GLYPH_COUNT = 256

# Lane atlas: one antialiased glyph per cell, sampled 1:1 by the renderer.
LANE_GLYPH_W = 32
LANE_GLYPH_H = 64

_FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "IBMPlexMono-Regular.ttf"
)

_LANE_POINT_SIZE = 52
"""Point size (at 1x) for the lane font. Chosen so IBM Plex Mono's advance
width (~31 px) fills the 32 px lane cell with a sliver of side bearing."""

_LANE_SUPERSAMPLE = 4
"""Glyphs are rendered at this multiple of the cell size, then Lanczos-
downsampled, for smooth and evenly weighted antialiased coverage."""


def generate_font_atlas() -> np.ndarray:
    """Generate an 8x16 1-bit bitmap font atlas for byte values 0-255.

    Each glyph is stored as 16 bytes (one byte per row, MSB = leftmost pixel).
    The full atlas (256 * 16 = 4096 bytes) is packed into 1024 uint32 values.
    """
    font = ImageFont.load_default(size=13)
    data = np.zeros((GLYPH_COUNT, GLYPH_H), dtype=np.uint8)

    for i in range(GLYPH_COUNT):
        ch = chr(i) if 32 <= i < 127 else ""
        if not ch:
            continue

        img = Image.new("L", (GLYPH_W, GLYPH_H), 0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), ch, font=font)
        cx = (GLYPH_W - (bbox[2] - bbox[0])) // 2
        cy = (GLYPH_H - (bbox[3] - bbox[1])) // 2
        draw.text((cx - bbox[0], cy - bbox[1]), ch, fill=255, font=font)

        pixels = np.array(img)
        for y in range(GLYPH_H):
            byte_val = 0
            for x in range(GLYPH_W):
                if pixels[y, x] > 100:
                    byte_val |= 0x80 >> x
            data[i, y] = byte_val

    return np.frombuffer(data.tobytes(), dtype=np.uint32).copy()


def _measure_baseline(font: ImageFont.FreeTypeFont, cell_h: int) -> int:
    """Return the baseline y (in pixels) that vertically centres the row.

    Renders every printable ASCII glyph against a common baseline, measures
    the combined ink extent above and below it, and returns the baseline that
    centres that extent in a ``cell_h``-tall cell. Centring the *ink* --- not
    the font's metric ascent, which reserves space for accents --- keeps the
    row visually balanced in the cell.
    """
    probe_baseline = cell_h
    scratch = Image.new("L", (cell_h * 2, cell_h * 3), 0)
    draw = ImageDraw.Draw(scratch)
    for i in range(32, 127):
        draw.text((0, probe_baseline), chr(i), font=font, fill=255, anchor="ls")
    rows = np.where(np.array(scratch).any(axis=1))[0]
    ink_ascent = probe_baseline - int(rows[0])
    ink_descent = int(rows[-1]) - probe_baseline
    return (cell_h - (ink_ascent + ink_descent)) // 2 + ink_ascent


def generate_lane_font_atlas() -> np.ndarray:
    """Generate a 32x64 8-bit antialiased coverage atlas for the lanes.

    Each printable glyph is rendered from IBM Plex Mono on a shared baseline
    at ``_LANE_SUPERSAMPLE``x, then Lanczos-downsampled into the 32x64 cell.
    Coverage is glyph-major, row-major within a glyph; the
    256 * 32 * 64 = 524288 bytes are packed into 131072 uint32 values for
    upload to a wgpu storage buffer.
    """
    ss = _LANE_SUPERSAMPLE
    font = ImageFont.truetype(str(_FONT_PATH), _LANE_POINT_SIZE * ss)
    cell_w, cell_h = LANE_GLYPH_W * ss, LANE_GLYPH_H * ss

    baseline_y = _measure_baseline(font, cell_h)
    pen_x = (cell_w - font.getlength("M")) / 2.0  # monospace: uniform advance

    data = np.zeros((GLYPH_COUNT, LANE_GLYPH_H, LANE_GLYPH_W), dtype=np.uint8)
    for i in range(32, 127):
        big = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(big).text(
            (pen_x, baseline_y), chr(i), font=font, fill=255, anchor="ls"
        )
        small = big.resize(
            (LANE_GLYPH_W, LANE_GLYPH_H), Image.Resampling.LANCZOS
        )
        data[i] = np.asarray(small, dtype=np.uint8)

    return np.frombuffer(data.tobytes(), dtype=np.uint32).copy()
