"""Bitmap font atlas for GPU text rendering.

Generates an 8x16 pixel font for all 256 byte values, packed into a uint32
array suitable for upload to a wgpu storage buffer.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GLYPH_W = 8
GLYPH_H = 16
GLYPH_COUNT = 256


def generate_font_atlas() -> np.ndarray:
    """Generate an 8x16 bitmap font atlas for byte values 0-255.

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
