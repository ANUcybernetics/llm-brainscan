import numpy as np

from brainscan.font import GLYPH_COUNT, GLYPH_H, GLYPH_W, generate_font_atlas


class TestFontAtlas:
    def test_atlas_shape(self):
        atlas = generate_font_atlas()
        expected_u32s = GLYPH_COUNT * GLYPH_H // 4
        assert atlas.shape == (expected_u32s,)
        assert atlas.dtype == np.uint32

    def test_printable_chars_have_pixels(self):
        atlas = generate_font_atlas()
        raw = np.frombuffer(atlas.tobytes(), dtype=np.uint8)
        raw = raw.reshape(GLYPH_COUNT, GLYPH_H)
        for i in range(33, 127):
            glyph = raw[i]
            assert glyph.sum() > 0, f"Printable char {i} ({chr(i)!r}) has no pixels"

    def test_space_is_blank(self):
        atlas = generate_font_atlas()
        raw = np.frombuffer(atlas.tobytes(), dtype=np.uint8)
        raw = raw.reshape(GLYPH_COUNT, GLYPH_H)
        assert raw[32].sum() == 0

    def test_non_printable_are_blank(self):
        atlas = generate_font_atlas()
        raw = np.frombuffer(atlas.tobytes(), dtype=np.uint8)
        raw = raw.reshape(GLYPH_COUNT, GLYPH_H)
        for i in range(32):
            assert raw[i].sum() == 0, f"Non-printable char {i} has pixels"

    def test_different_chars_have_different_glyphs(self):
        atlas = generate_font_atlas()
        raw = np.frombuffer(atlas.tobytes(), dtype=np.uint8)
        raw = raw.reshape(GLYPH_COUNT, GLYPH_H)
        assert not np.array_equal(raw[ord("A")], raw[ord("B")])
        assert not np.array_equal(raw[ord("a")], raw[ord("z")])
