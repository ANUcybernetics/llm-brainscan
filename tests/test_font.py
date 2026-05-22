import numpy as np
import pytest

from brainscan.font import (
    GLYPH_COUNT,
    GLYPH_H,
    GLYPH_W,
    LANE_GLYPH_H,
    LANE_GLYPH_W,
    generate_font_atlas,
    generate_lane_font_atlas,
)


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


class TestLaneFontAtlas:
    @pytest.fixture(scope="class")
    def atlas(self):
        return generate_lane_font_atlas()

    @staticmethod
    def _coverage(atlas):
        return np.frombuffer(atlas.tobytes(), dtype=np.uint8).reshape(
            GLYPH_COUNT, LANE_GLYPH_H, LANE_GLYPH_W
        )

    def test_atlas_shape(self, atlas):
        expected_u32s = GLYPH_COUNT * LANE_GLYPH_W * LANE_GLYPH_H // 4
        assert atlas.shape == (expected_u32s,)
        assert atlas.dtype == np.uint32

    def test_coverage_is_antialiased(self, atlas):
        # A 1-bit atlas carries only 0 and 255; an antialiased atlas has
        # intermediate coverage along glyph edges.
        cov = self._coverage(atlas)
        intermediate = cov[(cov > 0) & (cov < 255)]
        assert intermediate.size > 0

    def test_printable_chars_have_ink(self, atlas):
        cov = self._coverage(atlas)
        for i in range(33, 127):
            assert cov[i].sum() > 0, f"Printable char {i} ({chr(i)!r}) has no ink"

    def test_space_is_blank(self, atlas):
        cov = self._coverage(atlas)
        assert cov[32].sum() == 0

    def test_non_printable_are_blank(self, atlas):
        cov = self._coverage(atlas)
        for i in range(32):
            assert cov[i].sum() == 0, f"Non-printable char {i} has ink"

    def test_glyphs_share_a_baseline(self, atlas):
        # Capitals without descenders all rest on one baseline: their lowest
        # inked row must coincide.
        cov = self._coverage(atlas)

        def bottom_row(ch):
            inked = np.where(cov[ord(ch)].any(axis=1))[0]
            return int(inked[-1])

        assert bottom_row("E") == bottom_row("H") == bottom_row("X")

    def test_glyphs_fit_within_cell(self, atlas):
        # Ink is centred in the cell, so the extreme rows stay clear; a
        # clipped, oversized glyph would paint solid coverage on an edge row.
        cov = self._coverage(atlas)
        printable = cov[33:127]
        assert printable[:, 0, :].max() < 8
        assert printable[:, -1, :].max() < 8

    def test_different_chars_have_different_glyphs(self, atlas):
        cov = self._coverage(atlas)
        assert not np.array_equal(cov[ord("A")], cov[ord("B")])
        assert not np.array_equal(cov[ord("a")], cov[ord("z")])
