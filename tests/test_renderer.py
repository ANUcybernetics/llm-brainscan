import numpy as np
import pytest

from brainscan.renderer import (
    COLORMAP_THERMAL,
    OffscreenRenderer,
    flatten_weights,
)


class TestFlattenWeights:
    def test_single_tensor(self):
        weights = {"a": np.array([[1, 2], [3, 4]], dtype=np.float32)}
        flat, count = flatten_weights(weights)
        np.testing.assert_array_equal(flat, [1, 2, 3, 4])
        assert count == 4

    def test_multiple_tensors(self):
        weights = {
            "a": np.array([1, 2], dtype=np.float32),
            "b": np.array([3, 4, 5], dtype=np.float32),
        }
        flat, count = flatten_weights(weights)
        np.testing.assert_array_equal(flat, [1, 2, 3, 4, 5])
        assert count == 5

    def test_custom_order(self):
        weights = {
            "a": np.array([1, 2], dtype=np.float32),
            "b": np.array([3, 4], dtype=np.float32),
        }
        flat, _ = flatten_weights(weights, layout_order=["b", "a"])
        np.testing.assert_array_equal(flat, [3, 4, 1, 2])

    def test_order_skips_missing(self):
        weights = {"a": np.array([1], dtype=np.float32)}
        flat, count = flatten_weights(weights, layout_order=["b", "a"])
        np.testing.assert_array_equal(flat, [1])
        assert count == 1

    def test_output_is_float32(self):
        weights = {"a": np.array([1, 2], dtype=np.float64)}
        flat, _ = flatten_weights(weights)
        assert flat.dtype == np.float32


class TestOffscreenRenderer:
    @pytest.fixture
    def small_renderer(self):
        return OffscreenRenderer(32, 32)

    def test_render_returns_correct_shape(self, small_renderer):
        data = np.zeros(32 * 32, dtype=np.float32)
        img = small_renderer.render(data)
        assert img.shape == (32, 32, 4)
        assert img.dtype == np.uint8

    def test_render_all_zeros_is_midpoint(self, small_renderer):
        data = np.zeros(32 * 32, dtype=np.float32)
        img = small_renderer.render(data)
        # Diverging colourmap at 0.0: r=0.5, g=0.5, b=0.5 -> ~128
        centre = img[16, 16, :3]
        assert all(120 <= c <= 136 for c in centre), f"Expected ~128, got {centre}"

    def test_render_positive_is_reddish(self, small_renderer):
        data = np.ones(32 * 32, dtype=np.float32)
        img = small_renderer.render(data)
        pixel = img[16, 16]
        assert pixel[0] > pixel[2], f"Expected R > B for positive weights, got {pixel}"

    def test_render_negative_is_bluish(self, small_renderer):
        data = -np.ones(32 * 32, dtype=np.float32)
        img = small_renderer.render(data)
        pixel = img[16, 16]
        assert pixel[2] > pixel[0], f"Expected B > R for negative weights, got {pixel}"

    def test_gpu_normalisation_matches_range(self, small_renderer):
        data = np.full(32 * 32, 5.0, dtype=np.float32)
        img = small_renderer.render(data)
        pixel = img[16, 16]
        data_unit = np.ones(32 * 32, dtype=np.float32)
        img_unit = small_renderer.render(data_unit)
        pixel_unit = img_unit[16, 16]
        np.testing.assert_array_equal(pixel, pixel_unit)

    def test_gpu_normalisation_all_zeros(self, small_renderer):
        data = np.zeros(32 * 32, dtype=np.float32)
        img = small_renderer.render(data)
        centre = img[16, 16, :3]
        assert all(120 <= c <= 136 for c in centre), f"Expected ~128, got {centre}"

    def test_partial_fill(self, small_renderer):
        data = np.ones(100, dtype=np.float32)
        img = small_renderer.render(data)
        assert img.shape == (32, 32, 4)
        # First 100 pixels should be coloured, rest should be background
        flat_img = img.reshape(-1, 4)
        coloured = flat_img[:100]
        background = flat_img[100:]
        assert coloured[:, 0].mean() > 100, "Filled pixels should be coloured"
        bg_rgb = background[:, :3]
        assert bg_rgb.max() < 30, f"Background should be dark, got max {bg_rgb.max()}"

    def test_thermal_colormap(self):
        renderer = OffscreenRenderer(32, 32, colormap=COLORMAP_THERMAL)
        positive = np.ones(32 * 32, dtype=np.float32)
        img_pos = renderer.render(positive)
        negative = -np.ones(32 * 32, dtype=np.float32)
        img_neg = renderer.render(negative)
        # Thermal: positive=brighter (yellow/white), negative=darker (black/blue)
        assert img_pos[16, 16, :3].sum() > img_neg[16, 16, :3].sum()

    def test_different_sizes(self):
        for w, h in [(16, 16), (64, 32), (100, 50)]:
            renderer = OffscreenRenderer(w, h)
            data = np.random.randn(w * h).astype(np.float32)
            img = renderer.render(data)
            assert img.shape == (h, w, 4)

    def test_multiple_renders(self, small_renderer):
        for i in range(5):
            value = (i - 2) / 2.0
            data = np.full(32 * 32, value, dtype=np.float32)
            img = small_renderer.render(data)
            assert img.shape == (32, 32, 4)

    def test_render_with_text_no_strip(self, small_renderer):
        data = np.zeros(32 * 32, dtype=np.float32)
        img = small_renderer.render(data, text_chars=None, text_probs=None)
        assert img.shape == (32, 32, 4)

    def test_render_with_model_weights(self, small_renderer):
        """Integration test: render actual model weight data."""
        from brainscan.model import GPT
        from brainscan.snapshot import capture_weights

        model = GPT(vocab_size=256, sequence_len=16, n_layer=1, n_head=1, n_embd=32)
        weights = capture_weights(model)
        np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
        flat, count = flatten_weights(np_weights)
        buf = np.zeros(32 * 32, dtype=np.float32)
        n = min(count, 32 * 32)
        buf[:n] = flat[:n]
        img = small_renderer.render(buf)
        assert img.shape == (32, 32, 4)
        assert np.any(img[:, :, :3] > 20), "Image should have non-background pixels"


class TestTextStripRenderer:
    @pytest.fixture
    def text_renderer(self):
        return OffscreenRenderer(64, 64, text_strip_height=32, text_scale=1)

    def test_text_strip_dimensions(self, text_renderer):
        assert text_renderer.text_y == 32
        assert text_renderer.text_cols == 8  # 64 / (8 * 1)

    def test_render_with_text(self, text_renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        chars = np.array([ord("A"), ord("B"), ord("C")], dtype=np.uint32)
        probs = np.array([1.0, 0.5, 0.1], dtype=np.float32)
        img = text_renderer.render(weights, text_chars=chars, text_probs=probs)
        assert img.shape == (64, 64, 4)

    def test_text_pixels_differ_from_background(self, text_renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        chars = np.array([ord("W")] * 8, dtype=np.uint32)
        probs = np.ones(8, dtype=np.float32)
        img = text_renderer.render(weights, text_chars=chars, text_probs=probs)
        text_region = img[32:48, :, :3]
        assert text_region.max() > 30, "Text region should have visible pixels"

    def test_empty_text_is_dark(self, text_renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        img = text_renderer.render(weights)
        text_region = img[32:, :, :3]
        assert text_region.max() < 30, f"Empty text region should be dark, got max {text_region.max()}"

    def test_high_prob_brighter_than_low(self, text_renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        chars_hi = np.array([ord("X")] * 8, dtype=np.uint32)
        probs_hi = np.ones(8, dtype=np.float32)
        img_hi = text_renderer.render(weights, text_chars=chars_hi, text_probs=probs_hi)

        chars_lo = np.array([ord("X")] * 8, dtype=np.uint32)
        probs_lo = np.full(8, 0.1, dtype=np.float32)
        img_lo = text_renderer.render(weights, text_chars=chars_lo, text_probs=probs_lo)

        hi_brightness = img_hi[32:48, :, :3].astype(float).sum()
        lo_brightness = img_lo[32:48, :, :3].astype(float).sum()
        assert hi_brightness > lo_brightness, "High probability text should be brighter"
