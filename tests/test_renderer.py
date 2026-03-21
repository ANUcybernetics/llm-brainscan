import numpy as np
import pytest

from brainscan.renderer import (
    COLORMAP_DIVERGING,
    COLORMAP_THERMAL,
    OffscreenRenderer,
    flatten_weights,
    normalise_weights,
)


class TestNormaliseWeights:
    def test_normalises_to_unit_range(self):
        data = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        result = normalise_weights(data)
        assert result.max() == pytest.approx(1.0)
        assert result.min() == pytest.approx(-0.5)

    def test_symmetric_values(self):
        data = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
        result = normalise_weights(data)
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_all_zeros(self):
        data = np.zeros(10, dtype=np.float32)
        result = normalise_weights(data)
        np.testing.assert_allclose(result, np.zeros(10))

    def test_single_value(self):
        data = np.array([5.0], dtype=np.float32)
        result = normalise_weights(data)
        np.testing.assert_allclose(result, [1.0])

    def test_preserves_dtype(self):
        data = np.array([1.0, -1.0], dtype=np.float32)
        result = normalise_weights(data)
        assert result.dtype == np.float32


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
        # Thermal: positive should be brighter (yellow/white), negative darker (black/blue)
        assert img_pos[16, 16, :3].sum() > img_neg[16, 16, :3].sum()

    def test_different_sizes(self):
        for w, h in [(16, 16), (64, 32), (100, 50)]:
            renderer = OffscreenRenderer(w, h)
            data = np.random.randn(w * h).astype(np.float32)
            normed = normalise_weights(data)
            img = renderer.render(normed)
            assert img.shape == (h, w, 4)

    def test_multiple_renders(self, small_renderer):
        for i in range(5):
            value = (i - 2) / 2.0
            data = np.full(32 * 32, value, dtype=np.float32)
            img = small_renderer.render(data)
            assert img.shape == (32, 32, 4)

    def test_render_with_model_weights(self, small_renderer):
        """Integration test: render actual model weight data."""
        import torch
        from brainscan.model import GPT
        from brainscan.snapshot import capture_weights

        model = GPT(vocab_size=256, sequence_len=16, n_layer=1, n_head=1, n_embd=32)
        weights = capture_weights(model)
        np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
        flat, count = flatten_weights(np_weights)
        normed = normalise_weights(flat)
        # Pad or truncate to renderer size
        buf = np.zeros(32 * 32, dtype=np.float32)
        n = min(count, 32 * 32)
        buf[:n] = normed[:n]
        img = small_renderer.render(buf)
        assert img.shape == (32, 32, 4)
        assert np.any(img[:, :, :3] > 20), "Image should have non-background pixels"
