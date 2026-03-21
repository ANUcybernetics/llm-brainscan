"""Lay out weight tensors on an 8K canvas (7680x4320) with one pixel per parameter."""

import math

import torch
import numpy as np

WIDTH = 7680
HEIGHT = 4320
TOTAL_PIXELS = WIDTH * HEIGHT


def compute_layout(param_groups: dict[str, int]) -> dict[str, dict]:
    total = sum(v for k, v in param_groups.items() if k != "total")
    assert total <= TOTAL_PIXELS, f"Model has {total} params but display has {TOTAL_PIXELS} pixels"

    layout = {}
    y_cursor = 0
    x_cursor = 0
    row_height = 0

    for name, count in param_groups.items():
        if name == "total":
            continue

        preferred_w = min(WIDTH, int(math.sqrt(count * (WIDTH / HEIGHT))))
        preferred_w = max(1, preferred_w)
        h = math.ceil(count / preferred_w)
        w = min(preferred_w, WIDTH)

        if x_cursor + w > WIDTH:
            y_cursor += row_height + 2
            x_cursor = 0
            row_height = 0

        layout[name] = {
            "x": x_cursor,
            "y": y_cursor,
            "w": w,
            "h": h,
            "count": count,
        }

        row_height = max(row_height, h)
        x_cursor += w + 2

    return layout


def weights_to_image(
    weights: dict[str, torch.Tensor],
    layout: dict[str, dict],
    colormap: str = "diverging",
) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for name, tensor in weights.items():
        if name not in layout:
            continue
        info = layout[name]
        flat = tensor.float().cpu().flatten().numpy()
        vmax = max(abs(flat.min()), abs(flat.max()), 1e-8)
        normalised = flat / vmax

        if colormap == "diverging":
            rgb = _diverging_colormap(normalised)
        else:
            rgb = _viridis_colormap((normalised + 1) / 2)

        x, y, w, h = info["x"], info["y"], info["w"], info["h"]
        padded = np.zeros(w * h, dtype=normalised.dtype)
        padded[: len(flat)] = normalised
        pixels = np.zeros((w * h, 3), dtype=np.uint8)
        pixels[: len(flat)] = rgb
        pixels_2d = pixels.reshape(h, w, 3)

        y_end = min(y + h, HEIGHT)
        x_end = min(x + w, WIDTH)
        canvas[y : y_end, x : x_end] = pixels_2d[: y_end - y, : x_end - x]

    return canvas


def _diverging_colormap(values: np.ndarray) -> np.ndarray:
    r = np.clip((values * 255).astype(np.int16) + 128, 0, 255).astype(np.uint8)
    b = np.clip((-values * 255).astype(np.int16) + 128, 0, 255).astype(np.uint8)
    g = np.clip(128 - np.abs(values * 128).astype(np.int16), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _viridis_colormap(values: np.ndarray) -> np.ndarray:
    t = np.clip(values, 0, 1)
    r = np.clip((68 + t * (187)).astype(np.uint8), 0, 255)
    g = np.clip((1 + t * (254)).astype(np.uint8), 0, 255)
    b = np.clip((84 + t * (80) - t * t * 80).astype(np.uint8), 0, 255)
    return np.stack([r, g, b], axis=-1)
