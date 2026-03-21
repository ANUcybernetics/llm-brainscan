"""Lay out weight tensors on an 8K canvas (7680x4320).

Information flows left to right: embeddings → transformer blocks → output.
Each section is a vertical column; matrices stack top-to-bottom within their
column with small gutters between them.

The layout preserves spatial consistency across frames (critical for animation
during training), so the same parameter always occupies the same pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WIDTH = 7680
HEIGHT = 4320
TOTAL_PIXELS = WIDTH * HEIGHT
GUTTER = 4


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int
    count: int
    name: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "count": self.count}


@dataclass(frozen=True)
class Section:
    """A vertical column of matrices in the layout."""

    label: str
    param_names: list[str]


def default_sections(n_layer: int = 8) -> list[Section]:
    """Define the left-to-right section ordering for a GPT model."""
    sections = [
        Section("embed", ["wte.weight", "wpe.weight"]),
    ]
    for i in range(n_layer):
        p = f"blocks.{i}"
        sections.append(
            Section(
                f"block_{i}",
                [
                    f"{p}.ln_1.weight",
                    f"{p}.ln_1.bias",
                    f"{p}.attn.c_attn.weight",
                    f"{p}.attn.c_proj.weight",
                    f"{p}.ln_2.weight",
                    f"{p}.ln_2.bias",
                    f"{p}.mlp.c_fc.weight",
                    f"{p}.mlp.c_proj.weight",
                ],
            )
        )
    sections.append(
        Section("output", ["ln_f.weight", "ln_f.bias", "lm_head.weight"]),
    )
    return sections


def _section_param_total(
    section: Section, param_counts: dict[str, int]
) -> int:
    return sum(param_counts.get(name, 0) for name in section.param_names)


def _column_width(total_params: int, height: int, gutter: int, n_items: int) -> int:
    """Compute the column width that fits total_params in the given height.

    Accounts for gutters between items within the column.
    """
    avail_h = height - max(0, n_items - 1) * gutter
    if avail_h <= 0:
        return total_params
    return max(1, math.ceil(total_params / avail_h))


def compute_layout(
    param_counts: dict[str, int],
    sections: list[Section] | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    gutter: int = GUTTER,
) -> dict[str, Rect]:
    """Compute pixel layout for all parameters.

    Args:
        param_counts: mapping from parameter name to element count.
        sections: left-to-right section ordering. Defaults to standard GPT layout.
        width: canvas width in pixels.
        height: canvas height in pixels.
        gutter: pixels of padding between matrices and between sections.

    Returns:
        mapping from parameter name to Rect with pixel coordinates.
    """
    if sections is None:
        sections = default_sections()

    section_widths: list[int] = []
    for section in sections:
        total = _section_param_total(section, param_counts)
        n_items = sum(1 for name in section.param_names if param_counts.get(name, 0) > 0)
        col_w = _column_width(total, height, gutter, n_items)
        section_widths.append(col_w)

    total_gutters = max(0, len(sections) - 1) * gutter
    total_content_w = sum(section_widths)
    total_w = total_content_w + total_gutters

    if total_w > width:
        scale = (width - total_gutters) / total_content_w
        section_widths = [max(1, int(w * scale)) for w in section_widths]

    layout: dict[str, Rect] = {}
    x_cursor = 0

    for section, col_w in zip(sections, section_widths):
        y_cursor = 0

        for name in section.param_names:
            count = param_counts.get(name, 0)
            if count == 0:
                continue
            h = math.ceil(count / col_w)
            rect = Rect(x=x_cursor, y=y_cursor, w=col_w, h=h, count=count, name=name)
            layout[name] = rect
            y_cursor += h + gutter

        x_cursor += col_w + gutter

    return layout


def layout_to_flat_order(layout: dict[str, Rect]) -> list[str]:
    """Return parameter names in layout order (for flattening weights)."""
    return sorted(layout.keys(), key=lambda n: (layout[n].x, layout[n].y))


def layout_summary(layout: dict[str, Rect]) -> str:
    """Human-readable summary of the layout."""
    lines = []
    sorted_rects = sorted(layout.values(), key=lambda r: (r.x, r.y))
    current_x = -1
    for rect in sorted_rects:
        if rect.x != current_x:
            current_x = rect.x
            lines.append(f"\n  x={rect.x}:")
        lines.append(
            f"    {rect.name:45s}  {rect.w:4d}x{rect.h:<5d}  "
            f"({rect.count:>10,} params)  @ ({rect.x},{rect.y})"
        )
    total = sum(r.count for r in layout.values())
    max_x = max(r.x + r.w for r in layout.values())
    max_y = max(r.y + r.h for r in layout.values())
    lines.append(f"\n  total: {total:,} params in {max_x}x{max_y} canvas")
    return "\n".join(lines)
