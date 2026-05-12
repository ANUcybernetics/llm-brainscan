"""Lay out weight tensors on an 8K canvas (7680x4320).

Information flows left to right: embeddings → transformer blocks → output.
Each section is a vertical column; matrices stack top-to-bottom within their
column with small gutters between them.

The layout preserves spatial consistency across frames (critical for animation
during training), so the same parameter always occupies the same pixel.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

WIDTH = 7680
HEIGHT = 4320
TEXT_STRIP_HEIGHT = 224
LAYOUT_HEIGHT = HEIGHT - TEXT_STRIP_HEIGHT
TOTAL_PIXELS = WIDTH * HEIGHT
GUTTER = 4
GLYPH_W = 8
GLYPH_H = 16


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
class Item:
    """A single parameter slot in a group, optionally labelled."""

    param_name: str
    label: str | None


@dataclass(frozen=True)
class Group:
    """A vertically-stacked group of items inside a section."""

    label: str
    items: list[Item]


@dataclass(frozen=True)
class Section:
    """A vertical column of groups in the layout."""

    label: str
    groups: list[Group]


@dataclass(frozen=True)
class TextOverlay:
    """A single short string to be drawn in a gutter or label band."""

    text: str
    x: int
    y: int


def iter_param_names(section: Section) -> Iterable[str]:
    """Yield parameter names in display order for a section."""
    for group in section.groups:
        for item in group.items:
            yield item.param_name


def default_sections(n_layer: int = 8) -> list[Section]:
    """Define the left-to-right section ordering for a GPT model.

    Each transformer block is split into an `attn` and `mlp` group; the four
    substantial matrices (`c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`)
    carry the labels `qkv`, `proj`, `up`, `down`.
    """
    sections: list[Section] = [
        Section(
            label="EMBED",
            groups=[
                Group(
                    label="",
                    items=[
                        Item("wte.weight", None),
                        Item("wpe.weight", None),
                    ],
                )
            ],
        )
    ]
    for i in range(n_layer):
        p = f"blocks.{i}"
        sections.append(
            Section(
                label=f"BLK {i}",
                groups=[
                    Group(
                        label="attn",
                        items=[
                            Item(f"{p}.ln_1.weight", None),
                            Item(f"{p}.ln_1.bias", None),
                            Item(f"{p}.attn.c_attn.weight", "qkv"),
                            Item(f"{p}.attn.c_proj.weight", "proj"),
                        ],
                    ),
                    Group(
                        label="mlp",
                        items=[
                            Item(f"{p}.ln_2.weight", None),
                            Item(f"{p}.ln_2.bias", None),
                            Item(f"{p}.mlp.c_fc.weight", "up"),
                            Item(f"{p}.mlp.c_proj.weight", "down"),
                        ],
                    ),
                ],
            )
        )
    sections.append(
        Section(
            label="OUT",
            groups=[
                Group(
                    label="",
                    items=[
                        Item("ln_f.weight", None),
                        Item("ln_f.bias", None),
                        Item("lm_head.weight", None),
                    ],
                )
            ],
        )
    )
    return sections


def _section_param_total(section: Section, param_counts: dict[str, int]) -> int:
    return sum(param_counts.get(name, 0) for name in iter_param_names(section))


def _column_width(
    item_counts: list[int], height: int, gutter: int
) -> int:
    """Compute the minimum column width that fits all items in the given height.

    Uses the total param count as a starting estimate, then widens if
    accumulated per-item ceil() rounding causes overflow.
    """
    n_items = len(item_counts)
    total_params = sum(item_counts)
    avail_h = height - max(0, n_items - 1) * gutter
    if avail_h <= 0:
        return total_params
    col_w = max(1, math.ceil(total_params / avail_h))
    while True:
        used = sum(math.ceil(c / col_w) for c in item_counts)
        total_h = used + max(0, n_items - 1) * gutter
        if total_h <= height:
            return col_w
        col_w += 1


def compute_layout(
    param_counts: dict[str, int],
    sections: list[Section] | None = None,
    width: int = WIDTH,
    height: int = LAYOUT_HEIGHT,
    section_gutter: int = GUTTER,
    group_gutter: int = GUTTER,
    item_gutter: int = GUTTER,
    label_gap_px: int = GUTTER,
    section_label_height: int = 0,
    min_section_width: int = 1,
) -> dict[str, Rect]:
    """Compute pixel layout for all parameters.

    Args:
        param_counts: mapping from parameter name to element count.
        sections: left-to-right section ordering. Defaults to standard GPT layout.
        width: canvas width in pixels.
        height: canvas height in pixels.
        section_gutter: pixels between sections.
        group_gutter: pixels between groups inside a section.
        item_gutter: pixels between unlabelled consecutive items in a group.
        label_gap_px: pixels above an item that carries a non-None label
            (replaces ``item_gutter`` for that item). Does not apply to the
            first item of a group.
        section_label_height: pixels reserved at the top of each section column
            for a section label. Items inside groups start below this band.
        min_section_width: minimum column width for a section. Floors narrow
            sections so they remain visible.

    Returns:
        mapping from parameter name to Rect with pixel coordinates.
    """
    if sections is None:
        sections = default_sections()

    section_widths: list[int] = []
    for section in sections:
        item_counts: list[int] = []
        chrome = section_label_height
        for g_idx, group in enumerate(section.groups):
            if g_idx > 0:
                chrome += group_gutter
            first_in_group = True
            for item in group.items:
                count = param_counts.get(item.param_name, 0)
                if count == 0:
                    continue
                if not first_in_group:
                    chrome += label_gap_px if item.label is not None else item_gutter
                first_in_group = False
                item_counts.append(count)
        avail_h = max(1, height - chrome)
        col_w = max(_column_width(item_counts, avail_h, 0), min_section_width)
        section_widths.append(col_w)

    total_gutters = max(0, len(sections) - 1) * section_gutter
    total_content_w = sum(section_widths)
    total_w = total_content_w + total_gutters

    if total_w > width:
        scale = (width - total_gutters) / total_content_w
        section_widths = [max(1, int(w * scale)) for w in section_widths]

    layout: dict[str, Rect] = {}
    x_cursor = 0

    for section, col_w in zip(sections, section_widths, strict=True):
        y_cursor = section_label_height
        for g_idx, group in enumerate(section.groups):
            if g_idx > 0:
                y_cursor += group_gutter
            for i_idx, item in enumerate(group.items):
                count = param_counts.get(item.param_name, 0)
                if count == 0:
                    continue
                if i_idx > 0:
                    y_cursor += label_gap_px if item.label is not None else item_gutter
                h = math.ceil(count / col_w)
                rect = Rect(
                    x=x_cursor, y=y_cursor, w=col_w, h=h,
                    count=count, name=item.param_name,
                )
                layout[item.param_name] = rect
                y_cursor += h
        x_cursor += col_w + section_gutter

    return layout


def compute_text_overlays(
    layout: dict[str, Rect], sections: list[Section]
) -> list[TextOverlay]:
    """Return overlays for every section label and every labelled matrix.

    Section labels are centred in the section-label band at the top of each
    section column. Matrix labels are left-aligned 1 px above their matrix.
    """
    overlays: list[TextOverlay] = []
    for section in sections:
        rects = [
            layout[name]
            for name in iter_param_names(section)
            if name in layout
        ]
        if not rects:
            continue
        sx = min(r.x for r in rects)
        sw = max(r.x + r.w for r in rects) - sx
        if section.label:
            glyph_w = GLYPH_W * len(section.label)
            label_x = sx + (sw - glyph_w) // 2
            overlays.append(TextOverlay(text=section.label, x=label_x, y=0))
        for group in section.groups:
            for item in group.items:
                if item.label is None or item.param_name not in layout:
                    continue
                rect = layout[item.param_name]
                overlays.append(
                    TextOverlay(
                        text=item.label,
                        x=rect.x,
                        y=rect.y - GLYPH_H - 1,
                    )
                )
    return overlays


def layout_to_flat_order(layout: dict[str, Rect]) -> list[str]:
    """Return parameter names in layout order (for flattening weights)."""
    return sorted(layout.keys(), key=lambda n: (layout[n].x, layout[n].y))


def place_weights_on_canvas(
    weights: dict[str, np.ndarray],
    layout: dict[str, Rect],
    width: int,
    height: int,
    normalize_per_rect: bool = True,
) -> np.ndarray:
    """Place each parameter at its layout `Rect` on a 2D canvas.

    Pixels outside any rect (gutters, label bands, padding) are zero. The
    first ``rect.count`` cells of each rect (row-major) hold the parameter's
    flat values; remaining cells inside the rect are also zero (since
    ``rect.h * rect.w`` >= ``rect.count``).

    When ``normalize_per_rect`` is True (default), each rect's values are
    rescaled by dividing by their own ``max(|value|)`` so that every matrix
    spans the full ±1 range. This makes per-matrix structure visible in
    the colormapped output, at the cost of losing the global scale
    comparison between matrices. When False, raw values are placed.
    """
    canvas = np.zeros((height, width), dtype=np.float32)
    for name, rect in layout.items():
        if name not in weights:
            continue
        flat = np.asarray(weights[name], dtype=np.float32).ravel()
        if normalize_per_rect:
            scale = float(np.max(np.abs(flat))) if flat.size > 0 else 0.0
            if scale > 1e-10:
                flat = flat / scale
        block = np.zeros(rect.h * rect.w, dtype=np.float32)
        n = min(rect.count, flat.size, block.size)
        block[:n] = flat[:n]
        canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w] = block.reshape(rect.h, rect.w)
    return canvas


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
