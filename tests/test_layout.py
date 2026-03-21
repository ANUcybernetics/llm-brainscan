import pytest

from brainscan.layout import (
    GUTTER,
    HEIGHT,
    WIDTH,
    Rect,
    Section,
    compute_layout,
    default_sections,
    layout_summary,
    layout_to_flat_order,
)
from brainscan.model import GPT


@pytest.fixture
def model_param_counts():
    model = GPT()
    return {name: p.numel() for name, p in model.named_parameters()}


@pytest.fixture
def default_layout(model_param_counts):
    return compute_layout(model_param_counts)


class TestDefaultSections:
    def test_has_embed_blocks_output(self):
        sections = default_sections()
        labels = [s.label for s in sections]
        assert labels[0] == "embed"
        assert labels[-1] == "output"
        assert all(f"block_{i}" in labels for i in range(8))

    def test_section_count(self):
        assert len(default_sections()) == 10  # embed + 8 blocks + output

    def test_custom_layer_count(self):
        assert len(default_sections(n_layer=4)) == 6  # embed + 4 blocks + output

    def test_all_model_params_covered(self, model_param_counts):
        sections = default_sections()
        all_names = set()
        for section in sections:
            all_names.update(section.param_names)
        for name in model_param_counts:
            assert name in all_names, f"Parameter {name} not in any section"


class TestComputeLayout:
    def test_all_params_placed(self, default_layout, model_param_counts):
        for name in model_param_counts:
            assert name in default_layout, f"Missing layout for {name}"

    def test_total_count_matches(self, default_layout, model_param_counts):
        layout_total = sum(r.count for r in default_layout.values())
        model_total = sum(model_param_counts.values())
        assert layout_total == model_total

    def test_no_overlaps(self, default_layout):
        rects = list(default_layout.values())
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                x_overlap = a.x < b.x + b.w and b.x < a.x + a.w
                y_overlap = a.y < b.y + b.h and b.y < a.y + a.h
                assert not (x_overlap and y_overlap), (
                    f"Overlap between {a.name} ({a.x},{a.y},{a.w},{a.h}) "
                    f"and {b.name} ({b.x},{b.y},{b.w},{b.h})"
                )

    def test_fits_within_canvas(self, default_layout):
        for rect in default_layout.values():
            assert rect.x >= 0
            assert rect.y >= 0
            assert rect.x + rect.w <= WIDTH, f"{rect.name} exceeds width"
            assert rect.y + rect.h <= HEIGHT, f"{rect.name} exceeds height"

    def test_rect_area_covers_params(self, default_layout):
        for rect in default_layout.values():
            area = rect.w * rect.h
            assert area >= rect.count, f"{rect.name}: area {area} < count {rect.count}"

    def test_left_to_right_ordering(self, default_layout):
        embed_x = default_layout["wte.weight"].x
        block0_x = default_layout["blocks.0.attn.c_attn.weight"].x
        block7_x = default_layout["blocks.7.attn.c_attn.weight"].x
        output_x = default_layout["lm_head.weight"].x
        assert embed_x < block0_x < block7_x < output_x

    def test_blocks_have_ascending_x(self, default_layout):
        block_xs = []
        for i in range(8):
            x = default_layout[f"blocks.{i}.attn.c_attn.weight"].x
            block_xs.append(x)
        assert block_xs == sorted(block_xs)
        assert len(set(block_xs)) == 8  # all different

    def test_within_block_vertical_stacking(self, default_layout):
        for i in range(8):
            prefix = f"blocks.{i}"
            names = [
                f"{prefix}.ln_1.weight",
                f"{prefix}.attn.c_attn.weight",
                f"{prefix}.attn.c_proj.weight",
                f"{prefix}.ln_2.weight",
                f"{prefix}.mlp.c_fc.weight",
                f"{prefix}.mlp.c_proj.weight",
            ]
            ys = [default_layout[n].y for n in names]
            assert ys == sorted(ys), f"Block {i} matrices not stacked top-to-bottom"

    def test_gutters_between_matrices(self, default_layout):
        for i in range(8):
            prefix = f"blocks.{i}"
            attn = default_layout[f"{prefix}.attn.c_attn.weight"]
            proj = default_layout[f"{prefix}.attn.c_proj.weight"]
            gap = proj.y - (attn.y + attn.h)
            assert gap >= GUTTER, f"Block {i} attn->proj gap {gap} < {GUTTER}"

    def test_all_block_columns_same_width(self, default_layout):
        widths = set()
        for i in range(8):
            w = default_layout[f"blocks.{i}.attn.c_attn.weight"].w
            widths.add(w)
        assert len(widths) == 1, f"Block columns have different widths: {widths}"

    def test_gutters_between_sections(self, default_layout):
        embed_right = default_layout["wte.weight"].x + default_layout["wte.weight"].w
        block0_left = default_layout["blocks.0.ln_1.weight"].x
        gap = block0_left - embed_right
        assert gap >= GUTTER, f"Section gap {gap} < {GUTTER}"

    def test_embed_section_narrower_than_blocks(self, default_layout):
        embed_w = default_layout["wte.weight"].w
        block_w = default_layout["blocks.0.attn.c_attn.weight"].w
        assert embed_w < block_w


class TestCustomLayout:
    def test_custom_sections(self):
        params = {"a": 100, "b": 200, "c": 300}
        sections = [
            Section("left", ["a"]),
            Section("right", ["b", "c"]),
        ]
        layout = compute_layout(params, sections=sections, width=100, height=100)
        assert layout["a"].x < layout["b"].x

    def test_single_section(self):
        params = {"a": 50, "b": 50}
        sections = [Section("all", ["a", "b"])]
        layout = compute_layout(params, sections=sections, width=20, height=20)
        assert layout["a"].x == layout["b"].x
        assert layout["a"].y < layout["b"].y

    def test_missing_params_skipped(self):
        params = {"a": 100}
        sections = [Section("s", ["a", "missing"])]
        layout = compute_layout(params, sections=sections, width=50, height=50)
        assert "a" in layout
        assert "missing" not in layout


class TestLayoutHelpers:
    def test_flat_order_left_to_right(self, default_layout):
        order = layout_to_flat_order(default_layout)
        xs = [default_layout[name].x for name in order]
        assert xs == sorted(xs)

    def test_flat_order_contains_all(self, default_layout, model_param_counts):
        order = layout_to_flat_order(default_layout)
        assert set(order) == set(model_param_counts.keys())

    def test_summary_contains_all_names(self, default_layout):
        summary = layout_summary(default_layout)
        for name in default_layout:
            assert name in summary

    def test_rect_to_dict(self):
        r = Rect(x=10, y=20, w=30, h=40, count=100, name="test")
        d = r.to_dict()
        assert d == {"x": 10, "y": 20, "w": 30, "h": 40, "count": 100}
