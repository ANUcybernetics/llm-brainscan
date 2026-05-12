import numpy as np
import pytest

from brainscan.layout import (
    GUTTER,
    HEIGHT,
    LAYOUT_HEIGHT,
    TEXT_STRIP_HEIGHT,
    WIDTH,
    Group,
    Item,
    Rect,
    Section,
    TextOverlay,
    compute_layout,
    compute_text_overlays,
    default_sections,
    iter_param_names,
    layout_summary,
    layout_to_flat_order,
    place_weights_on_canvas,
)
from brainscan.model import GPT


@pytest.fixture
def model_param_counts():
    model = GPT()
    return {name: p.numel() for name, p in model.named_parameters()}


@pytest.fixture
def default_layout(model_param_counts):
    return compute_layout(model_param_counts)


class TestItemGroup:
    def test_item_holds_param_name_and_label(self):
        it = Item(param_name="blocks.0.attn.c_attn.weight", label="qkv")
        assert it.param_name == "blocks.0.attn.c_attn.weight"
        assert it.label == "qkv"

    def test_item_label_can_be_none(self):
        it = Item(param_name="ln_f.weight", label=None)
        assert it.label is None

    def test_group_holds_label_and_items(self):
        g = Group(label="attn", items=[Item("a", "qkv"), Item("b", None)])
        assert g.label == "attn"
        assert len(g.items) == 2

    def test_section_holds_groups(self):
        s = Section(label="block_0", groups=[Group("attn", [Item("a", None)])])
        assert s.label == "block_0"
        assert len(s.groups) == 1
        assert s.groups[0].label == "attn"

    def test_iter_param_names_flattens(self):
        s = Section(
            label="x",
            groups=[
                Group("g1", [Item("a", None), Item("b", "lab")]),
                Group("g2", [Item("c", None)]),
            ],
        )
        assert list(iter_param_names(s)) == ["a", "b", "c"]


class TestDefaultSections:
    def test_has_embed_blocks_output(self):
        sections = default_sections()
        labels = [s.label for s in sections]
        assert labels[0] == "EMBED"
        assert labels[-1] == "OUT"
        assert all(f"BLK {i}" in labels for i in range(8))

    def test_section_count(self):
        assert len(default_sections()) == 10  # embed + 8 blocks + output

    def test_custom_layer_count(self):
        assert len(default_sections(n_layer=4)) == 6  # embed + 4 blocks + output

    def test_all_model_params_covered(self, model_param_counts):
        sections = default_sections()
        all_names: set[str] = set()
        for section in sections:
            all_names.update(iter_param_names(section))
        for name in model_param_counts:
            assert name in all_names, f"Parameter {name} not in any section"

    def test_block_items_carry_matrix_labels(self):
        sections = default_sections()
        for i in range(8):
            blk = next(s for s in sections if s.label == f"BLK {i}")
            attn = next(g for g in blk.groups if g.label == "attn")
            mlp = next(g for g in blk.groups if g.label == "mlp")
            attn_labels = {it.param_name: it.label for it in attn.items}
            mlp_labels = {it.param_name: it.label for it in mlp.items}
            p = f"blocks.{i}"
            assert attn_labels[f"{p}.ln_1.weight"] is None
            assert attn_labels[f"{p}.ln_1.bias"] is None
            assert attn_labels[f"{p}.attn.c_attn.weight"] == "qkv"
            assert attn_labels[f"{p}.attn.c_proj.weight"] == "proj"
            assert mlp_labels[f"{p}.ln_2.weight"] is None
            assert mlp_labels[f"{p}.ln_2.bias"] is None
            assert mlp_labels[f"{p}.mlp.c_fc.weight"] == "up"
            assert mlp_labels[f"{p}.mlp.c_proj.weight"] == "down"


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
            assert rect.y + rect.h <= LAYOUT_HEIGHT, f"{rect.name} exceeds layout height"

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


class TestGutterKnobs:
    def _params(self):
        return {"a": 8, "b": 8, "c": 8, "d": 8}

    def _two_group_section(self) -> list[Section]:
        return [
            Section(
                "S",
                groups=[
                    Group("g1", [Item("a", None), Item("b", None)]),
                    Group("g2", [Item("c", None), Item("d", None)]),
                ],
            )
        ]

    def test_item_gutter_between_unlabelled_items(self):
        layout = compute_layout(
            self._params(),
            sections=self._two_group_section(),
            width=20, height=200,
            section_gutter=0, group_gutter=0, item_gutter=7,
        )
        a, b = layout["a"], layout["b"]
        assert b.y - (a.y + a.h) == 7

    def test_group_gutter_between_groups(self):
        layout = compute_layout(
            self._params(),
            sections=self._two_group_section(),
            width=20, height=200,
            section_gutter=0, group_gutter=11, item_gutter=2,
        )
        b = layout["b"]
        c = layout["c"]
        assert c.y - (b.y + b.h) == 11

    def test_section_gutter_between_sections(self):
        params = {"a": 4, "b": 4}
        sections = [
            Section("L", groups=[Group("", [Item("a", None)])]),
            Section("R", groups=[Group("", [Item("b", None)])]),
        ]
        layout = compute_layout(
            params, sections=sections, width=200, height=20,
            section_gutter=13, group_gutter=0, item_gutter=0,
        )
        a = layout["a"]
        b = layout["b"]
        assert b.x - (a.x + a.w) == 13

    def test_label_gap_above_labelled_items(self):
        params = {"a": 4, "b": 4, "c": 4}
        sections = [
            Section(
                "S",
                groups=[
                    Group("g", [
                        Item("a", None),
                        Item("b", "lab"),
                        Item("c", None),
                    ])
                ],
            )
        ]
        layout = compute_layout(
            params, sections=sections, width=20, height=200,
            section_gutter=0, group_gutter=0, item_gutter=3, label_gap_px=17,
        )
        # b is labelled --- its leading gap is label_gap_px
        assert layout["b"].y - (layout["a"].y + layout["a"].h) == 17
        # c is unlabelled --- its leading gap is item_gutter
        assert layout["c"].y - (layout["b"].y + layout["b"].h) == 3

    def test_label_gap_does_not_apply_to_first_item_in_group(self):
        # If the first item of a group carries a label, no leading gap is added
        # (the spec says the first item in each group does not get a leading gap).
        params = {"a": 4, "b": 4}
        sections = [
            Section(
                "S",
                groups=[
                    Group("g", [Item("a", "lab")]),
                    Group("g2", [Item("b", None)]),
                ],
            )
        ]
        layout = compute_layout(
            params, sections=sections, width=20, height=200,
            section_gutter=0, group_gutter=5, item_gutter=2, label_gap_px=17,
        )
        assert layout["a"].y == 0
        assert layout["b"].y - (layout["a"].y + layout["a"].h) == 5

    def test_section_label_height_pushes_items_down(self):
        params = {"a": 4, "b": 4}
        sections = [
            Section("S", groups=[Group("", [Item("a", None), Item("b", None)])])
        ]
        layout = compute_layout(
            params, sections=sections, width=20, height=200,
            section_gutter=0, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=24,
        )
        assert layout["a"].y == 24
        assert layout["b"].y == layout["a"].y + layout["a"].h

    def test_min_section_width_floor_applied(self):
        # Tiny sections (only 4 params total) would naturally compute a 1px
        # column; the floor should widen them to min_section_width.
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(
            params, sections=sections, width=200, height=200,
            section_gutter=0, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=0,
            min_section_width=80,
        )
        assert layout["a"].w == 80

    def test_min_section_width_does_not_exceed_natural_width(self):
        # When the natural width is already greater than the floor,
        # min_section_width does not change anything.
        params = {"a": 1000}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(
            params, sections=sections, width=200, height=10,
            section_gutter=0, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=0,
            min_section_width=20,
        )
        assert layout["a"].w >= 100  # 1000 / 10 = 100, well above 20


class TestTextStripConstants:
    def test_layout_height_plus_strip_equals_height(self):
        assert LAYOUT_HEIGHT + TEXT_STRIP_HEIGHT == HEIGHT

    def test_text_strip_height(self):
        assert TEXT_STRIP_HEIGHT == 224

    def test_band_heights_sum_to_text_strip(self):
        """The three band heights in train.py must sum to TEXT_STRIP_HEIGHT;
        otherwise weights overlap the text strip (or leave a dark gap).
        """
        from brainscan.train import AUDIENCE_HEIGHT, CAPTIONS_HEIGHT, MODEL_LANE_HEIGHT

        assert (
            AUDIENCE_HEIGHT + MODEL_LANE_HEIGHT + CAPTIONS_HEIGHT
            == TEXT_STRIP_HEIGHT
        )


class TestCustomLayout:
    def test_custom_sections(self):
        params = {"a": 100, "b": 200, "c": 300}
        sections = [
            Section("left", groups=[Group("", [Item("a", None)])]),
            Section("right", groups=[Group("", [Item("b", None), Item("c", None)])]),
        ]
        layout = compute_layout(params, sections=sections, width=100, height=100)
        assert layout["a"].x < layout["b"].x

    def test_single_section(self):
        params = {"a": 50, "b": 50}
        sections = [Section("all", groups=[Group("", [Item("a", None), Item("b", None)])])]
        layout = compute_layout(params, sections=sections, width=20, height=20)
        assert layout["a"].x == layout["b"].x
        assert layout["a"].y < layout["b"].y

    def test_missing_params_skipped(self):
        params = {"a": 100}
        sections = [Section("s", groups=[Group("", [Item("a", None), Item("missing", None)])])]
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


class TestPlaceWeightsOnCanvas:
    def test_param_lands_at_rect_position(self):
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(
            params, sections=sections, width=10, height=10,
            section_gutter=0, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=0,
        )
        weights = {"a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        canvas = place_weights_on_canvas(weights, layout, width=10, height=10, normalize_per_rect=False)
        # rect for "a" starts at (0, 0); width is what compute_layout decided.
        rect = layout["a"]
        # The rect.count cells (in row-major) inside the rect hold the values;
        # the rest of the rect is zero-padded; pixels outside the rect are zero.
        block = canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w].ravel()
        np.testing.assert_array_equal(block[:4], [1.0, 2.0, 3.0, 4.0])
        if block.size > 4:
            np.testing.assert_array_equal(block[4:], 0.0)

    def test_gutter_pixels_are_zero(self):
        # Two adjacent sections separated by a section gutter --- the gutter
        # column must be zero on the canvas, regardless of source values.
        params = {"a": 4, "b": 4}
        sections = [
            Section("L", groups=[Group("", [Item("a", None)])]),
            Section("R", groups=[Group("", [Item("b", None)])]),
        ]
        layout = compute_layout(
            params, sections=sections, width=20, height=4,
            section_gutter=3, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=0,
        )
        weights = {
            "a": np.full(4, 7.0, dtype=np.float32),
            "b": np.full(4, 9.0, dtype=np.float32),
        }
        canvas = place_weights_on_canvas(weights, layout, width=20, height=4, normalize_per_rect=False)
        a, b = layout["a"], layout["b"]
        # Pixels strictly between a.x+a.w and b.x are gutter and must be zero.
        for x in range(a.x + a.w, b.x):
            assert canvas[0, x] == 0.0, f"gutter pixel ({x},0) not zero"

    def test_canvas_shape_and_dtype(self):
        layout: dict = {}
        canvas = place_weights_on_canvas({}, layout, width=8, height=4, normalize_per_rect=False)
        assert canvas.shape == (4, 8)
        assert canvas.dtype == np.float32

    def test_missing_param_in_weights_skipped(self):
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(params, sections=sections, width=10, height=10)
        # weights dict missing "a" --- should not raise, canvas all zero
        canvas = place_weights_on_canvas({}, layout, width=10, height=10, normalize_per_rect=False)
        assert (canvas == 0.0).all()

    def test_normalize_per_rect_scales_each_rect_to_unit_range(self):
        params = {"a": 4, "b": 4}
        sections = [
            Section("S", groups=[Group("", [Item("a", None), Item("b", None)])])
        ]
        layout = compute_layout(
            params, sections=sections, width=10, height=20,
            section_gutter=0, group_gutter=0, item_gutter=0,
            label_gap_px=0, section_label_height=0,
        )
        # Two rects with different scales: a peaks at 0.5, b peaks at 4.0
        weights = {
            "a": np.array([0.1, -0.2, 0.5, -0.3], dtype=np.float32),
            "b": np.array([1.0, -2.0, 4.0, -1.5], dtype=np.float32),
        }
        canvas = place_weights_on_canvas(
            weights, layout, width=10, height=20, normalize_per_rect=True,
        )
        a_rect = layout["a"]
        b_rect = layout["b"]
        a_block = canvas[a_rect.y:a_rect.y + a_rect.h, a_rect.x:a_rect.x + a_rect.w].ravel()
        b_block = canvas[b_rect.y:b_rect.y + b_rect.h, b_rect.x:b_rect.x + b_rect.w].ravel()
        # First a.count cells of a's rect = a's values / 0.5
        np.testing.assert_allclose(a_block[:4], [0.2, -0.4, 1.0, -0.6])
        # First b.count cells of b's rect = b's values / 4.0
        np.testing.assert_allclose(b_block[:4], [0.25, -0.5, 1.0, -0.375])

    def test_normalize_per_rect_handles_all_zero_rect(self):
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(params, sections=sections, width=10, height=10)
        # All-zero weights should not raise (no division by zero)
        weights = {"a": np.zeros(4, dtype=np.float32)}
        canvas = place_weights_on_canvas(
            weights, layout, width=10, height=10, normalize_per_rect=True,
        )
        rect = layout["a"]
        block = canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
        assert (block == 0.0).all()

    def test_normalize_per_rect_default_is_true(self):
        # Verifies the production default: normalisation is on by default.
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(params, sections=sections, width=10, height=10)
        weights = {"a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
        canvas = place_weights_on_canvas(weights, layout, width=10, height=10)
        rect = layout["a"]
        block = canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w].ravel()
        # Default normalise: values divided by 4.0 → [0.25, 0.5, 0.75, 1.0]
        np.testing.assert_allclose(block[:4], [0.25, 0.5, 0.75, 1.0])


class TestComputeTextOverlays:
    def _setup(self):
        # Mirror the production shape: a labelled item is never first in its
        # group, so its label gap sits below other content (not inside the
        # section-label band).
        params = {
            "a": 16,
            "blocks.0.ln_1.weight": 4,
            "blocks.0.attn.c_attn.weight": 64,
        }
        sections = [
            Section("EMBED", groups=[Group("", [Item("a", None)])]),
            Section(
                "BLK 0",
                groups=[
                    Group(
                        "attn",
                        [
                            Item("blocks.0.ln_1.weight", None),
                            Item("blocks.0.attn.c_attn.weight", "qkv"),
                        ],
                    ),
                ],
            ),
        ]
        layout = compute_layout(
            params, sections=sections, width=400, height=400,
            section_gutter=10, group_gutter=0, item_gutter=2,
            label_gap_px=16, section_label_height=24,
            min_section_width=80,
        )
        return layout, sections

    def test_section_labels_emitted(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        texts = [o.text for o in overlays]
        assert "EMBED" in texts
        assert "BLK 0" in texts

    def test_matrix_label_emitted_for_labelled_item(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        texts = [o.text for o in overlays]
        assert "qkv" in texts

    def test_no_label_for_unlabelled_item(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        # "a" is unlabelled, so its param name must not appear as overlay text
        assert all(o.text != "a" for o in overlays)

    def test_section_label_within_label_band(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        embed = next(o for o in overlays if o.text == "EMBED")
        # 8x16 glyph at 1× scale; band is 24 px tall starting at y=0
        assert 0 <= embed.y <= 24 - 16

    def test_section_label_horizontally_centered(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        embed = next(o for o in overlays if o.text == "EMBED")
        # Embed section: column starts at x=0, width = layout["a"].w
        rect = layout["a"]
        glyph_w = 8 * len(embed.text)
        expected_x = rect.x + (rect.w - glyph_w) // 2
        assert embed.x == expected_x

    def test_matrix_label_above_matrix(self):
        layout, sections = self._setup()
        overlays = compute_text_overlays(layout, sections)
        qkv = next(o for o in overlays if o.text == "qkv")
        rect = layout["blocks.0.attn.c_attn.weight"]
        # Label baseline 1 px above the matrix top, glyph height 16
        assert qkv.y == rect.y - 16 - 1
        # Left-aligned with the matrix
        assert qkv.x == rect.x
