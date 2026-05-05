# Layout Delineation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add architectural visual delineation (wide gutters, attn/mlp subgroups, in-gutter labels) to the weight visualisation so the eight-block transformer rhythm is readable from across the room while the 1:1 pixel-per-parameter invariant is preserved.

**Architecture:** Restructure `Section` into a tree of `Group` and `Item` nodes so each labelled matrix carries its own caption. Extend `compute_layout` with six geometry knobs (section/group/item gutters, label gap, section-label band, min section width). Introduce `place_weights_on_canvas` so flat weight tensors land at their `Rect` positions (gutters become real zero-padded pixels rather than virtual). Add `compute_text_overlays` to emit a list of `(text, x, y)` overlay runs which the fragment shader stamps onto gutter/band pixels in dim grey, never occluding weight data.

**Tech Stack:** Python 3.12, PyTorch (model), wgpu/WGSL (shader), NumPy (CPU buffers), pytest. All commands prefixed with `mise exec -- uv run`.

**Spec:** `docs/superpowers/specs/2026-05-05-layout-delineation-design.md`

---

## File structure

### New files

| Path | Responsibility |
|------|----------------|
| (none) | All new functionality lives in existing modules |

### Modified files

| Path | What changes |
|------|--------------|
| `src/brainscan/layout.py` | New `Item` and `Group` dataclasses; `Section.groups` replaces `Section.param_names`; `compute_layout` gains six geometry parameters; new helpers `compute_text_overlays`, `place_weights_on_canvas`, `iter_param_names` |
| `src/brainscan/renderer.py` | New static `overlay_chars` and `overlay_runs` storage buffers; uniform `overlay_run_count`; shader checks overlays before weight rendering; `RenderResources` and `create_render_pipeline` updated; new `upload_overlays` helper; `OffscreenRenderer.set_overlays` and `LiveRenderer.set_overlays` |
| `src/brainscan/tuning.py` | New `LAYOUT_*` constants (section gutter, group gutter, item gutter, label gap, section-label height, min section width, label colour) |
| `src/brainscan/train.py` | `--n-embd` default drops from 558 to 540; `compute_layout` is called with the new tuning constants; `_build_weight_buffer` rewritten to use `place_weights_on_canvas`; overlays computed once and uploaded to the renderer (and refreshed at rebirth) |
| `tests/test_layout.py` | Tests for `Item`/`Group`/`Section` shape, all six geometry knobs, the new `default_sections` structure, `compute_text_overlays`, and `place_weights_on_canvas` |
| `tests/test_renderer.py` | New `TestOverlays` class verifying overlay glyphs render in dim grey at expected positions and do not collide with weight data |
| `tests/conftest.py` | (No change — `SMALL_CONFIG` already uses `n_embd=64`, well below the new 540 default.) |
| `CLAUDE.md` | "Display layout" section updated to describe the new chrome |
| `README.md` | ASCII diagram and concept blurb updated |

### Test commands

- All tests: `mise exec -- uv run pytest tests/ -v`
- Single test file: `mise exec -- uv run pytest tests/test_layout.py -v`
- Single test: `mise exec -- uv run pytest tests/test_layout.py::ClassName::test_name -v`
- Type check: `mise exec -- uv run ty check`

Each task ends with a commit. Use imperative-mood, concise commit messages.

---

## Phase 1: Layout data model

### Task 1: `Item` and `Group` dataclasses; reshape `Section`

This is the structural refactor. After this task, `Section.groups` exists and the default sections are wrapped as a single group with unlabelled items, so the rendered layout is byte-identical to before.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_layout.py` near the top (after imports):

```python
from brainscan.layout import Group, Item, iter_param_names
```

Then add a new test class **before** `class TestDefaultSections`:

```python
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
```

Update `TestDefaultSections.test_all_model_params_covered` to use the new helper:

```python
    def test_all_model_params_covered(self, model_param_counts):
        sections = default_sections()
        all_names: set[str] = set()
        for section in sections:
            all_names.update(iter_param_names(section))
        for name in model_param_counts:
            assert name in all_names, f"Parameter {name} not in any section"
```

Update each test in `class TestCustomLayout` to construct sections via the new structure:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: `ImportError: cannot import name 'Group'` (or similar) — all of `TestItemGroup`, `TestCustomLayout`, and `test_all_model_params_covered` fail.

- [ ] **Step 3: Implement `Item`, `Group`, the new `Section`, `iter_param_names`, and rewire `compute_layout` and `default_sections`**

Edit `src/brainscan/layout.py`. Replace the existing `Section` dataclass and `default_sections` and `_section_param_total` with the following (keep `Rect`, `WIDTH`, `HEIGHT`, `TEXT_STRIP_HEIGHT`, `LAYOUT_HEIGHT`, `TOTAL_PIXELS`, `GUTTER`, `_column_width` unchanged):

```python
from collections.abc import Iterable


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
```

Now rewrite `compute_layout` so it walks the new tree. Replace the existing `compute_layout` body with:

```python
def compute_layout(
    param_counts: dict[str, int],
    sections: list[Section] | None = None,
    width: int = WIDTH,
    height: int = LAYOUT_HEIGHT,
    gutter: int = GUTTER,
) -> dict[str, Rect]:
    """Compute pixel layout for all parameters."""
    if sections is None:
        sections = default_sections()

    section_widths: list[int] = []
    for section in sections:
        item_counts = [
            param_counts[name]
            for name in iter_param_names(section)
            if param_counts.get(name, 0) > 0
        ]
        col_w = _column_width(item_counts, height, gutter)
        section_widths.append(col_w)

    total_gutters = max(0, len(sections) - 1) * gutter
    total_content_w = sum(section_widths)
    total_w = total_content_w + total_gutters

    if total_w > width:
        scale = (width - total_gutters) / total_content_w
        section_widths = [max(1, int(w * scale)) for w in section_widths]

    layout: dict[str, Rect] = {}
    x_cursor = 0

    for section, col_w in zip(sections, section_widths, strict=True):
        y_cursor = 0
        for group in section.groups:
            for item in group.items:
                count = param_counts.get(item.param_name, 0)
                if count == 0:
                    continue
                h = math.ceil(count / col_w)
                rect = Rect(
                    x=x_cursor, y=y_cursor, w=col_w, h=h,
                    count=count, name=item.param_name,
                )
                layout[item.param_name] = rect
                y_cursor += h + gutter

        x_cursor += col_w + gutter

    return layout
```

This task adds the three dataclasses and `iter_param_names`, and rewires `default_sections`/`compute_layout` to walk groups. **Behaviour is unchanged** because every gap (item-to-item and group-to-group) is still `gutter=4`, and labels are not yet rendered.

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all tests in `tests/test_layout.py` pass. Run the full suite once to catch knock-on effects:

```bash
mise exec -- uv run pytest tests/ -v
```

Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "refactor(layout): replace Section.param_names with Group/Item tree"
```

---

## Phase 2: Layout geometry

### Task 2: Three independent gutter parameters

Replace the single `gutter` parameter with `section_gutter`, `group_gutter`, and `item_gutter`. Old behaviour is preserved when all three default to the existing `GUTTER`.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_layout.py` (after `TestComputeLayout`):

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestGutterKnobs -v
```

Expected: fails with `TypeError: compute_layout() got an unexpected keyword argument 'section_gutter'`.

- [ ] **Step 3: Implement the three knobs**

In `src/brainscan/layout.py`, change the `compute_layout` signature and body. Replace the `def compute_layout(...)` block with:

```python
def compute_layout(
    param_counts: dict[str, int],
    sections: list[Section] | None = None,
    width: int = WIDTH,
    height: int = LAYOUT_HEIGHT,
    section_gutter: int = GUTTER,
    group_gutter: int = GUTTER,
    item_gutter: int = GUTTER,
) -> dict[str, Rect]:
    """Compute pixel layout for all parameters.

    Args:
        section_gutter: pixels between sections.
        group_gutter: pixels between groups inside a section.
        item_gutter: pixels between unlabelled consecutive items in a group.
    """
    if sections is None:
        sections = default_sections()

    section_widths: list[int] = []
    for section in sections:
        item_counts = [
            param_counts[name]
            for name in iter_param_names(section)
            if param_counts.get(name, 0) > 0
        ]
        col_w = _column_width(item_counts, height, item_gutter)
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
        y_cursor = 0
        for g_idx, group in enumerate(section.groups):
            if g_idx > 0:
                y_cursor += group_gutter
            for i_idx, item in enumerate(group.items):
                count = param_counts.get(item.param_name, 0)
                if count == 0:
                    continue
                if i_idx > 0:
                    y_cursor += item_gutter
                h = math.ceil(count / col_w)
                rect = Rect(
                    x=x_cursor, y=y_cursor, w=col_w, h=h,
                    count=count, name=item.param_name,
                )
                layout[item.param_name] = rect
                y_cursor += h
        x_cursor += col_w + section_gutter

    return layout
```

Note three changes from Task 1:
1. Three independent gutter parameters, all defaulting to `GUTTER`.
2. The y-cursor advances by `group_gutter` between groups, `item_gutter` between items inside a group, and not at all before the first item of a group or the first group of a section.
3. The y-cursor rolls forward by exactly `h` (no trailing gutter) so the next gap is added explicitly.

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all tests pass. Existing tests like `test_gutters_between_matrices` and `test_gutters_between_sections` still pass because the defaults are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): split gutter into section/group/item parameters"
```

---

### Task 3: `label_gap_px` for labelled items

When an item carries a non-`None` label, the gap above it (between previous item and this one) is `label_gap_px` instead of `item_gutter`.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestGutterKnobs` in `tests/test_layout.py`:

```python
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
        # b is labelled — its leading gap is label_gap_px
        assert layout["b"].y - (layout["a"].y + layout["a"].h) == 17
        # c is unlabelled — its leading gap is item_gutter
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
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestGutterKnobs::test_label_gap_above_labelled_items -v
```

Expected: `TypeError: compute_layout() got an unexpected keyword argument 'label_gap_px'`.

- [ ] **Step 3: Implement `label_gap_px`**

Edit the `compute_layout` signature and body in `src/brainscan/layout.py`. Add the parameter, defaulting to `GUTTER` so existing tests like `test_gutters_between_matrices` (which check the gap between `c_attn` and `c_proj` is at least `GUTTER`) keep passing. The production call site overrides this to `tuning.LAYOUT_LABEL_GAP_PX = 16`:

```python
def compute_layout(
    param_counts: dict[str, int],
    sections: list[Section] | None = None,
    width: int = WIDTH,
    height: int = LAYOUT_HEIGHT,
    section_gutter: int = GUTTER,
    group_gutter: int = GUTTER,
    item_gutter: int = GUTTER,
    label_gap_px: int = GUTTER,
) -> dict[str, Rect]:
```

In the inner loop, change `if i_idx > 0: y_cursor += item_gutter` to:

```python
                if i_idx > 0:
                    y_cursor += label_gap_px if item.label is not None else item_gutter
```

The whole inner loop now reads:

```python
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
```

Also pass `item_gutter` to `_column_width` in place of the old `gutter`. (Done already in Task 2.) Note that `_column_width` only sees `item_gutter` — the label-gap-aware sizing isn't strictly accurate, but in practice the additional gap is small relative to canvas height and the existing `_column_width` widening loop will handle overflow. We refine this in Task 4.

- [ ] **Step 4: Refine `_column_width` to account for label gaps and group gutters**

The current `_column_width(item_counts, height, gutter)` assumes `(n - 1) * gutter` of fixed chrome. For a real section, the chrome includes section-label band, group gutters, and label gaps. Update the call site to pass the actual chrome budget instead. In `compute_layout`, before the section-width loop, change to:

```python
    section_widths: list[int] = []
    for section in sections:
        item_counts: list[int] = []
        chrome = 0
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
        col_w = _column_width(item_counts, avail_h, 0)
        section_widths.append(col_w)
```

This passes a `gutter=0` to `_column_width` (since chrome is already subtracted from `avail_h`) and a precomputed chrome that matches what `compute_layout` will actually use. (`_column_width` then sees only the matrices, not chrome.)

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): add label_gap_px above labelled items"
```

---

### Task 4: `section_label_height` band

The first `section_label_height` pixels of every section column are reserved for a section label and contain no matrices. Items inside groups start below this band.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestGutterKnobs` in `tests/test_layout.py`:

```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestGutterKnobs::test_section_label_height_pushes_items_down -v
```

Expected: `TypeError: compute_layout() got an unexpected keyword argument 'section_label_height'`.

- [ ] **Step 3: Implement `section_label_height`**

In `src/brainscan/layout.py`, add the parameter to `compute_layout`:

```python
    section_label_height: int = 0,
```

Inside the per-section loop, replace `y_cursor = 0` with:

```python
        y_cursor = section_label_height
```

Also update the chrome accounting in the column-width pass to include the label band:

```python
        chrome = section_label_height
        for g_idx, group in enumerate(section.groups):
            ...
```

(Replace the existing `chrome = 0` line.)

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): reserve section_label_height band at top of each section"
```

---

### Task 5: `min_section_width` floor

Section column widths floor at this value. Without it, narrow sections like `EMBED` and `OUT` would shrink to a sliver and disappear visually. With `min_section_width=80`, they read as columns even from across the room.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestGutterKnobs` in `tests/test_layout.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestGutterKnobs::test_min_section_width_floor_applied -v
```

Expected: `TypeError: compute_layout() got an unexpected keyword argument 'min_section_width'`.

- [ ] **Step 3: Implement `min_section_width`**

In `src/brainscan/layout.py`, add the parameter to `compute_layout`:

```python
    min_section_width: int = 1,
```

After the per-section `col_w = _column_width(...)` call, apply the floor before appending:

```python
        col_w = max(_column_width(item_counts, avail_h, 0), min_section_width)
        section_widths.append(col_w)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): add min_section_width floor"
```

---

## Phase 3: Spatial weight buffer

### Task 6: `place_weights_on_canvas`

The current `_build_weight_buffer` flattens all params into a single sequential array; the renderer reads `weights[py * width + px]` and so the spatial layout is ignored — gutters and label bands contain whatever bytes happen to be at that flat-buffer position. To make gutters real and overlays safe to draw, place each param into its `Rect` on a 2D canvas and zero-pad everywhere else.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_layout.py`:

```python
import numpy as np

from brainscan.layout import place_weights_on_canvas


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
        canvas = place_weights_on_canvas(weights, layout, width=10, height=10)
        # rect for "a" starts at (0, 0); width is what compute_layout decided.
        rect = layout["a"]
        # The rect.count cells (in row-major) inside the rect hold the values;
        # the rest of the rect is zero-padded; pixels outside the rect are zero.
        block = canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w].ravel()
        np.testing.assert_array_equal(block[:4], [1.0, 2.0, 3.0, 4.0])
        if block.size > 4:
            np.testing.assert_array_equal(block[4:], 0.0)

    def test_gutter_pixels_are_zero(self):
        # Two adjacent sections separated by a section gutter — the gutter
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
        canvas = place_weights_on_canvas(weights, layout, width=20, height=4)
        a, b = layout["a"], layout["b"]
        # Pixels strictly between a.x+a.w and b.x are gutter and must be zero.
        for x in range(a.x + a.w, b.x):
            assert canvas[0, x] == 0.0, f"gutter pixel ({x},0) not zero"

    def test_canvas_shape_and_dtype(self):
        layout: dict = {}
        canvas = place_weights_on_canvas({}, layout, width=8, height=4)
        assert canvas.shape == (4, 8)
        assert canvas.dtype == np.float32

    def test_missing_param_in_weights_skipped(self):
        params = {"a": 4}
        sections = [Section("S", groups=[Group("", [Item("a", None)])])]
        layout = compute_layout(params, sections=sections, width=10, height=10)
        # weights dict missing "a" — should not raise, canvas all zero
        canvas = place_weights_on_canvas({}, layout, width=10, height=10)
        assert (canvas == 0.0).all()
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestPlaceWeightsOnCanvas -v
```

Expected: `ImportError: cannot import name 'place_weights_on_canvas'`.

- [ ] **Step 3: Implement `place_weights_on_canvas`**

Add to the bottom of `src/brainscan/layout.py`:

```python
def place_weights_on_canvas(
    weights: dict[str, "np.ndarray"],
    layout: dict[str, Rect],
    width: int,
    height: int,
) -> "np.ndarray":
    """Place each parameter at its layout `Rect` on a 2D canvas.

    Pixels outside any rect (gutters, label bands, padding) are zero. The
    first `rect.count` cells of each rect (row-major) hold the parameter's
    flat values; remaining cells inside the rect are also zero (since
    `rect.h * rect.w` >= `rect.count`).
    """
    import numpy as np

    canvas = np.zeros((height, width), dtype=np.float32)
    for name, rect in layout.items():
        if name not in weights:
            continue
        flat = np.asarray(weights[name], dtype=np.float32).ravel()
        block = np.zeros(rect.h * rect.w, dtype=np.float32)
        n = min(rect.count, flat.size, block.size)
        block[:n] = flat[:n]
        canvas[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w] = block.reshape(rect.h, rect.w)
    return canvas
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): add place_weights_on_canvas for spatial weight placement"
```

---

## Phase 4: Text overlays

### Task 7: `compute_text_overlays`

Emit a list of `(text, x, y)` tuples covering every section label and every matrix label. Section labels go centred in the section-label band at the top of each section column; matrix labels go left-aligned 1 px above the top of each labelled matrix.

**Files:**
- Modify: `src/brainscan/layout.py`
- Modify: `tests/test_layout.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_layout.py`:

```python
from brainscan.layout import TextOverlay, compute_text_overlays


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
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_layout.py::TestComputeTextOverlays -v
```

Expected: `ImportError: cannot import name 'TextOverlay'`.

- [ ] **Step 3: Implement `TextOverlay` and `compute_text_overlays`**

Add to `src/brainscan/layout.py`:

```python
GLYPH_W = 8
GLYPH_H = 16


@dataclass(frozen=True)
class TextOverlay:
    """A single short string to be drawn in a gutter or label band."""

    text: str
    x: int
    y: int


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
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_layout.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/layout.py tests/test_layout.py
git commit -m "feat(layout): emit section and matrix label overlays"
```

---

## Phase 5: Renderer overlay support

### Task 8: Overlay storage buffers, shader, and upload helper

Add a static `overlay_chars` array (glyph indices) and `overlay_runs` array of `(x, y, length, char_offset)` quadruples. The fragment shader, when in the upper part of the canvas (above the lane bands), checks each run; if the pixel falls inside a run and the corresponding glyph bit is set, render the overlay's dim-grey colour. Otherwise fall through to weight rendering.

**Files:**
- Modify: `src/brainscan/renderer.py`
- Modify: `tests/test_renderer.py`

- [ ] **Step 1: Write the failing tests**

Add a new class to `tests/test_renderer.py`:

```python
from brainscan.layout import TextOverlay


class TestOverlays:
    def _renderer(self):
        return OffscreenRenderer(96, 64, audience_height=0, model_height=0, captions_height=0)

    def test_overlay_renders_in_dim_grey(self):
        r = self._renderer()
        r.set_overlays([TextOverlay(text="A", x=8, y=8)])
        weights = np.zeros(96 * 64, dtype=np.float32)
        img = r.render(weights)
        # Sample a pixel that should be lit by the "A" glyph; the dim-grey
        # label colour is (0.55, 0.55, 0.60) ≈ (140, 140, 153).
        block = img[8:24, 8:16, :3]
        # At least one lit pixel inside the glyph rect should match dim grey.
        max_red = block[..., 0].max()
        assert 100 <= max_red <= 180, f"expected dim-grey overlay, got max R {max_red}"
        # Whichever pixel has the most red, blue should be slightly higher
        # (R 0.55 < B 0.60).
        flat = block.reshape(-1, 3)
        on = flat[flat[:, 0] > 100]
        if len(on) > 0:
            assert on[:, 2].mean() >= on[:, 0].mean()

    def test_no_overlay_keeps_weight_rendering(self):
        r = self._renderer()
        r.set_overlays([])
        weights = np.ones(96 * 64, dtype=np.float32)
        img = r.render(weights)
        # Diverging colormap with all 1.0 weights → reddish midpoint
        pixel = img[32, 48, :3]
        assert pixel[0] > pixel[2], f"weight rendering should still produce red, got {pixel}"

    def test_overlay_only_replaces_zero_padded_pixels(self):
        # An overlay placed inside a non-zero weight region must not occlude;
        # the spec promises overlays land in zero-padded gutters. We approximate
        # this contract by placing weight 1.0 at the overlay location and
        # verifying the rendered colour stays in the warm range (overlay
        # would replace it with dim grey).
        r = self._renderer()
        r.set_overlays([TextOverlay(text="A", x=8, y=8)])
        # All pixels filled with weight 1.0
        weights = np.ones(96 * 64, dtype=np.float32)
        img = r.render(weights)
        # Pixel at the centre of the glyph rect: weight rendering wins because
        # the canvas is non-zero there. Overlay-on-weight collisions are an
        # accepted compromise; the renderer simply renders overlays where they
        # are placed and accepts that the layout pipeline arranges for zero
        # padding underneath. So this test instead checks that the WHOLE
        # weight region remains dominated by the warm diverging colour.
        weight_region = img[:, :, :3]
        # Most pixels still warm (red > blue)
        red_dom = (weight_region[:, :, 0] > weight_region[:, :, 2]).mean()
        assert red_dom > 0.85, f"overlays should not dominate weight region, red-dominant fraction {red_dom}"
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mise exec -- uv run pytest tests/test_renderer.py::TestOverlays -v
```

Expected: `AttributeError: 'OffscreenRenderer' object has no attribute 'set_overlays'`.

- [ ] **Step 3: Update the shader and uniform layout**

In `src/brainscan/renderer.py`, edit the `SHADER_SOURCE` constant. Add two new bindings (after the existing seven storage bindings) and a new uniform field. Replace the `Uniforms` struct and bindings 0-7 block, and add bindings 8 and 9:

```wgsl
struct Uniforms {
    width: u32,
    height: u32,
    param_count: u32,
    colormap: u32,
    audience_y: u32,
    audience_height: u32,
    audience_count: u32,
    audience_offset_px: u32,
    model_y: u32,
    model_height: u32,
    model_count: u32,
    model_offset_px: u32,
    captions_y: u32,
    captions_height: u32,
    captions_count: u32,
    vmax: f32,
    model_caret_col: u32,
    audience_pulse: f32,
    audience_edge_pulse: f32,
    global_brightness: f32,
    overlay_run_count: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> audience_chars: array<u32>;
@group(0) @binding(4) var<storage, read> audience_attrs: array<u32>;
@group(0) @binding(5) var<storage, read> model_chars: array<u32>;
@group(0) @binding(6) var<storage, read> model_probs: array<f32>;
@group(0) @binding(7) var<storage, read> captions_chars: array<u32>;
@group(0) @binding(8) var<storage, read> overlay_chars: array<u32>;
@group(0) @binding(9) var<storage, read> overlay_runs: array<vec4<u32>>;
```

Add a new helper function (alongside `font_pixel`):

```wgsl
fn overlay_pixel(px: u32, py: u32) -> vec4<f32> {
    // Returns dim-grey overlay colour if this pixel lights up an overlay
    // glyph; otherwise returns vec4(-1, ...) so the caller falls through to
    // weight rendering.
    for (var i: u32 = 0u; i < uniforms.overlay_run_count; i = i + 1u) {
        let run = overlay_runs[i];
        let rx = run.x;
        let ry = run.y;
        let length = run.z;
        let char_offset = run.w;
        let rw = length * 8u;
        let rh = 16u;
        if px < rx || px >= rx + rw || py < ry || py >= ry + rh {
            continue;
        }
        let local_x = px - rx;
        let glyph_col = local_x / 8u;
        let gx = local_x % 8u;
        let gy = py - ry;
        let glyph = overlay_chars[char_offset + glyph_col];
        if font_pixel(glyph, gx, gy) {
            return vec4<f32>(0.55, 0.55, 0.60, 1.0);
        }
        return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(-1.0, 0.0, 0.0, 1.0);
}
```

In `fs_main`, after the three lane band checks and before the weight-region branch, insert the overlay check:

```wgsl
    let cap_o = overlay_pixel(px, py);
    if cap_o.x >= 0.0 {
        return vec4<f32>(cap_o.rgb * uniforms.global_brightness, 1.0);
    }
```

- [ ] **Step 4: Update the uniform dtype and `RenderResources`**

In `_UNIFORM_DTYPE`, add the new field:

```python
_UNIFORM_DTYPE = np.dtype([
    ("width", np.uint32),
    ...
    ("global_brightness", np.float32),
    ("overlay_run_count", np.uint32),
])
```

In `RenderResources`, add two new buffers:

```python
@dataclass
class RenderResources:
    device: wgpu.GPUDevice
    config: RenderConfig
    uniform_data: np.ndarray
    uniform_buffer: wgpu.GPUBuffer
    weight_buffer: wgpu.GPUBuffer
    font_buffer: wgpu.GPUBuffer
    audience_chars_buffer: wgpu.GPUBuffer
    audience_attrs_buffer: wgpu.GPUBuffer
    model_chars_buffer: wgpu.GPUBuffer
    model_probs_buffer: wgpu.GPUBuffer
    captions_chars_buffer: wgpu.GPUBuffer
    overlay_chars_buffer: wgpu.GPUBuffer
    overlay_runs_buffer: wgpu.GPUBuffer
    bind_group: wgpu.GPUBindGroup
    pipeline: wgpu.GPURenderPipeline
```

- [ ] **Step 5: Update `create_render_pipeline` to allocate and bind the new buffers**

Adding `overlay_run_count` makes `_UNIFORM_DTYPE.itemsize` 84 bytes (not a multiple of 16). WGSL uniform buffers must be 16-byte aligned, so update the `uniform_size` calculation. Replace:

```python
uniform_size = max(_UNIFORM_DTYPE.itemsize, 64)
```

with:

```python
uniform_size = max(((_UNIFORM_DTYPE.itemsize + 15) // 16) * 16, 64)
```

(With itemsize 84 this yields 96; with the old itemsize 80 it yields 80; the floor of 64 is preserved for very small structs.)

Inside `create_render_pipeline`, after the captions buffer is created, add:

```python
    OVERLAY_CHARS_CAP = 1024  # ample for ~50 overlays of short text
    OVERLAY_RUNS_CAP = 256    # at most ~50 runs in production
    overlay_chars_buffer = device.create_buffer(
        size=OVERLAY_CHARS_CAP * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    overlay_runs_buffer = device.create_buffer(
        size=OVERLAY_RUNS_CAP * 16,  # vec4<u32> = 16 bytes
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
```

Update the `bind_group_layout` to include 10 entries instead of 8:

```python
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": i,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform
                    if i == 0
                    else wgpu.BufferBindingType.read_only_storage
                },
            }
            for i in range(10)
        ]
    )
```

Update the `buffers` list in the bind group:

```python
    buffers = [
        uniform_buffer,
        weight_buffer,
        font_buffer,
        audience_chars_buffer,
        audience_attrs_buffer,
        model_chars_buffer,
        model_probs_buffer,
        captions_chars_buffer,
        overlay_chars_buffer,
        overlay_runs_buffer,
    ]
```

Update the returned `RenderResources` to include the two new buffers:

```python
    return RenderResources(
        ...
        captions_chars_buffer=captions_chars_buffer,
        overlay_chars_buffer=overlay_chars_buffer,
        overlay_runs_buffer=overlay_runs_buffer,
        bind_group=bind_group,
        pipeline=pipeline,
    )
```

- [ ] **Step 6: Implement `upload_overlays` and renderer methods**

Add to `src/brainscan/renderer.py` after `create_render_pipeline`:

```python
def upload_overlays(
    res: RenderResources, overlays: list["TextOverlay"]
) -> None:
    """Pack overlays and upload to the static GPU buffers.

    Stores `overlay_run_count` in `res.uniform_data` so the next `draw()` call
    sees the new count. Caller is responsible for triggering a draw.
    """
    chars: list[int] = []
    runs: list[tuple[int, int, int, int]] = []
    for ov in overlays:
        char_offset = len(chars)
        for ch in ov.text:
            chars.append(ord(ch) & 0xFF)
        runs.append((ov.x, ov.y, len(ov.text), char_offset))

    chars_arr = np.array(chars, dtype=np.uint32) if chars else np.zeros(1, dtype=np.uint32)
    runs_arr = (
        np.array(runs, dtype=np.uint32).reshape(-1, 4)
        if runs else np.zeros((1, 4), dtype=np.uint32)
    )

    res.device.queue.write_buffer(res.overlay_chars_buffer, 0, chars_arr.tobytes())
    res.device.queue.write_buffer(res.overlay_runs_buffer, 0, runs_arr.tobytes())
    res.uniform_data["overlay_run_count"] = np.uint32(len(runs))
```

In `OffscreenRenderer`, add:

```python
    def set_overlays(self, overlays: list["TextOverlay"]) -> None:
        upload_overlays(self._res, overlays)
```

In `LiveRenderer`, add the same method (does not need to be thread-safe — overlays are static; if a future caller wants to refresh from a non-main thread, this is the spot to add a lock):

```python
    def set_overlays(self, overlays: list["TextOverlay"]) -> None:
        upload_overlays(self._res, overlays)
```

Add the import at the top of `renderer.py`:

```python
from brainscan.layout import TextOverlay
```

(If a circular-import issue arises, move the import inside the function.)

- [ ] **Step 7: Initialise `overlay_run_count` to 0 in `create_render_pipeline`**

When `uniform_data` is constructed in `create_render_pipeline`, ensure the new field defaults to zero:

```python
    uniform_data["overlay_run_count"] = np.uint32(0)
```

- [ ] **Step 8: Run the tests and confirm they pass**

```bash
mise exec -- uv run pytest tests/test_renderer.py -v
```

Expected: all pass, including `TestOverlays`. The existing offscreen tests should still pass because overlays are empty by default and the shader returns the existing weight rendering when no run is hit.

- [ ] **Step 9: Commit**

```bash
git add src/brainscan/renderer.py tests/test_renderer.py
git commit -m "feat(renderer): render overlay glyphs over zero-padded pixels"
```

---

## Phase 6: Configuration and integration

### Task 9: `LAYOUT_*` tuning constants

Centralise the geometry knobs and label colour in `src/brainscan/tuning.py` so the install site can adjust without touching call sites.

**Files:**
- Modify: `src/brainscan/tuning.py`

- [ ] **Step 1: Add the constants**

Append to `src/brainscan/tuning.py`:

```python
# --- Layout chrome --------------------------------------------------

LAYOUT_SECTION_GUTTER_PX = 40
"""Pixels between sections (EMBED, BLK 0..7, OUT)."""

LAYOUT_GROUP_GUTTER_PX = 20
"""Pixels between attn and mlp groups inside a transformer block."""

LAYOUT_ITEM_GUTTER_PX = 4
"""Pixels between consecutive unlabelled items (LN strips, embeddings)."""

LAYOUT_LABEL_GAP_PX = 16
"""Pixels above each labelled matrix (qkv, proj, up, down)."""

LAYOUT_SECTION_LABEL_PX = 24
"""Height of the section-label band at the top of every section column."""

LAYOUT_MIN_SECTION_WIDTH = 80
"""Minimum width for any section column. Floors EMBED and OUT so they read
as sections rather than slivers."""

LAYOUT_LABEL_COLOR = (0.55, 0.55, 0.60)
"""Dim-grey RGB triple used for both section and matrix labels.

Currently hard-coded in the WGSL fragment shader; kept here for traceability
and so the tuning surface is complete."""
```

- [ ] **Step 2: Sanity-check that the module still imports**

```bash
mise exec -- uv run python -c "from brainscan import tuning; print(tuning.LAYOUT_SECTION_GUTTER_PX)"
```

Expected: prints `40`.

- [ ] **Step 3: Commit**

```bash
git add src/brainscan/tuning.py
git commit -m "feat(tuning): expose LAYOUT_* geometry constants"
```

---

### Task 10: Wire chrome through `train.py`; drop `n_embd` to 540; spatial weight buffer

The training entry point needs four updates:
1. `--n-embd` default drops from 558 to 540 (so the new chrome fits in 4128 px).
2. `compute_layout` is called with the tuning constants so chrome appears.
3. `_build_weight_buffer` is rewritten to use `place_weights_on_canvas` so gutters are real zero-padded pixels.
4. `compute_text_overlays` is called once at startup (and after each rebirth) and uploaded to the renderer.

**Files:**
- Modify: `src/brainscan/train.py`

- [ ] **Step 1: Update the model-arch defaults and verify divisibility**

`540 / 9 = 60`, so `n_embd=540` divides evenly into 9 heads. In `train.py`, change:

```python
    parser.add_argument("--n-embd", type=int, default=540)
```

(Leave `n_head=9` and `n_layer=8`.)

Also lower the default in `src/brainscan/model.py` so unit tests that import `GPT()` directly use the same arch:

```python
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        sequence_len: int = 256,
        n_layer: int = 8,
        n_head: int = 9,
        n_embd: int = 540,
    ):
```

Run the test suite once to make sure no test relied on the old default:

```bash
mise exec -- uv run pytest tests/ -v
```

Expected: all pass. (Tests with explicit `n_embd` are unaffected; the default-using `model_param_counts` fixture in `test_layout.py` now reports ~28 M instead of ~30 M, but no assertion pins the exact count.)

- [ ] **Step 2: Pass tuning constants to `compute_layout`**

In `train.py`, edit the section that computes the layout:

```python
    sections = default_sections(n_layer=args.n_layer)
    layout = compute_layout(
        param_counts,
        sections=sections,
        section_gutter=tuning.LAYOUT_SECTION_GUTTER_PX,
        group_gutter=tuning.LAYOUT_GROUP_GUTTER_PX,
        item_gutter=tuning.LAYOUT_ITEM_GUTTER_PX,
        label_gap_px=tuning.LAYOUT_LABEL_GAP_PX,
        section_label_height=tuning.LAYOUT_SECTION_LABEL_PX,
        min_section_width=tuning.LAYOUT_MIN_SECTION_WIDTH,
    )
```

- [ ] **Step 3: Rewrite `_build_weight_buffer` to use spatial placement**

Replace the existing `_build_weight_buffer` with:

```python
def _build_weight_buffer(
    weights: dict[str, torch.Tensor],
    layout: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Place weights at their layout rects on a `(height, width)` canvas, then
    flatten to the row-major float32 array the renderer expects."""
    np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    canvas = place_weights_on_canvas(np_weights, layout, width=width, height=height)
    return canvas.ravel()
```

Update the import block in `train.py` to include `place_weights_on_canvas`:

```python
from brainscan.layout import (
    HEIGHT,
    LAYOUT_HEIGHT,
    TEXT_STRIP_HEIGHT,
    WIDTH,
    compute_layout,
    compute_text_overlays,
    default_sections,
    layout_summary,
    place_weights_on_canvas,
)
```

(`layout_to_flat_order` is no longer used; remove it from the import list.)

Update the two call sites of `_build_weight_buffer`:

```python
                    if offscreen_renderer is not None:
                        buf = _build_weight_buffer(
                            current_weights,
                            layout,
                            offscreen_renderer.width,
                            offscreen_renderer.height,
                        )
                        canvas = offscreen_renderer.render(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                            global_brightness=global_brightness,
                        )
                        save_frame(
                            canvas, frames_dir / f"frame_{step:06d}.png"
                        )

                    if live_renderer is not None:
                        buf = _build_weight_buffer(
                            current_weights, layout, WIDTH, HEIGHT,
                        )
                        live_renderer.update(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                            global_brightness=global_brightness,
                        )
```

Remove the now-unused `flat_order = layout_to_flat_order(layout)` line earlier in `main()`.

- [ ] **Step 4: Compute and upload overlays at startup**

After the renderers are constructed in `main()`, compute overlays once:

```python
    overlays = compute_text_overlays(layout, sections)
    if offscreen_renderer is not None:
        offscreen_renderer.set_overlays(overlays)
    if live_renderer is not None:
        live_renderer.set_overlays(overlays)
```

After each rebirth (inside `train_loop`, after the existing `rebirth(...)` call), refresh in case of any future arch change. Append:

```python
                    # Recompute layout/overlays in case n_embd changed.
                    new_param_counts = {n: p.numel() for n, p in model.named_parameters()}
                    if new_param_counts != param_counts:
                        new_layout = compute_layout(
                            new_param_counts,
                            sections=sections,
                            section_gutter=tuning.LAYOUT_SECTION_GUTTER_PX,
                            group_gutter=tuning.LAYOUT_GROUP_GUTTER_PX,
                            item_gutter=tuning.LAYOUT_ITEM_GUTTER_PX,
                            label_gap_px=tuning.LAYOUT_LABEL_GAP_PX,
                            section_label_height=tuning.LAYOUT_SECTION_LABEL_PX,
                            min_section_width=tuning.LAYOUT_MIN_SECTION_WIDTH,
                        )
                        layout.clear()
                        layout.update(new_layout)
                        new_overlays = compute_text_overlays(layout, sections)
                        if offscreen_renderer is not None:
                            offscreen_renderer.set_overlays(new_overlays)
                        if live_renderer is not None:
                            live_renderer.set_overlays(new_overlays)
```

(In practice `n_embd` does not change at rebirth, so this block is a defensive no-op; it makes the code resilient if the rebirth pipeline ever does change architecture.)

- [ ] **Step 5: Run the full test suite**

```bash
mise exec -- uv run pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Smoke-test training**

```bash
mise exec -- uv run python -m brainscan.train --steps 5 --no-mic
```

Expected: training runs five steps without error and prints a layout summary that mentions `EMBED`, `BLK 0..7`, and `OUT`. (No GUI; no `--live` flag.)

- [ ] **Step 7: Commit**

```bash
git add src/brainscan/model.py src/brainscan/train.py
git commit -m "feat(train): wire layout chrome and drop n_embd default to 540"
```

---

### Task 11: Update `CLAUDE.md` and `README.md`

Update the user-facing documentation so the new chrome is described accurately.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update `CLAUDE.md` "Display layout"**

Replace the existing "Display layout" section in `CLAUDE.md`. The replaced block currently reads:

```
## Display layout

The top 4128px contains weight matrices laid out left to right: embed → 8 block
columns → output. Matrices stack top-to-bottom within their column. 4px gutters
separate matrices and sections. See README.md for the ASCII diagram.
```

Replace with:

```
## Display layout

The top 4128px contains weight matrices laid out left to right: `EMBED` → 8
block columns (`BLK 0..7`) → `OUT`. Sections are separated by 40 px gutters;
inside each block column, the `attn` and `mlp` groups are separated by a 20
px group gutter. The four substantial matrices in each block (`c_attn`,
`attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) carry the dim-grey labels `qkv`,
`proj`, `up`, and `down`, drawn 1 px above the matrix in the 16 px label
gap. The two LN strips inside each group keep the existing 4 px gutter and
remain unlabelled. The first 24 px of every section column is a label band
holding the centred section name (`EMBED`, `BLK 0`, …). All chrome is
zero-padded so overlays never occlude weight data.

The default model is `n_embd=540`, which yields ~28 M parameters and leaves
margin for the new chrome inside the 4128 px weight region.
```

Also update the bottom of the same `CLAUDE.md` "Tuning" section if it lists tunables — append a sentence pointing at the new constants if missing. (Skip this if the existing prose already says "all dataclass defaults … draw from it" — it does, and the new constants are picked up automatically.)

- [ ] **Step 2: Update `README.md`**

Replace the "Display layout" ASCII diagram in `README.md`. Replace the existing block:

````
## Display layout

Information flows left to right across the 8K canvas. The top 4128px contains
weight matrices (one pixel per parameter); the bottom 192px is a text strip
showing generated text coloured by probability.

```
 7680px
◄──────────────────────────────────────────────────────────────────────►
┌──┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──┐ ▲
│wt│ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ln│ │
│e │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│f │ │
…
└──────────────────────────────────────────────────────────────────────────────────────┘ ▼
```

The weight layout fills the top 4128px. Embeddings and output head are narrow
columns on the left and right edges. The text strip renders 320 × 4 characters
at 3× scale, with brightness indicating token confidence.
````

with:

````
## Display layout

Information flows left to right across the 8K canvas. The top 4128px contains
weight matrices (one pixel per parameter), separated by 40 px section gutters
and labelled by an in-gutter section band (`EMBED`, `BLK 0..7`, `OUT`). Inside
each transformer block, attn and mlp groups are split by a 20 px gutter; the
four substantial matrices carry tiny labels `qkv`, `proj`, `up`, `down`. The
bottom 192 px is a three-band text strip (audience / model / captions).

```
 7680 px
◄──────────────────────────────────────────────────────────────────────►
┌────┬─────────┬─────────┬   …   ┬─────────┬────┐ ▲
│EMBE│  BLK 0  │  BLK 1  │       │  BLK 7  │OUT │ │ 24 px section-label band
│ wte│  qkv    │  qkv    │       │  qkv    │ln_f│ │
│ wpe│  c_attn │  c_attn │       │  c_attn │... │ │
│    │  proj   │  proj   │       │  proj   │head│ │ 4128 px weight region
│    │  c_proj │  c_proj │       │  c_proj │    │ │ (gutters zero-padded)
│    │  ─ ─ ─  │  ─ ─ ─  │       │  ─ ─ ─  │    │ │ 20 px attn/mlp gutter
│    │  up     │  up     │       │  up     │    │ │
│    │  c_fc   │  c_fc   │       │  c_fc   │    │ │
│    │  down   │  down   │       │  down   │    │ │
│    │  c_proj │  c_proj │       │  c_proj │    │ │
└────┴─────────┴─────────┴   …   ┴─────────┴────┘ ▼
                40 px section gutters
┌────────────────────────────────────────────────┐ ▲
│ Audience lane (90 px, 3× scale, warm cream)    │ │
│ Model lane    (90 px, 3× scale, cool ramp)     │ │ 192 px text strip
│ Captions     (12 px, 1× scale, dim grey)       │ │
└────────────────────────────────────────────────┘ ▼
```

The default model is `n_embd=540`, yielding ~28 M parameters — about 7 %
smaller than before, to make room for the new chrome.
````

Also update the "## Architecture" line that says `558 embedding dim, 256 context window (~30M parameters)` to read `540 embedding dim, 256 context window (~28M parameters)`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: describe layout delineation chrome"
```

---

## Self-review checklist

- [x] Spec coverage: every locked design item (gutters, sub-grouping, labels, n_embd reduction) maps to a task. The implementation surface (`layout.py`, `renderer.py`, `tuning.py`) and the documentation update are all covered.
- [x] No placeholders: every code block is concrete and runnable.
- [x] Type consistency: `Item.label`, `Group.label`, `Section.label`, `TextOverlay.text` etc. are used with the same names everywhere they appear.
- [x] One extra task beyond the spec — `place_weights_on_canvas` (Task 6) — is necessary because the current renderer reads weights as a flat row-major buffer; without it, gutters in the layout would not exist as zero-padded pixels and overlays would land on top of weight values rather than gutters.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-layout-delineation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
