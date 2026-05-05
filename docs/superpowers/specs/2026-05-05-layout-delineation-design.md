# Layout delineation --- design spec

Date: 2026-05-05
Status: Approved (brainstorm complete; ready for implementation plan)
Builds on: `docs/superpowers/specs/2026-04-26-conversational-brain-design.md`

## Concept

The current weight visualisation packs ~30M parameters into the 7680×4128 weight
canvas at one pixel per parameter, with 4px gutters between matrices and
between sections. From any viewing distance the result reads as a single
undifferentiated speckle: the eight-block transformer rhythm and the
attention/MLP substructure are present in the layout but invisible in the
image.

This spec adds *architectural* visual delineation --- wide gutters,
sub-grouping inside each block, and tiny in-gutter labels --- so a viewer
can perceive the network's repeating eight-block structure from across the
room and read the role of each weight matrix up close. The 1:1 pixel-per-
parameter invariant is preserved; the latent dimension shrinks slightly to
absorb the new chrome.

## Locked design constraints

Carried forward from the conversational-brain spec; restated where this
design touches them.

1. **1:1 pixels for parameters**: the upper 4128px is weights only.
   Gutters, label gaps, and the section-label band are part of that 4128px,
   not stolen from the bottom 192px text strip.
2. **Atmospheric reading first**: the audience must perceive structural
   rhythm without a legend. Labels exist as a reward for close inspection,
   not as a key.
3. **Architecture stays GPT-2**: no change to attention or MLP shapes. The
   subgrouping below mirrors the existing per-block parameter set.
4. **Daily rebirth**: the layout is recomputed once at startup and at each
   rebirth. Section labels are regenerated at the same time.

## Visual structure

The 7680×4128 weight canvas now has explicit chrome:

```
x = 0 ────────────────────────────────────────────────────────────────────  x = 7680
       ┌────────┬─────────────┬─────────────┬───   …   ───┬─────────────┬────────┐
y=  0  │ EMBED  │   BLK 0     │   BLK 1     │     …       │   BLK 7     │  OUT   │   24px label band
y= 24  │ ┌────┐ │ ┌─────────┐ │ ┌─────────┐ │             │ ┌─────────┐ │ ┌────┐ │   (font 8×16, 1× scale, dim grey)
       │ │wte │ │ │ qkv     │ │ │ qkv     │ │             │ │ qkv     │ │ │ln_f│ │
       │ │    │ │ │  c_attn │ │ │  c_attn │ │             │ │  c_attn │ │ │... │ │
       │ │    │ │ │         │ │ │         │ │             │ │         │ │ │    │ │
       │ ├────┤ │ ├─────────┤ │ ├─────────┤ │             │ ├─────────┤ │ │head│ │
       │ │wpe │ │ │ proj    │ │ │ proj    │ │             │ │ proj    │ │ │    │ │
       │ │    │ │ │  c_proj │ │ │  c_proj │ │             │ │  c_proj │ │ │    │ │
       │ │    │ │ │         │ │ │         │ │             │ │         │ │ │    │ │
       │ │    │ │ │ ─────── │ │ │ ─────── │ │             │ │ ─────── │ │ │    │ │   20px group gap
       │ │    │ │ │ up      │ │ │ up      │ │             │ │ up      │ │ │    │ │
       │ │    │ │ │  c_fc   │ │ │  c_fc   │ │             │ │  c_fc   │ │ │    │ │
       │ │    │ │ │         │ │ │         │ │             │ │         │ │ │    │ │
       │ │    │ │ ├─────────┤ │ ├─────────┤ │             │ ├─────────┤ │ │    │ │
       │ │    │ │ │ down    │ │ │ down    │ │             │ │ down    │ │ │    │ │
       │ │    │ │ │  c_proj │ │ │  c_proj │ │             │ │  c_proj │ │ │    │ │
       │ └────┘ │ └─────────┘ │ └─────────┘ │             │ └─────────┘ │ └────┘ │
y=4128 └────────┴──┬──────────┴──┬──────────┴────  …  ────┴──┬──────────┴────────┘
                   │             │                           │
              40px section gutters between every column
```

### Geometry

| Knob                    | Value | Notes                                              |
| ----------------------- | ----- | -------------------------------------------------- |
| Section gutter          | 40 px | Between EMBED, each block, and OUT                 |
| Group gutter            | 20 px | Between attn and mlp groups inside a block         |
| Item gutter (unlabelled)|  4 px | Around LN matrices and as default                  |
| Label gap (labelled)    | 16 px | Above each labelled matrix (qkv, proj, up, down)   |
| Section label band      | 24 px | Top of every column; font 8×16 at 1× scale, centred |
| Min section width       | 80 px | Floor for EMBED and OUT so they read as sections   |

Total new vertical chrome per block: 24 (label band) + 4×16 (label gaps) +
20 (group gutter) + a handful of 4px LN gutters = ~140 px. Available matrix
height: 4128 − 140 ≈ 3988 px.

### Sub-grouping within blocks

Each transformer block splits into two groups separated by the 20 px group
gutter:

- `attn` group: `ln_1.weight`, `ln_1.bias`, `attn.c_attn.weight`,
  `attn.c_proj.weight`
- `mlp` group: `ln_2.weight`, `ln_2.bias`, `mlp.c_fc.weight`,
  `mlp.c_proj.weight`

The four substantial matrices (c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)
each get a 16 px label gap above them. The two LN strips inside each group
are 1--2 px tall and remain unlabelled with their existing 4 px gutters.

### Labels

Two kinds, both rendered with the existing 8×16 bitmap font at 1× scale,
dim grey (R 0.55, G 0.55, B 0.60).

1. **Section labels** (24 px band at the top of every column, centred):
   - `EMBED`, `BLK 0`, `BLK 1`, … `BLK 7`, `OUT`
2. **Matrix labels** (left-aligned, 1 px above the top of the matrix, in
   the 16 px label gap):
   - `qkv` above `c_attn`
   - `proj` above `attn.c_proj`
   - `up` above `c_fc`
   - `down` above `mlp.c_proj`

No group labels (`attn`, `mlp`) --- the 20 px gap and the matrix labels
together convey the structure.

### n_embd reduction

The current default (~576) does not fit the new chrome.

Horizontal budget: the canvas is 7680 px wide. Nine 40 px section gutters
take 360 px; the 80 px floors on EMBED and OUT take 160 px. The eight
block columns must fit in 7680 − 360 − 160 = 7160 px, capping each block
column at **≤ 895 px**.

Vertical budget per block: 4128 px total, minus the 24 px section label
band, minus per-block chrome of roughly 120 px (four LN strips with 4 px
gutters, four 16 px label gaps, one 20 px group gutter), leaves
**~3984 px** for the four substantial matrices stacked together.

The four labelled matrices in a block contain `12·n_embd²` parameters.
With 895 × 3984 ≈ 3.57 M pixels available per block,
**n_embd² ≤ 297 k → n_embd ≤ 545**.

The new default is **n_embd = 540**, which yields ~28 M parameters ---
about 7 % smaller than the current model. The visual difference is
negligible; the legibility gain is significant.

## Implementation surface

### `src/brainscan/layout.py`

- New dataclass `Group(label: str, items: list[Item])` where each `Item`
  carries `(param_name, label: str | None)`.
- Replace `Section.param_names` with `Section.groups: list[Group]`.
  (Sections that contain a single conceptual group, like `EMBED` and `OUT`,
  hold one `Group` whose items have `label=None`.)
- Extend `compute_layout` to accept `section_gutter`, `group_gutter`,
  `item_gutter`, `label_gap_px`, `section_label_height`, `min_section_width`.
- Per-item vertical spacing depends on whether the item has a label:
  labelled items get `label_gap_px` above them, unlabelled items get
  `item_gutter`.
- The first item in each group does not get a leading gap; the inter-group
  gap is `group_gutter`. The label band consumes the first
  `section_label_height` pixels of every column.
- `default_sections(n_layer)` returns the new structure with `qkv`, `proj`,
  `up`, `down` labels on the four substantial matrices per block.
- New helper `compute_text_overlays(layout, section_metas)` returns a list
  of `(text, x, y)` tuples covering every section label and every matrix
  label, suitable for upload to the GPU.

### `src/brainscan/renderer.py`

- New static "overlay text" buffer alongside the existing audience / model
  / captions buffers. Shape: a flat array of glyph indices plus a parallel
  array of `(x, y, length)` triples for each overlay run.
- Shader gains an `overlay_band` function that, for any pixel in the upper
  4128px, checks whether the pixel falls inside an overlay glyph; if yes,
  returns the dim-grey label colour, else delegates to the existing weight
  rendering. Overlay glyphs are drawn over zero-padded pixels (the gutters
  and label band) so they never occlude weight data.
- Overlay buffers are populated once at pipeline creation and after each
  rebirth (since `n_embd` does not change at rebirth, the layout and
  overlays are stable; we still rebuild on rebirth for safety).

### `src/brainscan/tuning.py`

Expose the geometry constants so they can be tuned in one place:

```python
LAYOUT_SECTION_GUTTER_PX  = 40
LAYOUT_GROUP_GUTTER_PX    = 20
LAYOUT_ITEM_GUTTER_PX     = 4
LAYOUT_LABEL_GAP_PX       = 16
LAYOUT_SECTION_LABEL_PX   = 24
LAYOUT_MIN_SECTION_WIDTH  = 80
LAYOUT_LABEL_COLOR        = (0.55, 0.55, 0.60)
```

The default `n_embd` constant in `train.py` (and any matching defaults in
tests) drops to **540**.

### Tests

- `tests/test_layout.py`: cover the new gutter rules, the min-width floor,
  the label-gap accounting, and that `compute_text_overlays` emits the
  expected `(text, x, y)` triples for a small canonical configuration.
- `tests/test_renderer.py`: extend the existing offscreen test to assert
  that a known section label (`BLK 0`) and a known matrix label (`qkv`)
  appear at the expected pixel positions in dim grey, and that no overlay
  pixel collides with weight data.

### Documentation

- Update the ASCII layout diagram and the "Display layout" section of
  `CLAUDE.md` to describe the new chrome.
- Update `README.md` if it duplicates the layout description.

## Risks and open questions

- **Tiny-matrix labels are illegible**. The four labelled matrices (qkv,
  proj, up, down) are each ≥ 100 px tall at the chosen geometry; the
  labels are above them, not inside them, so this is fine. The two LN
  matrices remain unlabelled by design.
- **Overlay sampling cost**. The shader will run an overlay-glyph check on
  every weight-region pixel. This is cheap (a constant-time bitmap-font
  lookup against a small set of overlays per column) but worth confirming
  on the Jetson Orin once implemented.
- **n_embd choice**. 540 is a clean target that fits the new layout with
  margin. Any future change to gutters or labels should be paired with a
  recalculation. The geometry constants in `tuning.py` plus the layout
  algorithm's existing `_column_width` solver mean the system fails
  gracefully (column widths shrink to fit) rather than overflowing the
  canvas.
- **Out-of-scope**: this spec does not touch the bottom 192 px text strip,
  the conversation state machine, or the rebirth pipeline.
