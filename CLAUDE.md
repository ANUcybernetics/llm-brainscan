# LLM Brainscan

## Project overview

Visualise a character-level GPT's weight matrices during training on an 8K
display (one pixel per parameter). The model has ~32M params to fit within the
33.2M pixel budget.

## Stack

- Python 3.12, managed with uv via mise
- PyTorch for model training and inference
- wgpu (WebGPU via wgpu-native/Vulkan) for GPU-side rendering and colourmaps
- All commands: `mise exec -- uv run ...`
- Tests: `mise exec -- uv run pytest tests/ -v`

## Code layout

```
src/brainscan/
├── model.py      # GPT model (vanilla transformer, char-level, vocab=256)
├── data.py       # byte-level data loading and batching
├── snapshot.py   # weight/activation capture for visualisation
├── layout.py     # maps param tensors to 8K canvas (left-to-right sections)
├── renderer.py   # wgpu offscreen/windowed renderer with WGSL shaders
└── train.py      # training script with snapshot and rendering integration
tests/
├── conftest.py   # shared fixtures (SMALL_CONFIG for fast tests)
├── test_model.py
├── test_data.py
├── test_snapshot.py
├── test_layout.py
└── test_renderer.py
```

## Key constraints

- total trainable params must be ≤ 33,177,600 (8K pixel count)
- character-level (vocab_size=256) to minimise embedding overhead
- must run on NVIDIA Jetson Orin 64GB (no torch.compile on ARM64)
- dev machine has an RTX 6000 Ada; Jetson is the deployment target

## Display layout

Information flows left to right: embed → 8 block columns → output. Each block
column is 929px wide and fills the full 4320px height. Matrices stack
top-to-bottom within their column (ln → attn → proj → ln → mlp_fc → mlp_proj).
4px gutters separate matrices and sections. See README.md for the ASCII diagram.

The layout is defined by `Section` objects in `layout.py`. The `compute_layout`
function assigns pixel coordinates; `layout_to_flat_order` gives the flattening
order for the renderer's storage buffer.

## Rendering pipeline

1. Capture weights as a dict of tensors (`snapshot.py`)
2. Flatten in layout order and normalise to [-1, 1] (`renderer.py`)
3. Upload to a wgpu storage buffer
4. Fragment shader applies colourmap (diverging or thermal) per pixel
5. Read back as RGBA numpy array or display on canvas

The renderer works headless (offscreen) via Vulkan --- no window or display
server required.

## Conventions

- use `mise exec -- uv run` prefix for all commands
- pytest for testing; aim for comprehensive coverage
- no type: ignore or # noqa unless genuinely necessary
