# LLM Brainscan

Visualise transformer weight changes during training and activations during
inference, displayed on an 8K screen with one pixel per parameter.

## Concept

A character-level GPT with ~32M trainable parameters --- sized so that every
parameter gets exactly one pixel on an 8K display (7680×4320 = 33,177,600
pixels). Watch the model learn in real time: see attention heads form, MLP
features sharpen, and embedding clusters emerge.

The byte-level vocabulary (256 tokens) keeps embedding overhead under 1%, so
nearly all pixels show the transformer weights where the interesting learning
happens.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). If you have
[mise](https://mise.jdx.dev/), everything is configured:

```sh
mise trust
mise exec -- uv sync
mise exec -- uv run pytest
```

Without mise:

```sh
uv sync
uv run pytest
```

## Training

```sh
# Save weight visualisation frames as PNGs
mise exec -- uv run python -m brainscan.train --steps 1000 --save-images

# Live fullscreen display (renders directly to the screen, no readback)
mise exec -- uv run python -m brainscan.train --live

# Both at once
mise exec -- uv run python -m brainscan.train --live --save-images
```

This downloads Tiny Shakespeare on first run and trains the model. `--live`
opens a fullscreen window showing weight matrices in real time (one pixel per
parameter). `--save-images` writes frames to `output/frames/`.

## Architecture

Default model: 8 layers, 9 attention heads, 558 embedding dim, 256 context
window (~30M parameters).

```
src/brainscan/
├── model.py      # character-level GPT
├── data.py       # byte-level encode/decode and batching
├── snapshot.py   # weight capture, deltas, activation hooks
├── layout.py     # 8K canvas layout engine
├── font.py       # bitmap font atlas for GPU text rendering
├── renderer.py   # wgpu offscreen + live fullscreen renderer
└── train.py      # training with snapshot and live display integration
```

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
│  │         │         │         │         │         │         │         │         │  │ │
│  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │  │ │
│  │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │  │4128px
│  │         │         │         │         │         │         │         │         │  │ │
│  │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │  │ │
├──│─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│wp│ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │lm│ │
│e │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │         │         │         │         │         │         │         │         │hd│ │
│  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │  │ │
│  │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │  │ │
├──┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──┤ ▼
│ Generated text output (bitmap font, 3× scale, coloured by probability)       192px │
│ ROMEO: O, she doth teach the torches to burn bright! It seems she hangs upon the    │4320px
│ cheek of night Like a rich jewel in an Ethiope's ear...                              │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘ ▼
```

The weight layout fills the top 4128px. Embeddings and output head are narrow
columns on the left and right edges. The text strip renders 320 × 4 characters
at 3× scale, with brightness indicating token confidence.

## Hardware

- **dev**: any machine with a CUDA GPU and a Vulkan-capable display
- **target**: NVIDIA Jetson Orin 64GB connected to an 8K display (1:1 pixel
  mapping, disable torch.compile for ARM64)

The `--live` mode uses rendercanvas with GLFW for fullscreen presentation. On
the Jetson's shared memory architecture, the fragment shader normalises weights
on the GPU (no CPU-side normalisation pass), minimising memory copies.

## Licence

MIT --- see [LICENCE](LICENCE). Copyright (c) 2026 Ben Swift.
