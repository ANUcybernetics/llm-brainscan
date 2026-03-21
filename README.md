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
mise exec -- uv run python -m brainscan.train --steps 1000 --save-images
```

This downloads Tiny Shakespeare on first run, trains the model, and optionally
saves weight visualisation frames to `output/frames/`.

## Architecture

Default model: 8 layers, 9 attention heads, 576 embedding dim, 256 context
window.

```
src/brainscan/
├── model.py      # character-level GPT
├── data.py       # byte-level encode/decode and batching
├── snapshot.py   # weight capture, deltas, activation hooks
├── layout.py     # 8K canvas layout engine
├── renderer.py   # wgpu offscreen/windowed renderer
└── train.py      # training with snapshot integration
```

## Display layout

Information flows left to right across the 8K canvas. Each transformer block
is a column; matrices stack vertically within each column with small gutters
between them.

```
 7680px
◄──────────────────────────────────────────────────────────────────────►
┌──┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──┐ ▲
│wt│ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ ln1     │ln│ │
│e │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│f │ │
│  │         │         │         │         │         │         │         │         │  │ │
│  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │ c_attn  │  │ │
│  │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │ QKV     │  │ │
│  │         │         │         │         │         │         │         │         │  │ │
│  │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │ c_proj  │  │ │
├──│─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │4320px
│wp│ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │ ln2     │lm│ │
│e │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │         │         │         │         │         │         │         │         │hd│ │
│  │         │         │         │         │         │         │         │         │  │ │
│  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │ mlp.fc  │  │ │
│  │         │         │         │         │         │         │         │         │  │ │
│  │─────────│─────────│─────────│─────────│─────────│─────────│─────────│─────────│  │ │
│  │         │         │         │         │         │         │         │         │  │ │
│  │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │mlp.proj │  │ │
│  │         │         │         │         │         │         │         │         │  │ │
└──┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──┘ ▼
 ▲   block 0   block 1   block 2   block 3   block 4   block 5   block 6   block 7   ▲
embed                                                                               output
69px                          929px per block                                        35px
```

Each block column is 929px wide and fills the full 4320px height. The two MLP
matrices (c_fc and c_proj) dominate each column (~67% of block area), with the
attention matrices above them. Embeddings and the output head are narrow columns
on the left and right edges.

## Hardware

- **dev**: any machine with a CUDA GPU
- **target**: NVIDIA Jetson Orin 64GB (disable torch.compile for ARM64)

## Licence

Not yet decided.
