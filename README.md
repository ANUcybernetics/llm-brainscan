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

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). If you have
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

### On Jetson Orin

JetPack 6.x ships Python 3.10 and the only CUDA-enabled torch wheels for
aarch64 + CUDA 12.6 are cp310 (via
[jetson-ai-lab](https://pypi.jetson-ai-lab.io)). The wheel source is already
declared in `pyproject.toml` with a `python_version == '3.10'` marker, so
`uv sync` picks the right wheel automatically — just pin Python 3.10
locally and install the audio backend:

```sh
cat > mise.local.toml <<EOF
[tools]
python = "3.10"
EOF
mise install
mise exec -- uv sync
sudo apt install libportaudio2  # required by sounddevice for the STT thread
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

Default model: 8 layers, 9 attention heads, 540 embedding dim, 256 context
window (~28M parameters).

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

The default model is `n_embd=540`, yielding ~28 M parameters --- about 7 %
smaller than before, to make room for the new chrome.

## Hardware

- **dev**: any machine with a CUDA GPU and a Vulkan-capable display
- **target**: NVIDIA Jetson Orin 64GB connected to an 8K display (1:1 pixel
  mapping, disable torch.compile for ARM64)

The `--live` mode uses rendercanvas with GLFW for fullscreen presentation. On
the Jetson's shared memory architecture, the fragment shader normalises weights
on the GPU (no CPU-side normalisation pass), minimising memory copies.

## Licence

MIT --- see [LICENCE](LICENCE). Copyright (c) 2026 Ben Swift.
