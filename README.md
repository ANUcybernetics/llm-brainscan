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
└── train.py      # training with snapshot integration
```

## Hardware

- **dev**: any machine with a CUDA GPU
- **target**: NVIDIA Jetson Orin 64GB (disable torch.compile for ARM64)

## Licence

Not yet decided.
