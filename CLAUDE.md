# LLM Brainscan

## Project overview

Visualise a character-level GPT's weight matrices during training on an 8K
display (one pixel per parameter). The model has ~32M params to fit within the
33.2M pixel budget.

## Stack

- Python 3.12, managed with uv via mise
- PyTorch for model training and inference
- All commands should be run via `mise exec -- uv run ...`
- Tests: `mise exec -- uv run pytest tests/ -v`

## Code layout

```
src/brainscan/
├── model.py      # GPT model (vanilla transformer, char-level)
├── data.py       # byte-level data loading and batching
├── snapshot.py   # weight/activation capture for visualisation
├── layout.py     # maps param tensors to 8K canvas regions
└── train.py      # training script with snapshot integration
tests/
├── conftest.py   # shared fixtures (SMALL_CONFIG for fast tests)
├── test_model.py
├── test_data.py
└── test_snapshot.py
```

## Key constraints

- total trainable params must be ≤ 33,177,600 (8K pixel count)
- character-level (vocab_size=256) to minimise embedding overhead
- must run on NVIDIA Jetson Orin 64GB (no torch.compile on ARM64)
- dev machine has a beefy CUDA GPU; Jetson is the deployment target

## Conventions

- use `mise exec -- uv run` prefix for all commands
- pytest for testing; aim for comprehensive coverage
- no type: ignore or # noqa unless genuinely necessary
