# LLM Brainscan

## Project overview

Visualise a character-level GPT's weight matrices during training on an 8K
display (one pixel per parameter). The model has ~30M params; the bottom 192px
of the display is a text strip showing inference output.

## Stack

- Python 3.12, managed with uv via mise
- PyTorch for model training and inference
- wgpu (WebGPU via wgpu-native/Vulkan) for GPU-side rendering and colourmaps
- rendercanvas (GLFW backend) for live fullscreen window display
- All commands: `mise exec -- uv run ...`
- Tests: `mise exec -- uv run pytest tests/ -v`
- Type checking: `mise exec -- uv run ty check`

## Code layout

```
src/brainscan/
├── model.py      # GPT model (vanilla transformer, char-level, vocab=256) + generate()
├── data.py       # decode(), TextBuffer, prepare_batches
├── stt.py        # SpeechConfig, is_speech(), transcribe(), SpeechListener
├── snapshot.py   # capture_weights() --- detached clone of all model params
├── layout.py     # maps param tensors to 8K canvas (left-to-right sections)
├── font.py       # bitmap font atlas (8x16 glyphs) for GPU text rendering
├── renderer.py   # RenderConfig, RenderResources, create_render_pipeline(), draw()
└── train.py      # training loop, prepare_display_buffers(), render_frame()
tests/
├── conftest.py        # shared fixtures (SMALL_CONFIG, device, small_model)
├── test_model.py      # architecture, forward pass, generate()
├── test_data.py       # decode, prepare_batches
├── test_snapshot.py   # capture_weights
├── test_layout.py     # sections, compute_layout, overlaps, ordering
├── test_font.py       # font atlas shape and glyph coverage
├── test_renderer.py   # offscreen, live, text strip, display scaling
├── test_text_buffer.py # TextBuffer append, persistence
├── test_stt.py        # speech detection, transcription, audio loop
└── test_train.py      # training loop, display buffers, render_frame, e2e
```

## Key constraints

- total trainable params must fit in layout area (7680 × 4128 = 31,703,040 px)
- character-level (vocab_size=256) to minimise embedding overhead
- must run on NVIDIA Jetson Orin 64GB (no torch.compile on ARM64)
- dev machine has an RTX 6000 Ada; Jetson is the deployment target

## Display layout

The top 4128px contains weight matrices laid out left to right: embed → 8 block
columns → output. Matrices stack top-to-bottom within their column. 4px gutters
separate matrices and sections. See README.md for the ASCII diagram.

The bottom 192px is a text strip showing generated text at 3× scale (24×48 pixel
glyphs), coloured by softmax probability --- bright for confident tokens, dim
for uncertain ones. 320 columns × 4 rows = 1,280 characters visible.

The layout is defined by `Section` objects in `layout.py`. The `compute_layout`
function assigns pixel coordinates; `layout_to_flat_order` gives the flattening
order for the renderer's storage buffer.

## Rendering pipeline

1. Capture weights as a dict of tensors (`snapshot.py:capture_weights`)
2. Prepare display buffers (`train.py:prepare_display_buffers`) --- flatten in
   layout order, zero-pad to canvas size, convert text chars/probs to arrays
3. Upload raw weights, font atlas, and text data to wgpu storage buffers
4. Fragment shader normalises by vmax uniform, applies colourmap per pixel
   (top) and renders bitmap font text coloured by probability (bottom strip)
5. Read back as RGBA numpy array (offscreen) or present to display (live)

GPU resources are managed via dataclasses and pure functions:
- `RenderConfig` (frozen dataclass): width, height, colormap, text settings
- `RenderResources` (dataclass): GPU buffers, bind group, pipeline
- `create_render_pipeline(config, device, format)` → `RenderResources`
- `draw(resources, target_view, weights, ...)` renders a frame

Two renderer classes provide the high-level API:
- `OffscreenRenderer`: headless via Vulkan, returns numpy array (no display needed)
- `LiveRenderer`: fullscreen window via rendercanvas/GLFW, training runs in
  background thread, `update()` pushes data thread-safely, `--live` flag in
  `train.py`

## Speech-to-text

Audio processing uses dataclasses and pure functions:
- `SpeechConfig` (frozen dataclass): model size, thresholds, sample rate
- `is_speech(audio, threshold)` --- pure function for speech detection
- `transcribe(model, audio)` --- pure function for Whisper transcription
- `SpeechListener` --- thin thread/queue wrapper around the pure functions

## Key APIs

- `GPT.generate(prompt_bytes, max_tokens, device=None)` --- autoregressive
  sampling, returns `(tokens, probs)`. Preserves training/eval mode.
- `prepare_display_buffers(weights, flat_order, canvas_pixels, ...)` --- shared
  helper for both offscreen and live rendering paths
- `render_frame(renderer, weights, flat_order, ...)` --- convenience wrapper
  that calls `prepare_display_buffers` with the renderer's own dimensions

## Conventions

- use `mise exec -- uv run` prefix for all commands
- pytest for testing; aim for comprehensive coverage
- no type: ignore or # noqa unless genuinely necessary
