# LLM Brainscan

## Project overview

Visualise a character-level GPT's weight matrices during training on an 8K
display (one pixel per parameter). The model has ~30M params; the bottom 192px
of the display is a three-band text strip carrying a live conversation between
the audience (via mic) and the model (via streaming generation), plus a
whispered captions footer. The piece resets at a configured wall-clock time
each day so visitors at the same time of day see the same stage of learning.

## Stack

- Python 3.12, managed with uv via mise
- PyTorch for model training and inference
- wgpu (WebGPU via wgpu-native/Vulkan) for GPU-side rendering and colourmaps
- rendercanvas (GLFW backend) for live fullscreen window display
- faster-whisper for speech-to-text
- piper-tts (optional, `uv sync --extra tts`) for offline text-to-speech
- All commands: `mise exec -- uv run ...`
- Tests: `mise exec -- uv run pytest tests/ -v`
- Type checking: `mise exec -- uv run ty check`

## Code layout

```
src/brainscan/
├── model.py        # GPT model + generate() + streaming_generate()
├── data.py         # decode(), TextBuffer (with rotate()), prepare_batches
├── stt.py          # SpeechConfig, is_speech(), transcribe(), SpeechListener
│                   #   (with partial_callback + speech_end_callback)
├── tts.py          # TTSEngine wrapper around piper (optional, no-op when off)
├── lanes.py        # LaneBuffer circular char buffer (partial/committed)
├── captions.py     # CaptionsState, compose_caption() for the 12px footer
├── conversation.py # MUSE/LISTENING/RESPONDING state machine, Conversation.step()
├── rebirth.py      # rotate_audience_log(), rebirth(), RebirthScheduler
├── audio_drone.py  # Optional sub-bass DroneOscillator tracking loss
├── snapshot.py     # capture_weights() --- detached clone of all model params
├── layout.py       # maps param tensors to 8K canvas (left-to-right sections)
├── font.py         # bitmap font atlas (8x16 glyphs) for GPU text rendering
├── renderer.py     # RenderConfig, RenderResources, LaneFrame, CaptionsFrame,
│                   #   create_render_pipeline(), draw()
└── train.py        # training loop driven by Conversation.step()
tests/
├── conftest.py            # shared fixtures (SMALL_CONFIG, device, small_model)
├── test_model.py          # architecture, forward pass, generate, streaming_generate
├── test_data.py           # decode, prepare_batches
├── test_snapshot.py       # capture_weights
├── test_layout.py         # sections, compute_layout, overlaps, ordering
├── test_font.py           # font atlas shape and glyph coverage
├── test_renderer.py       # three-band rendering, lane scroll, display scaling
├── test_text_buffer.py    # TextBuffer append, persistence, rotate
├── test_stt.py            # speech detection, transcription, partial/end callbacks
├── test_tts.py            # TTSEngine enabled/disabled paths (mocked piper)
├── test_lanes.py          # LaneBuffer push, replace_tail, commit_partial
├── test_captions.py       # compose_caption layout
├── test_conversation.py   # state machine transitions, timing, cooldown
├── test_rebirth.py        # rotate_audience_log, rebirth(), RebirthScheduler
├── test_audio_drone.py    # DroneOscillator loss-to-pitch mapping
└── test_train.py          # train loop wiring, smoke test, daily-seed reproducibility
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

The bottom 192px is split into three horizontal bands:

```
y = 4128 ───────────────────────────────────────
         AUDIENCE LANE   90px   3× scale, 320 cols × 1 row, warm cream
y = 4218 ───────────────────────────────────────
         MODEL LANE      90px   3× scale, 320 cols × 1 row, cool ramp
y = 4308 ───────────────────────────────────────
         CAPTIONS FOOTER 12px   1× scale, 960 chars, dim grey
y = 4320 ───────────────────────────────────────
```

Each lane is a single row of 320 chars at 3× scale (24×48 px glyphs); the right
edge is "now" and older chars drift left. Sub-pixel scrolling is driven by per-
lane `*_offset_px` uniforms. The captions footer is unscrolled and rendered
1× scale.

The weight layout is defined by `Section` objects in `layout.py`. The
`compute_layout` function assigns pixel coordinates; `layout_to_flat_order`
gives the flattening order for the renderer's storage buffer.

## Rendering pipeline

1. Capture weights as a dict of tensors (`snapshot.py:capture_weights`).
2. Build a flat weight buffer (`train.py:_build_weight_buffer`) --- flattens
   in layout order, zero-pads to canvas size.
3. Snapshot the audience and model `LaneBuffer`s into `LaneFrame`s; compose
   the captions row into a `CaptionsFrame` (`train.py:_build_lane_frames`).
4. Upload raw weights, font atlas, and the three lane/captions buffers to
   wgpu storage buffers.
5. Fragment shader normalises weights by `vmax`, applies colourmap per pixel
   (weight region) and renders three text bands with their own colour rules
   (audience: warm cream, dimmed for `ATTR_PARTIAL`, slightly dimmer for
   `ATTR_SOURCE_TAG`; model: cool blue-cream brightness `0.25 + prob * 0.75`;
   captions: dim grey on charcoal).
6. Read back as RGBA numpy array (offscreen) or present to display (live).

GPU resources are managed via dataclasses and pure functions:
- `RenderConfig` (frozen dataclass): width, height, colormap, lane heights.
  Defaults audience/model/captions heights to 0 (lanes disabled); the train
  CLI passes the production `90/90/12`.
- `LaneFrame` / `CaptionsFrame`: per-lane payload (chars, attrs/probs,
  count, offset_px).
- `RenderResources` (dataclass): GPU buffers, bind group, pipeline.
- `create_render_pipeline(config, device, format)` → `RenderResources`.
- `draw(resources, target_view, weights, audience=, model=, captions=)`.

Two renderer classes provide the high-level API:
- `OffscreenRenderer`: headless via Vulkan, returns numpy array (no display).
- `LiveRenderer`: fullscreen window via rendercanvas/GLFW, training runs in
  background thread, `update()` pushes data thread-safely, `--live` flag.

## Conversation state machine

`conversation.py:Conversation` is a pure dataclass driven by the train loop.
Three states:

- **MUSE** (default ~70%): the model autoregressively continues from a rolling
  prompt at one token per ~150ms. Training continues in parallel.
- **LISTENING**: triggered by mic RMS > threshold. Partial Whisper transcripts
  appear in the audience lane in greyed colour (`ATTR_PARTIAL`); muse pacing
  slows to one token per ~600ms.
- **RESPONDING**: triggered when Whisper commits. The committed text is
  promoted to full brightness with a `▸ mic ▸ ` source-tag prefix; the muse
  generator is interrupted and replaced by a generator seeded with the
  committed text, running at one token per ~50ms (~3s for 60 tokens). When
  TTS is enabled, the response text is also spoken aloud. After completion a
  cooldown (3s + tts_duration) prevents re-triggering on the model's own
  audio.

`Conversation.step(now, listener_snapshot, token_fn) -> StepEvents` is pure:
it advances state, asks the caller's `token_fn` for at most one token per
call, and returns events the caller acts on (`speak_events`,
`new_corpus_text`). All I/O lives outside `Conversation`.

Every committed audience utterance is also appended to the training
`TextBuffer`, so conversation literally feeds the model's future selves.

## Daily rebirth

`--rebirth-at HH:MM` schedules a daily reset. At rebirth time:

1. The audience log file is rotated to `output/audience/YYYY-MM-DD.txt`.
2. The model is re-initialised via `model.apply(model._init_weights)` after
   `torch.manual_seed(daily_seed)` where `daily_seed = sha256(date+seed)[:4]`
   (reproducible).
3. The training corpus is rebuilt as `last_N_days_audience + seed_corpus`
   where N comes from `--persist-audience-days` (default 7, 0 disables).
4. The optimiser is rebuilt fresh.

Returning visitors meaningfully shape the long-term character of the piece
across weeks even though each day's model is "young". The daily seed is
logged to `output/rebirth.log`.

## Speech-to-text

Audio processing uses dataclasses and pure functions:
- `SpeechConfig` (frozen dataclass): model size, thresholds, sample rate.
- `is_speech(audio, threshold)` --- pure function for speech detection.
- `transcribe(model, audio)` --- pure function for Whisper transcription.
- `SpeechListener` --- thin thread/queue wrapper. Optional callbacks:
  `partial_callback(text)` fires while `in_speech` at most once per
  `partial_interval_seconds`; `speech_end_callback()` fires once on the
  silence-after-speech transition (so the conversation can clear partial
  state).

## Text-to-speech (optional)

`tts.py:TTSEngine` wraps piper. Disabled by default; enable with `--speak`.
When piper isn't installed (the dep is optional), the engine is a silent
no-op so dev machines and CI work without the model file. `speak(text)`
plays via sounddevice and returns the duration estimate; the train loop
extends the conversation cooldown by that duration so the listener doesn't
re-trigger on the model's own playback.

## Key APIs

- `GPT.generate(prompt_bytes, max_tokens, device=None)` --- autoregressive
  sampling, returns `(tokens, probs)`.
- `GPT.streaming_generate(prompt_bytes, device=None, emit_prompt=False)` ---
  generator yielding `(token, prob)` pairs one at a time, used by the
  conversation driver to pace generation against wall-clock time.
- `train.py:_build_weight_buffer(weights, flat_order, canvas_pixels)` ---
  flatten weights into a zero-padded float32 array.
- `train.py:_build_lane_frames(convo, captions_state)` --- snapshot the
  conversation's lane buffers and caption state into renderer payloads.
- `Conversation.step(now, listener_snapshot, token_fn)` --- advance the
  state machine; returns `StepEvents` describing requested side effects.
- `rebirth(model, seed_corpus, audience_dir, persist_days, seed)` --- reset
  weights and rebuild corpus; returns `RebirthResult(corpus, seed)`.

## CLI flags

- `--live` — fullscreen window via rendercanvas/GLFW.
- `--save-images` — write PNGs to `output/frames/`.
- `--no-mic` — disable speech listener.
- `--speak` — enable TTS for model responses (requires piper extra).
- `--drone` — enable sub-bass drone tracking loss (atmospheric extra).
- `--rebirth-at HH:MM` — daily rebirth time (24h, local).
- `--persist-audience-days N` — N past audience-log days prepended to corpus
  on rebirth (default 7, 0 disables).
- `--seed N` — base seed for daily rebirth (per-day seed = sha256(date+N)).

Plus the existing model/training knobs (`--n-layer`, `--n-head`, `--n-embd`,
`--sequence-len`, `--batch-size`, `--lr`, `--steps`, `--snapshot-every`,
`--gen-tokens`, `--silence-threshold`, `--chunk-seconds`,
`--min-speech-seconds`, `--max-speech-seconds`, `--whisper-model`,
`--whisper-device`, `--data`, `--output-dir`).

## Conventions

- use `mise exec -- uv run` prefix for all commands
- pytest for testing; aim for comprehensive coverage
- no type: ignore or # noqa unless genuinely necessary
