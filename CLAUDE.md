# LLM Brainscan

## Project overview

Visualise a character-level GPT's weight matrices during training on an 8K
display (one pixel per parameter). The model has ~30M params; the bottom 224px
of the display is a two-lane text strip carrying a live conversation between
the audience (via mic) and the model (via streaming generation). The piece
resets at a configured wall-clock time each day so visitors at the same time
of day see the same stage of learning.

## Stack

- Python 3.12 (3.10 on Jetson Orin via `mise.local.toml`), managed with uv via mise
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
├── conversation.py # MUSE/LISTENING/RESPONDING state machine, Conversation.step()
├── rebirth.py      # rotate_audience_log(), rebirth(), RebirthScheduler
├── audio_drone.py  # Optional sub-bass DroneOscillator tracking loss
├── snapshot.py     # capture_weights() --- detached clone of all model params
├── layout.py       # maps param tensors to 8K canvas (left-to-right sections)
├── font.py         # chrome-label bitmap atlas + IBM Plex Mono lane atlas
├── renderer.py     # RenderConfig, RenderResources, LaneFrame,
│                   #   create_render_pipeline(), draw()
└── train.py        # training loop driven by Conversation.step()
tests/
├── conftest.py            # shared fixtures (SMALL_CONFIG, device, small_model)
├── test_model.py          # architecture, forward pass, generate, streaming_generate
├── test_data.py           # decode, prepare_batches
├── test_snapshot.py       # capture_weights
├── test_layout.py         # sections, compute_layout, overlaps, ordering
├── test_font.py           # bitmap + antialiased lane atlas: shape, coverage
├── test_renderer.py       # two-lane rendering, lane scroll, display scaling
├── test_text_buffer.py    # TextBuffer append, persistence, rotate
├── test_stt.py            # speech detection, transcription, partial/end callbacks
├── test_tts.py            # TTSEngine enabled/disabled paths (mocked piper)
├── test_lanes.py          # LaneBuffer push, replace_tail, commit_partial
├── test_conversation.py   # state machine transitions, timing, cooldown
├── test_rebirth.py        # rotate_audience_log, rebirth(), RebirthScheduler
├── test_audio_drone.py    # DroneOscillator loss-to-pitch mapping
└── test_train.py          # train loop wiring, smoke test, weekly-seed reproducibility
```

## Key constraints

- total trainable params must fit in layout area (7680 × 4128 = 31,703,040 px)
- character-level (vocab_size=256) to minimise embedding overhead
- must run on NVIDIA Jetson Orin 64GB (no torch.compile on ARM64)
- dev machine has an RTX 6000 Ada; Jetson is the deployment target

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
margin for the new chrome inside the 4096 px weight region.

The bottom 224px is split into two horizontal lanes:

```
y = 4096 ───────────────────────────────────────
         AUDIENCE LANE  112px   4× scale, 240 cols × 1 row, warm cream
y = 4208 ───────────────────────────────────────
         MODEL LANE     112px   4× scale, 240 cols × 1 row, cool ramp
y = 4320 ───────────────────────────────────────
```

Each lane is a single row of 240 chars at 4× scale (32×64 px glyphs); the right
edge is "now" and older chars drift left. Sub-pixel scrolling is driven by per-
lane `*_offset_px` uniforms.

The weight layout is defined by `Section` objects in `layout.py`. The
`compute_layout` function assigns pixel coordinates; `layout_to_flat_order`
gives the flattening order for the renderer's storage buffer.

## Rendering pipeline

1. Capture weights as a dict of tensors (`snapshot.py:capture_weights`).
2. Build a flat weight buffer (`train.py:_build_weight_buffer`) --- places
   each tensor at its layout rect, normalising per rect by the
   `tuning.WEIGHT_VMAX_PERCENTILE` percentile of `|w|` (outliers exceed ±1
   and clamp to the hot end), zero-pads chrome to canvas size. The train
   loop also builds a delta buffer (`train.py:_build_delta_buffer`):
   normalised `|Δw|` vs the previous snapshot, driving the learning shimmer.
3. Snapshot the audience and model `LaneBuffer`s into `LaneFrame`s
   (`train.py:_build_lane_frames`).
4. Upload normalised weights, deltas, both font atlases, and the two lane
   buffers to wgpu storage buffers.
5. Fragment shader normalises weights by `vmax` (1.0 in production since the
   buffer is pre-normalised), asinh-stretches, and applies the colourmap per
   pixel. The diverging map is black-centred: sign carries hue (negative
   blue, positive orange), magnitude carries luminance, extremes whiten.
   Nonzero weights get a dark floor tint so exact-zero chrome stays black
   and matrices read as panels. A green-white shimmer proportional to `|Δw|`
   flashes on each weight upload and decays
   (`tuning.SHIMMER_HALF_LIFE_SECONDS`). The two text lanes have their own
   colour rules (audience: warm cream, dimmed for `ATTR_PARTIAL`, slightly
   dimmer for `ATTR_SOURCE_TAG`; model: cool blue-cream brightness
   `0.25 + prob * 0.75`).
6. Read back as RGBA numpy array (offscreen) or present to display (live).

GPU resources are managed via dataclasses and pure functions:
- `RenderConfig` (frozen dataclass): width, height, colormap, lane heights.
  Defaults audience/model heights to 0 (lanes disabled); the train CLI
  passes the production `112/112`.
- `LaneFrame`: per-lane payload (chars, attrs/probs, count, offset_px).
- `RenderResources` (dataclass): GPU buffers, bind group, pipeline.
- `create_render_pipeline(config, device, format)` → `RenderResources`.
- `draw(resources, target_view, weights, audience=, model=, flat_deltas=,
  shimmer=)`.

Two renderer classes provide the high-level API:
- `OffscreenRenderer`: headless via Vulkan, returns numpy array (no display).
- `LiveRenderer`: fullscreen window via rendercanvas/GLFW, training runs in
  background thread, `update()` (full frame) and `update_lanes()` (text strip
  only, decoupled from weight capture) push data thread-safely, `--live` flag.

## Conversation state machine

`conversation.py:Conversation` is a pure dataclass driven by the train loop.
Three states:

- **MUSE** (default ~70%): the model autoregressively continues from a rolling
  prompt at one token per ~150ms. Training continues in parallel.
- **LISTENING**: triggered by mic RMS > threshold. Partial Whisper transcripts
  appear in the audience lane in greyed colour (`ATTR_PARTIAL`); muse pacing
  slows to one token per ~600ms.
- **RESPONDING**: triggered when Whisper commits. The committed text is
  promoted to full brightness with a `> mic > ` source-tag prefix; the muse
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

## Training corpus

The corpus has two parts:

- **Seed corpus** — large, immutable, memory-mapped via `np.memmap` at
  startup. Loaded from `--data` (path to a single file). Sized for the
  artwork (e.g., PG-19 at ~10 GB). On the Jetson the seed file lives on
  the external SSD: `/ssd/brainscan/data/pg19.bin`. The dev machine
  defaults to TinyShakespeare (auto-downloaded to `data/`).
- **Audience buffer** — small, mutable, in-RAM (`TextBuffer`). Accumulates
  spoken input live and persists to `output/audience_input.txt`.

`prepare_batches` samples each batch as a mixture: `seed_weight` of rows
(default `0.9` from `tuning.SEED_CORPUS_SAMPLING_WEIGHT`) come from the
seed; the rest from the audience buffer. When the audience is shorter
than `sequence_len + 1`, all rows fall back to the seed. This lets a
small, growing audience contribution shape training meaningfully even
when the seed is orders of magnitude larger.

To build the PG-19 seed file:
`mise exec -- uv run scripts/download_pg19.py /ssd/brainscan/data/pg19.bin`
(the script declares its `datasets` dep inline, no project-level changes).

## Weekly rebirth

`--rebirth-at "DOW HH:MM"` (e.g., `"MON 02:00"`) schedules a weekly reset.
At rebirth time:

1. The audience log file is rotated to `output/audience/YYYY-MM-DD.txt`,
   dated to the Monday of the week that just closed.
2. The model is re-initialised via `model.apply(model._init_weights)` after
   `torch.manual_seed(weekly_seed)` where
   `weekly_seed = sha256(iso_week_key+seed)[:4]` and `iso_week_key` is e.g.
   `"2026-W18"` (reproducible per ISO week).
3. The audience buffer is reset to `last_N_cycles_audience` where N comes
   from `--persist-audience-cycles` (default 2, 0 disables); each cycle
   file is roughly one week of audience speech. The seed corpus itself
   is loaded once at startup and persists across rebirths.
4. The optimiser is rebuilt fresh.

Each rebirth fires at most once per ISO week. If the system is down at the
target moment, it fires the next time the loop ticks within the same ISO
week. Returning visitors over a week witness a single brain growing; across
weeks the persisted audience layers shape the long-term character. The
weekly seed and cycle is logged to `output/rebirth.log`.

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
- `train.py:_build_lane_frames(convo)` --- snapshot the conversation's
  audience and model lane buffers into renderer payloads.
- `Conversation.step(now, listener_snapshot, token_fn)` --- advance the
  state machine; returns `StepEvents` describing requested side effects.
- `rebirth(model, audience_dir, persist_count, seed)` --- reset weights
  and load historical audience; returns `RebirthResult(audience, seed)`.
  The seed corpus is loaded once at training startup, not at rebirth.

## CLI flags

- `--live` — fullscreen window via rendercanvas/GLFW.
- `--save-images` — write PNGs to `output/frames/`.
- `--no-mic` — disable speech listener.
- `--speak` — enable TTS for model responses (requires piper extra).
- `--drone` — enable sub-bass drone tracking loss (atmospheric extra).
- `--rebirth-at "DOW HH:MM"` — weekly rebirth time (e.g., `"MON 02:00"`,
  24h local). Default off.
- `--persist-audience-cycles N` — N past audience-log cycles (~weeks)
  prepended to corpus on rebirth (default 2, 0 disables).
- `--seed N` — base seed for weekly rebirth
  (per-week seed = sha256(iso-week+N)).

Plus the existing model/training knobs (`--n-layer`, `--n-head`, `--n-embd`,
`--sequence-len`, `--batch-size`, `--lr`, `--steps`, `--snapshot-every`,
`--gen-tokens`, `--silence-threshold`, `--chunk-seconds`,
`--min-speech-seconds`, `--max-speech-seconds`, `--whisper-model`,
`--whisper-device`, `--data`, `--output-dir`).

## Tuning

`src/brainscan/tuning.py` is the single place to adjust exhibition-site knobs
--- pacing intervals, speech thresholds, TTS/drone gain, pulse decay, rebirth
fade duration, and so on. All dataclass defaults in `conversation.py`,
`stt.py`, `tts.py`, and `audio_drone.py` draw from it, as do the magic numbers
in the train loop. Non-tunable technical constants (sample rates, vocab size,
model architecture) stay with their respective modules.

## Deployment on Jetson

Live exhibition runs as a user-level systemd service. The unit file is
`deploy/brainscan.service` and is installed under
`~jane/.config/systemd/user/brainscan.service`. The service waits for
`graphical-session.target` so it starts only after the auto-login X
session is up.

System packages required on the Jetson (beyond a base JetPack install):

```sh
sudo apt install -y wmctrl     # EWMH fullscreen — without it the GNOME
                               # top panel stays visible over the canvas
```

Network setup for kiosk installs that rely on WiFi (e.g. ANU-Secure):
edit `deploy/install-wifi.sh` with the site's WPA2-Enterprise credentials,
uncomment the nmcli block, and run as root. The resulting NetworkManager
profile autoconnects on boot and survives reflashes-with-backup.

Install / refresh the brainscan service from a clean repo on the Jetson:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/brainscan.service ~/.config/systemd/user/brainscan.service
systemctl --user daemon-reload
systemctl --user enable --now brainscan.service
```

Day-to-day operations:

```sh
systemctl --user status brainscan          # is it running?
systemctl --user restart brainscan         # after a code pull
journalctl --user-unit brainscan -f        # tail logs (auto-flushed; PYTHONUNBUFFERED=1)
journalctl --user-unit brainscan --since "1 hour ago"
```

Note: use `--user-unit brainscan`, not `--user -u brainscan`. Per-user
journald isn't enabled on the Jetson, so user-unit logs land in the
system journal and need the system-journal flag form to filter.

The unit auto-restarts on failure (10 s backoff, max 5 restarts per
10 min) and re-launches whenever the X session restarts.

## Conventions

- use `mise exec -- uv run` prefix for all commands
- pytest for testing; aim for comprehensive coverage
- no type: ignore or # noqa unless genuinely necessary
