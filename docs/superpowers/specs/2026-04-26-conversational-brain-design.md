# The Conversational Brain --- design spec

Date: 2026-04-26
Status: Approved (brainstorm complete; ready for implementation plan)
Successor to: `backlog/tasks/task-1 - Design-speech-I-O-UX-for-text-strip-and-training-loop.md`

## Concept

LLM Brainscan currently shows a 30M-parameter character-level GPT's weight
matrices on an 8K display, with a thin text strip generating Shakespeare and a
microphone that feeds both inference and training. At the *stop-and-look*
viewer timescale (30s --- 3 min) the weights barely change, so the piece risks
reading as static.

This spec turns the piece into a **conversation**. The microphone becomes the
central rhythm; the model speaks back, sometimes literally; and a daily
rebirth gives the work a clear arc from morning gibberish to evening Shakespeare.
The weight visualisation stays sacred (1:1 pixels for parameters), but the
192px text strip is rebuilt as the social surface of the piece.

## Locked design constraints

These were resolved during the brainstorm and govern every decision below.

1. **Engagement profile**: stop-and-look (30s --- 3min). Must hook in a glance
   and reward 1--3 minutes of attention.
2. **Sacred 1:1 invariant**: the upper 4128px is weights only. Anything else
   lives in the existing 192px strip; we do not steal pixels from the brain.
3. **Whispered captions**: peripheral labels only. No legends, no dashboards.
4. **Mic is bidirectional**: speech triggers inference *and* appends to the
   training corpus. Both behaviours stay.
5. **Architecture stays GPT-2**: no RoPE, RMSNorm, SwiGLU, GQA, or MoE in this
   spec. Pedagogical anchor wins.
6. **Daily rebirth**: the model resets at a configured wall-clock time each day.
   Visitors at the same time of day see roughly the same stage of learning.

## Visual layout

The 192px text strip is split into three horizontal bands.

```
y = 4128 ─────────────────────────────────────────────────────────────
         ▸ AUDIENCE LANE                  90px   3× scale, 320 cols × 1 row
         (warm cream on charcoal, ▸ source-tag prefix)
y = 4218 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         ▸ MODEL LANE                     90px   3× scale, 320 cols × 1 row
         (cool ramp, probability-coloured glyphs, caret cursor)
y = 4308 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         ▸ CAPTIONS FOOTER                12px   1× scale, ~960 chars
y = 4320 ═════════════════════════════════════════════════════════════
```

### Lane behaviour

Each lane shows **one scrolling row** of 320 chars at 3× scale (24×48 px
glyphs). The 90px lane height accommodates one 48px-tall row of glyphs
vertically centred with 21px margins above and below; the margins absorb
ascender/descender headroom and let the source-tag prefix (audience lane)
and caret cursor (model lane) be drawn without clipping.

The right edge is "now"; older characters drift left and fall off the
display. This means a glance gives the most recent content; a longer gaze
lets the viewer read history before it scrolls away.

### Audience lane (top, 90px)

- Glyph colour: warm cream `(0.94, 0.88, 0.72)` on charcoal `(0.04, 0.04, 0.06)`.
- Each utterance is prefixed by a source tag, e.g. `▸ mic ▸ `, rendered in a
  dimmer hue than the speech itself.
- During LISTENING, partial transcriptions appear in greyed-out form
  `(0.50, 0.46, 0.38)` at the right edge, replaced character-by-character as
  Whisper's segment estimates stabilise.
- On commit, the source tag pulses (one-frame brightness boost) and the text
  promotes to full-brightness cream.

### Model lane (bottom, 90px)

- Glyph colour: existing probability-driven brightness, but with a cool ramp:
  `(brightness, brightness * 0.95, brightness * 1.10)` --- a gentle blue-cream
  bias instead of the current warm bias. The two lanes must read as
  unmistakably different from across the room.
- A caret `▌` is rendered at the rightmost active position during streaming
  generation, removed when generation pauses.
- The probability-to-brightness mapping stays as-is: `0.25 + prob * 0.75`.

### Captions footer (12px, 1× scale)

A monospace, single-row band at the very bottom. ~960 visible characters
total at 1× scale (8×16 px glyphs).

- **Left ~40 chars**: state tag, e.g. `listening...`, `thinking...`,
  `responding...`, `musing`, `training step 2,430 loss 1.82`, `dawn 06:00`.
- **Right ~40 chars**: section label for the cursor location reading the
  weights view, e.g. `block 4 attn c_attn` or `embed wte`. The cursor is a
  conceptual one --- since we are not pointing at a specific pixel, this is
  the section currently visible at the centre of the screen, or the section
  most recently activated by inference (if a brief activation pulse is later
  added).
- Middle stays blank by default. Reserved for occasional event lines (e.g.
  `audience_input rotated to 2026-04-25.txt`) that fade in for ~5s and out.
- Captions render in dim grey `(0.40, 0.40, 0.42)` so they are legible but
  do not compete with the lanes.

### Scrolling

Implemented as a circular character buffer per lane. The fragment shader
samples from the buffer with an offset so visual scroll is sub-character
smooth at the cost of ~one extra uniform per lane (current_offset_px). At
scroll speeds tied to generation rate, a viewer reads new tokens as they
appear without the lane jittering.

## Conversational state machine

The training loop drives a four-state machine. Transitions are *visible* ---
each transition has a small visual or audible event the viewer can perceive.

```
       ┌──────── MUSE ────────┐
       │                      │
       ▼                      ▲
   LISTENING ──────────► RESPONDING
       │                      │
       └──────────────────────┘
```

### State: MUSE (default, ~70% of wall time)

- The model autoregressively generates from a rolling prompt seeded by the
  most recent committed lane content (its own last response or muse, plus
  any audience trail).
- **Generation rate is intentionally slow**: one token per ~150ms. A viewer
  can read the model thinking. This is *not* the natural speed of inference
  --- it is paced for human attention.
- Training continues in parallel as today: snapshot/redraw every N steps
  pushes new weights to the live renderer.
- Captions footer reads `musing` on the left.

### State: LISTENING

- Triggered by RMS over `silence_threshold` (existing `is_speech` logic).
- Captions footer flips to `listening...`.
- Muse generation rate slows to one token per ~600ms (the model "pays
  attention").
- The audience lane begins receiving partial Whisper transcriptions in the
  greyed-out colour, updating each chunk.
- A sub-pixel pulse on the audience lane right edge marks each successful
  partial.
- No waveform visualisation --- too busy at 3m viewing distance.

### State: RESPONDING

- Triggered by Whisper committing an utterance (the existing
  `_do_transcribe` path).
- The audience lane prefix `▸ mic ▸ ` brightens to commit colour.
- The current muse generation in the model lane is **interrupted**, not
  finished. The model's response, seeded with the committed audience text,
  begins streaming in its place.
- Generation rate **temporarily speeds up** to one token per ~50ms, so a
  ~60-token reply lands in ~3s. Humans expect conversational latency; this
  state explicitly violates the slow-muse pacing to feel responsive.
- Captions footer reads `thinking...` for the first ~200ms of generation
  (model loading the prompt context), then `responding...`.
- If TTS is enabled (`--speak`), the response is also spoken aloud (see §3).

### Transition: RESPONDING → MUSE

- After the response completes, a short cooldown period (3s) elapses during
  which LISTENING cannot trigger. This prevents Whisper from re-triggering
  on the model's own TTS playback or on echoey acoustics.
- The MUSE state resumes seeded by the response text, so the model
  "continues its thought" from where the audience took it.

### Mic-as-corpus

Independent of the visual state machine, every committed audience utterance
is appended to the training `TextBuffer` for ongoing training, exactly as
today. So conversation literally feeds the model's future selves --- a
visitor speaking at 11am subtly shapes how the model speaks at 4pm.

## Audio (TTS)

Enabled via `--speak` flag (default off; on for installation use). Off by
default to keep test runs silent.

### Engine

`piper` (offline, runs on CPU, sub-second latency, good Australian English
voice available --- `en_AU-fitch-medium` is the leading candidate). Adds
~30MB model file; install via `uv add piper-tts`.

### Behaviour

- TTS plays only during RESPONDING. MUSE is silent. This makes the audio
  feel *earned*, not chatter.
- Voice should be quiet (peak ~-12 dBFS), slightly reverberant (small-room
  IR), and panned slightly off-centre so it feels like a presence in the
  installation space rather than a chatbot.
- Playback latency adds to the LISTENING-suppression window: cooldown is
  `3s + tts_duration` after RESPONDING completes.

### Optional sub-bass drone

A separate `--drone` flag (default off, off by default for installation
too --- this is opt-in atmospheric extra) adds a slow sine drone whose
pitch tracks training loss: lower pitch as the model gets smarter. Almost
subliminal (peak -18 dBFS, 40-60 Hz range). Implemented as a single
oscillator updated each snapshot.

## Daily rebirth and persistence

A `--rebirth-at HH:MM` flag (default off) schedules a daily reset at the
specified wall-clock time.

### At rebirth time

1. Fade canvas to charcoal over ~2s (renderer stops pushing new frames;
   existing frame brightness ramps down).
2. Re-seed RNG with `seed = today_iso + base_seed_from_args`. Log the seed
   so the day is reproducible.
3. Re-initialise model weights via existing `_init_weights`.
4. Discard optimiser state (rebuild fresh AdamW).
5. **Rotate** `output/audience_input.txt` to
   `output/audience/YYYY-MM-DD.txt` (where YYYY-MM-DD is *yesterday*).
   Create a fresh empty `audience_input.txt`.
6. Reset the training `TextBuffer` to its seed (Shakespeare). If
   `--persist-audience-days N` is set with N > 0 (default 7), prepend the
   last N days of rotated audience logs to the seed. Pass `0` to disable.
7. Fade back in over ~2s --- "dawn".
8. Captions footer flashes `dawn 06:00` (or whatever the configured time)
   for ~10s, then resumes normal display.

### Across-day continuity

`--persist-audience-days N` (single flag, default 7, set to 0 to disable):
on rebirth, the prior N days of audience input get prepended to the
training corpus. This means returning visitors meaningfully shape the
long-term character of the piece across weeks, even though each day's
model is "young".

### Reproducibility

The daily seed is logged to `output/rebirth.log` as
`YYYY-MM-DD seed=<int> persist_days=<int>`. This makes a specific day
reproducible (re-run with `--seed <int>` and the same persist files).

## Implementation outline

### New files

- **`src/brainscan/conversation.py`** (~150 lines)
  - `ConversationState` enum: MUSE, LISTENING, RESPONDING.
  - `Conversation` dataclass + driver: holds the state, transition
    predicates, generation-rate per state, the rolling lane buffers, and
    cooldown timers.
  - `step(now, listener_state, model)` --- pure-ish function called from
    the training loop, returns updated state and any
    side-effect-requesting events (e.g. "play TTS for this text").

- **`src/brainscan/tts.py`** (~80 lines)
  - `TTSEngine` thin wrapper around piper.
  - `speak(text)` returns a duration estimate; plays asynchronously.
  - No-op when disabled.

- **`src/brainscan/captions.py`** (~50 lines)
  - `compose_caption(state, training_state, layout_cursor)` --- pure
    function returning the left/right/middle text fragments.
  - Layout into the 12px footer's character grid.

- **`src/brainscan/rebirth.py`** (~80 lines)
  - `RebirthScheduler` --- watches wall clock, fires when scheduled.
  - `rebirth(model, optimiser, training_data, rng_seed)` --- pure-ish
    function performing the reset; returns new objects rather than
    mutating in place where feasible.
  - `rotate_audience_log(path, target_dir, date)` --- pure-ish file op.

- **`src/brainscan/audio_drone.py`** (~50 lines)
  - Optional sub-bass oscillator. Disabled by default. Implemented via
    `sounddevice` output stream that we're already using for input.

### Modified files

- **`src/brainscan/renderer.py`**
  - Shader (~100 lines of WGSL): two text lanes with separate scroll
    offsets, separate colour functions, plus the captions footer (1×
    scale, dim grey).
  - New uniforms: `audience_y`, `audience_height`, `model_y`,
    `model_height`, `captions_y`, `captions_height`, `audience_offset_px`,
    `model_offset_px`, `audience_count`, `model_count`, `captions_count`.
  - New storage buffers: `audience_chars`, `audience_attrs` (for greyed
    partials and source tags), `model_chars`, `model_probs` (existing,
    repurposed), `captions_chars`.
  - `RenderConfig` gains lane heights and caption height (each defaulting
    to the values above).

- **`src/brainscan/train.py`**
  - Replace the inline mic poll loop with a `Conversation` driver tick.
  - Wire `--speak`, `--drone`, `--rebirth-at`, `--persist-audience-days`
    args.
  - Snapshot/redraw cadence stays as today.

- **`src/brainscan/stt.py`**
  - Add `partial_callback: Callable[[str], None] | None` to
    `SpeechListener`. When set, called every chunk with the running
    partial transcript while in_speech is True. Reuses existing audio
    buffer; calls Whisper on partial buffer for live transcription.
  - Performance: partial transcribes happen every chunk (~2s) by default;
    too frequent causes Whisper to be the bottleneck. Configurable via
    `partial_interval_seconds`.

- **`src/brainscan/data.py`**
  - `TextBuffer.rotate(target_path)` --- atomically move current persist
    file to target, leave buffer in memory unchanged so callers can decide
    when/whether to also reset the buffer.

### Tests

New: `test_conversation.py`, `test_rebirth.py`, `test_captions.py`,
`test_tts.py` (the last with the engine mocked for CI).

Modified: `test_renderer.py` (dual-lane rendering, scroll offset,
captions), `test_stt.py` (partial callback path), `test_data.py`
(`rotate` semantics), `test_train.py` (state machine integration with a
fake clock and mock listener).

### Effort estimate

Roughly five focused sessions:

1. Dual-lane renderer + captions footer (shader, uniforms, tests).
2. Conversation state machine + lane buffers + integration in train.py.
3. STT partial callback + LISTENING visualisation.
4. TTS integration (`piper`, `--speak`, cooldown handling).
5. Daily rebirth + log rotation + persist-audience.

Drone is a half-session if/when wanted; not on the critical path.

## Out of scope

The following ideas surfaced during the brainstorm and are *deliberately*
deferred:

- Render-channel cycling on the weight pixels (deltas, gradients, Adam
  momentum). This was Approach A's centre of gravity; it would compound
  well with the conversational design but is its own project.
- Inference-time activation pulses on the weight matrices. Would require
  forward hooks and a different uniform path.
- Live attention-pattern overlays.
- Architectural modernisation (RoPE, RMSNorm, SwiGLU, GQA, MoE).
- Explicit "scenes" tied to time of day (Approach C). Light captions
  changes ("dawn", time tags) capture some of this without the full
  scene-system complexity.

These can be specced separately later. The conversational design is
self-contained.

## Open questions for the planner

1. Should `partial_callback` use Whisper's own partial decoding, or simply
   re-run the small model on the growing buffer each chunk? The latter is
   simpler and probably fast enough at the `small` model size. Plan should
   benchmark both.
2. Is there an existing local TTS in the project's macOS dev environment
   (`say` command) we should fall back to during development to avoid the
   piper dependency on the dev machine? Worth checking before locking
   `piper` as the only option.
3. Generation-rate pacing (150ms / 600ms / 50ms) is asserted by feel. The
   implementation plan should include an explicit step that tunes these
   against a real exhibition viewer at 3m distance before locking in
   defaults.

## Acceptance for "v1 complete"

- Dual-lane text strip renders correctly on the 8K target with both lanes
  visibly distinct from 3m away.
- A viewer can speak; their words appear in the audience lane within 3s
  of speech end; the model's response begins within 1s of commit.
- TTS speaks model responses audibly but quietly; does not re-trigger
  itself.
- A configured daily rebirth fires correctly, rotates the audience log,
  and re-seeds the model.
- Whispered captions footer is legible up close, ignorable at 3m.
- Existing tests pass; new tests for conversation, captions, rebirth, and
  dual-lane rendering pass.
