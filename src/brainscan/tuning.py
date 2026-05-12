"""Centralised tuning parameters for the conversational brain.

These defaults are placeholder values; tune at the install site against a real
exhibition viewer at 3m distance before the piece goes live. The non-tunable
technical constants (sample rates, buffer sizes, model architecture) live with
their respective modules.
"""

from __future__ import annotations

# --- Conversation pacing (seconds per token) -------------------------

MUSE_TOKEN_INTERVAL = 0.15
"""Default muse rate. Slower = more readable; too fast = illegible at 3m."""

LISTENING_TOKEN_INTERVAL = 0.6
"""Slowed muse during LISTENING. Should be perceptibly slower than MUSE."""

RESPONDING_TOKEN_INTERVAL = 0.05
"""Fast generation during RESPONDING. ~60 tokens at 0.05s = 3s reply."""

RESPONSE_TOKEN_COUNT = 60
"""Tokens per response before transition back to MUSE."""

COOLDOWN_SECONDS = 3.0
"""LISTENING-suppression window after a response ends. Prevents re-trigger
on the model's own TTS playback or echoey acoustics."""

# --- Speech-to-text (Whisper) ---------------------------------------

SILENCE_THRESHOLD = 0.01
"""RMS amplitude threshold for is_speech. Raise for noisy environments."""

MIN_SPEECH_SECONDS = 0.5
"""Minimum speech duration before a Whisper transcribe is run on commit."""

MAX_SPEECH_SECONDS = 30.0
"""Hard cap on accumulated speech before a forced transcribe."""

PARTIAL_INTERVAL_SECONDS = 1.0
"""Minimum interval between partial-transcript callbacks during in_speech."""

STT_CHUNK_SECONDS = 0.2
"""Audio capture granularity. Sets the wall-clock latency for speech-onset
and end-of-speech detection: a value of 0.2 means VAD evaluates 5x/second.
Smaller is more responsive but produces more wake-ups; 0.2 is comfortable
on the Jetson and lets a 0.5s utterance be detected after ~3 chunks."""

# --- TTS (piper) ----------------------------------------------------

TTS_GAIN_DB = -12.0
"""Quiet headroom for TTS playback. Spec target: peak ~-12 dBFS."""

# --- Drone ----------------------------------------------------------

DRONE_MIN_HZ = 40.0
"""Bottom of drone pitch range (low loss = high pitch in the drone)."""

DRONE_MAX_HZ = 60.0
"""Top of drone pitch range. Sub-audible peak pitch."""

DRONE_GAIN_DB = -18.0
"""Subliminal drone level. Spec: peak -18 dBFS."""

# --- Visual pulses --------------------------------------------------

PULSE_HALF_LIFE_SECONDS = 0.5
"""Decay constant for commit and partial pulses (visual shimmer)."""

# --- Captions footer ------------------------------------------------

EVENT_LINE_DURATION_SECONDS = 5.0
"""How long a captions event line stays on screen before clearing."""

CURSOR_LABEL_PERIOD_SECONDS = 3.0
"""How often the captions cursor label rotates to the next parameter name."""

# --- Rebirth --------------------------------------------------------

REBIRTH_FADE_DURATION_SECONDS = 2.0
"""Seconds for the fade-out and fade-in halves of the rebirth transition."""

PERSIST_AUDIENCE_CYCLES_DEFAULT = 2
"""Default for --persist-audience-cycles CLI flag (each cycle ≈ one week of audience)."""

SEED_CORPUS_SAMPLING_WEIGHT = 0.9
"""Fraction of each training batch sampled from the seed corpus vs audience.

Rest is sampled from accumulated audience speech, so visitor input continues
to shape training even when the seed corpus is orders of magnitude larger.
"""

# --- Layout chrome --------------------------------------------------

LAYOUT_SECTION_GUTTER_PX = 40
"""Pixels between sections (EMBED, BLK 0..7, OUT)."""

LAYOUT_GROUP_GUTTER_PX = 20
"""Pixels between attn and mlp groups inside a transformer block."""

LAYOUT_ITEM_GUTTER_PX = 4
"""Pixels between consecutive unlabelled items (LN strips, embeddings)."""

LAYOUT_LABEL_GAP_PX = 16
"""Pixels above each labelled matrix (qkv, proj, up, down)."""

LAYOUT_SECTION_LABEL_PX = 24
"""Height of the section-label band at the top of every section column."""

LAYOUT_MIN_SECTION_WIDTH = 80
"""Minimum width for any section column. Floors EMBED and OUT so they read
as sections rather than slivers."""

LAYOUT_LABEL_COLOR = (0.55, 0.55, 0.60)
"""Dim-grey RGB triple used for both section and matrix labels.

Currently hard-coded in the WGSL fragment shader; kept here for traceability
and so the tuning surface is complete."""
