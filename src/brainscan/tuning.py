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

SILENCE_THRESHOLD = 0.02
"""RMS amplitude threshold for is_speech. Raise for noisy environments.

0.02 (~-34 dBFS) sits above the measured busy-room noise floor at the Jetson
install (crowd murmur ~0.006-0.008 RMS, peaking ~0.014); the old 0.01 let those
peaks through and faster-whisper hallucinated filler ("Thank you.", "Hmm") on the
non-speech. Recalibrate against a quiet room plus a 3 m spoken test before relying on it."""

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

AUDIENCE_DRIFT_PX_PER_S = 24.0
"""Conveyor speed of the audience lane in canvas px/second. Old transcript
drifts left at this rate even during silence (one 32 px char cell per ~1.3 s;
a full 7680 px lane rolls off in ~5.3 min), and each new utterance lands at
the right edge with a gap proportional to the silence before it. 0 disables
the drift, restoring the push-driven lane."""

# --- Weight colourmap -----------------------------------------------

WEIGHT_STRETCH_K = 4.0
"""asinh contrast-stretch strength applied to per-rect-normalised weights
before colourmapping. 0 disables it (linear mapping); larger values lift the
bulk of each matrix's distribution towards the bright end. With the
black-centred diverging map this is the overall brightness knob for the
weight field. The percentile normalisation (WEIGHT_VMAX_PERCENTILE) already
lifts the bulk relative to outliers, so this stays modest: at 4.0 a Gaussian
init renders as a dark field with bright accents; much higher values wash
the field out to uniform brightness."""

WEIGHT_VMAX_PERCENTILE = 99.5
"""Percentile of |weight| used as each rect's normalisation max. Using a
percentile instead of the true max stops a handful of outlier weights from
setting the scale for the whole matrix; values beyond it clamp to the hot
(near-white) end of the colourmap."""

SHIMMER_STRENGTH = 0.6
"""Peak additive brightness of the learning shimmer (the green-white flash
showing exceptional |delta-w| between weight snapshots --- above-median
motion only, so the typical gradient-noise floor stays dark). 0 disables
the effect."""

SHIMMER_HALF_LIFE_SECONDS = 1.5
"""Half-life of the shimmer decay between weight uploads. Should be shorter
than the interval between snapshots so each upload reads as a fresh pulse."""

WEIGHT_FLOOR_TINT = (0.030, 0.032, 0.052)
"""Very dark blue-grey floor for nonzero weights, so near-zero weights stay
visible on panels that crush blacks while exact-zero padding (gutters, label
bands) remains true black.

Currently hard-coded in the WGSL fragment shader; kept here for traceability
and so the tuning surface is complete."""

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
