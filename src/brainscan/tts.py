"""Offline TTS via piper. No-op when disabled or when piper is unavailable."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from brainscan import tuning

log = logging.getLogger(__name__)

# Typed as Any so the None fallback below stays assignable; in production
# sounddevice is a main dep, but tests stub it out and CI machines may not
# have an audio driver available at import time.
sd: Any = None
try:
    import sounddevice as _sounddevice
    sd = _sounddevice
except Exception:  # pragma: no cover
    pass


def _load_voice(voice: str) -> Any:
    from piper import PiperVoice  # ty: ignore[unresolved-import]

    return PiperVoice.load(voice)


class TTSEngine:
    def __init__(
        self,
        enabled: bool = False,
        voice: str = "en_AU-fitch-medium",
        gain_db: float = tuning.TTS_GAIN_DB,
    ):
        self.enabled = enabled
        self.voice_name = voice
        self.gain_db = gain_db
        self._voice: Any = None

        if enabled:
            try:
                self._voice = _load_voice(voice)
            except Exception as e:  # pragma: no cover
                log.warning("TTS disabled: failed to load voice %s: %s", voice, e)
                self.enabled = False

    def speak(self, text: str) -> float:
        if not self.enabled or self._voice is None or not text.strip():
            return 0.0

        sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))
        chunks: list[bytes] = []
        for piece in self._voice.synthesize(text):
            chunks.append(piece.audio_int16_bytes)

        raw = b"".join(chunks)
        if not raw:
            return 0.0

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        gain = 10 ** (self.gain_db / 20.0)
        audio *= gain
        if sd is not None:
            sd.play(audio, samplerate=sample_rate)
        duration = len(audio) / sample_rate
        return float(duration)
