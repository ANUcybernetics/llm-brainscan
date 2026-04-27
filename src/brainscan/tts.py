"""Offline TTS via piper. No-op when disabled or when piper is unavailable."""

from __future__ import annotations

import logging

import numpy as np

from brainscan import tuning

log = logging.getLogger(__name__)

try:  # optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - sounddevice is in main deps but mocked in tests
    sd = None  # type: ignore[assignment]


def _load_voice(voice: str) -> object:
    from piper import PiperVoice  # type: ignore[import-not-found]

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
        self._voice: object | None = None

        if enabled:
            try:
                self._voice = _load_voice(voice)
            except Exception as e:  # pragma: no cover
                log.warning("TTS disabled: failed to load voice %s: %s", voice, e)
                self.enabled = False

    def speak(self, text: str) -> float:
        if not self.enabled or self._voice is None or not text.strip():
            return 0.0

        sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))  # type: ignore[union-attr]
        chunks: list[bytes] = []
        for piece in self._voice.synthesize(text):  # type: ignore[union-attr]
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
