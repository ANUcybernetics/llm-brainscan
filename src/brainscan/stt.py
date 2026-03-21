"""Speech-to-text input via faster-whisper and sounddevice."""

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0
SILENCE_THRESHOLD = 0.01
MIN_SPEECH_SECONDS = 0.5
MAX_SPEECH_SECONDS = 30.0


@dataclass(frozen=True)
class SpeechConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = SAMPLE_RATE
    chunk_seconds: float = CHUNK_SECONDS
    silence_threshold: float = SILENCE_THRESHOLD
    min_speech_seconds: float = MIN_SPEECH_SECONDS
    max_speech_seconds: float = MAX_SPEECH_SECONDS
    audio_device: int | None = None

    @property
    def min_samples(self) -> int:
        return int(self.min_speech_seconds * self.sample_rate)

    @property
    def max_samples(self) -> int:
        return int(self.max_speech_seconds * self.sample_rate)

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_seconds * self.sample_rate)


def is_speech(audio: np.ndarray, threshold: float) -> bool:
    return float(np.sqrt(np.mean(audio**2))) > threshold


def transcribe(model: object, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(  # type: ignore[union-attr]
        audio,
        language="en",
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments)


def load_whisper_model(config: SpeechConfig) -> object:
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
    )


class SpeechListener:
    def __init__(self, config: SpeechConfig | None = None, **kwargs):
        if config is not None:
            self.config = config
        else:
            self.config = SpeechConfig(**kwargs)

        self._text_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: object | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_text(self) -> list[str]:
        segments = []
        while True:
            try:
                segments.append(self._text_queue.get_nowait())
            except queue.Empty:
                break
        return segments

    def _load_model(self) -> None:
        self._model = load_whisper_model(self.config)

    def _run(self) -> None:
        import sounddevice as sd

        self._load_model()
        cfg = self.config
        log.info(
            "STT listener started (model=%s, device=%s)",
            cfg.model_size,
            cfg.device,
        )

        speech_buffer: list[np.ndarray] = []
        in_speech = False

        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=cfg.chunk_samples,
            device=cfg.audio_device,
        ) as stream:
            while not self._stop_event.is_set():
                audio, overflowed = stream.read(cfg.chunk_samples)
                if overflowed:
                    log.warning("Audio input overflowed")
                chunk = audio[:, 0]

                if is_speech(chunk, cfg.silence_threshold):
                    speech_buffer.append(chunk)
                    in_speech = True
                    total_samples = sum(len(c) for c in speech_buffer)
                    if total_samples >= cfg.max_samples:
                        self._do_transcribe(speech_buffer)
                        speech_buffer = []
                        in_speech = False
                elif in_speech:
                    total_samples = sum(len(c) for c in speech_buffer)
                    if total_samples >= cfg.min_samples:
                        self._do_transcribe(speech_buffer)
                    speech_buffer = []
                    in_speech = False

    def _do_transcribe(self, chunks: list[np.ndarray]) -> None:
        assert self._model is not None
        audio = np.concatenate(chunks)
        text = transcribe(self._model, audio)
        if text:
            log.info("Transcribed: %s", text)
            self._text_queue.put(text)
