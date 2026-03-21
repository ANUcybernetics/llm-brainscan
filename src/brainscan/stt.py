"""Speech-to-text input via faster-whisper and sounddevice."""

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0
SILENCE_THRESHOLD = 0.01
MIN_SPEECH_SECONDS = 0.5
MAX_SPEECH_SECONDS = 30.0


class SpeechListener:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = CHUNK_SECONDS,
        silence_threshold: float = SILENCE_THRESHOLD,
        min_speech_seconds: float = MIN_SPEECH_SECONDS,
        max_speech_seconds: float = MAX_SPEECH_SECONDS,
        audio_device: int | None = None,
        on_transcription: Callable[[str], None] | None = None,
    ):
        self._model_size = model_size
        self._whisper_device = device
        self._compute_type = compute_type
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._silence_threshold = silence_threshold
        self._min_samples = int(min_speech_seconds * sample_rate)
        self._max_samples = int(max_speech_seconds * sample_rate)
        self._audio_device = audio_device
        self._on_transcription = on_transcription

        self._text_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._model = None

    def _load_model(self):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size,
            device=self._whisper_device,
            compute_type=self._compute_type,
        )

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

    def _is_speech(self, audio: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(audio**2))) > self._silence_threshold

    def _run(self) -> None:
        import sounddevice as sd

        self._load_model()
        log.info(
            "STT listener started (model=%s, device=%s)",
            self._model_size,
            self._whisper_device,
        )

        chunk_samples = int(self._chunk_seconds * self._sample_rate)
        speech_buffer: list[np.ndarray] = []
        in_speech = False

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=self._audio_device,
        ) as stream:
            while not self._stop_event.is_set():
                audio, overflowed = stream.read(chunk_samples)
                if overflowed:
                    log.warning("Audio input overflowed")
                chunk = audio[:, 0]

                if self._is_speech(chunk):
                    speech_buffer.append(chunk)
                    in_speech = True
                    total_samples = sum(len(c) for c in speech_buffer)
                    if total_samples >= self._max_samples:
                        self._transcribe(speech_buffer)
                        speech_buffer = []
                        in_speech = False
                elif in_speech:
                    total_samples = sum(len(c) for c in speech_buffer)
                    if total_samples >= self._min_samples:
                        self._transcribe(speech_buffer)
                    speech_buffer = []
                    in_speech = False

    def _transcribe(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks)
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        if text:
            log.info("Transcribed: %s", text)
            self._text_queue.put(text)
            if self._on_transcription:
                self._on_transcription(text)
