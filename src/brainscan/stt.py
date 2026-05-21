"""Speech-to-text input via faster-whisper and sounddevice.

Architecture: producer/consumer split.

* `_run_audio` reads fixed-size chunks from the input stream and pushes
  them onto an internal audio queue. It never runs Whisper, so it cannot
  fall behind the device clock.
* `_run_processor` pops chunks, runs VAD via `_process_chunk`, and
  accumulates the per-utterance speech buffer. Final commits are
  transcribed inline (cheap; happens on silence). Partial transcripts
  during a long utterance run on a short-lived worker thread, with a
  pending-flag that lets the processor skip new partials while one is
  still in flight - so the processor never blocks on Whisper either.

`_process_chunk` is the per-chunk state machine and is deliberately the
unit-test seam: tests can drive it deterministically without threads or
audio I/O. The state lives on the listener (`_speech_buffer`,
`_in_speech`, `_last_partial_t`).

Callbacks (`partial_callback`, `speech_end_callback`) are public
attributes; set them either via the constructor or by direct assignment
*before* calling `start()` to avoid losing early speech events.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from brainscan import tuning

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = tuning.STT_CHUNK_SECONDS
SILENCE_THRESHOLD = tuning.SILENCE_THRESHOLD
MIN_SPEECH_SECONDS = tuning.MIN_SPEECH_SECONDS
MAX_SPEECH_SECONDS = tuning.MAX_SPEECH_SECONDS


@dataclass(frozen=True)
class SpeechConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = SAMPLE_RATE
    chunk_seconds: float = tuning.STT_CHUNK_SECONDS
    silence_threshold: float = tuning.SILENCE_THRESHOLD
    min_speech_seconds: float = tuning.MIN_SPEECH_SECONDS
    max_speech_seconds: float = tuning.MAX_SPEECH_SECONDS
    audio_device: int | None = None
    partial_interval_seconds: float = tuning.PARTIAL_INTERVAL_SECONDS

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


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(
        audio,
        language="en",
        vad_filter=True,
    )
    # Collapse whitespace-only joins (Whisper emits empty segments for
    # silence) to an empty string so the caller's emptiness check rejects them.
    return " ".join(seg.text.strip() for seg in segments).strip()


def load_whisper_model(config: SpeechConfig) -> WhisperModel:
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type,
    )


class SpeechListener:
    def __init__(
        self,
        config: SpeechConfig | None = None,
        partial_callback: Callable[[str], None] | None = None,
        speech_end_callback: Callable[[], None] | None = None,
        **kwargs,
    ):
        self.config = config if config is not None else SpeechConfig(**kwargs)
        self.partial_callback = partial_callback
        self.speech_end_callback = speech_end_callback

        self._text_queue: queue.Queue[str] = queue.Queue()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._audio_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None
        self._model: WhisperModel | None = None

        self._speech_buffer: list[np.ndarray] = []
        self._in_speech: bool = False
        self._last_partial_t: float = 0.0
        self._partial_pending = threading.Event()

    def start(self) -> None:
        if self._audio_thread is not None:
            return
        self._stop_event.clear()
        self._audio_thread = threading.Thread(
            target=self._run_audio, daemon=True, name="stt-audio"
        )
        self._processor_thread = threading.Thread(
            target=self._run_processor, daemon=True, name="stt-processor"
        )
        self._audio_thread.start()
        self._processor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=5.0)
            self._audio_thread = None
        if self._processor_thread is not None:
            self._processor_thread.join(timeout=5.0)
            self._processor_thread = None
        # Let any in-flight partial worker finish so its callback doesn't
        # fire after stop() returns.
        deadline = time.time() + 1.0
        while self._partial_pending.is_set() and time.time() < deadline:
            time.sleep(0.01)

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

    def _run_audio(self) -> None:
        import sounddevice as sd

        cfg = self.config
        log.info(
            "STT audio capture started (chunk=%.2fs, device=%s)",
            cfg.chunk_seconds, cfg.audio_device,
        )
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
                self._audio_queue.put(audio[:, 0].copy())

    def _run_processor(self) -> None:
        if self._model is None:
            self._load_model()
        cfg = self.config
        log.info(
            "STT processor started (model=%s, device=%s)",
            cfg.model_size, cfg.device,
        )
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process_chunk(chunk, time.time())
        # Drain any chunks the audio thread captured before stop_event;
        # stop() joins audio first so this consumes everything that was
        # buffered, with no risk of more chunks arriving mid-drain.
        while True:
            try:
                chunk = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            self._process_chunk(chunk, time.time())

    def _process_chunk(self, chunk: np.ndarray, now: float) -> None:
        """One step of the VAD/buffering state machine.

        Mutates `_speech_buffer`, `_in_speech`, `_last_partial_t`. Kicks
        a partial worker (if eligible) and triggers final transcription
        on silence-after-speech or max-samples cap.
        """
        cfg = self.config
        if is_speech(chunk, cfg.silence_threshold):
            self._speech_buffer.append(chunk)
            self._in_speech = True
            total_samples = sum(len(c) for c in self._speech_buffer)

            if (
                self.partial_callback is not None
                and not self._partial_pending.is_set()
                and now - self._last_partial_t >= cfg.partial_interval_seconds
            ):
                self._kick_partial(np.concatenate(self._speech_buffer))
                self._last_partial_t = now

            if total_samples >= cfg.max_samples:
                self._do_transcribe(self._speech_buffer)
                self._reset_speech_state()
                if self.speech_end_callback is not None:
                    self.speech_end_callback()
        elif self._in_speech:
            total_samples = sum(len(c) for c in self._speech_buffer)
            if total_samples >= cfg.min_samples:
                self._do_transcribe(self._speech_buffer)
            if self.speech_end_callback is not None:
                self.speech_end_callback()
            self._reset_speech_state()

    def _reset_speech_state(self) -> None:
        self._speech_buffer = []
        self._in_speech = False
        self._last_partial_t = 0.0

    def _kick_partial(self, audio: np.ndarray) -> None:
        """Run a partial transcription on a short-lived worker thread.

        Caller has already verified `_partial_pending` is clear; this
        method sets the flag and the worker clears it on exit.
        """
        self._partial_pending.set()
        assert self._model is not None

        def worker() -> None:
            try:
                assert self._model is not None
                text = transcribe(self._model, audio)
                cb = self.partial_callback
                if text and cb is not None:
                    cb(text)
            except Exception:
                log.exception("Partial transcription failed")
            finally:
                self._partial_pending.clear()

        threading.Thread(target=worker, daemon=True, name="stt-partial").start()

    def _do_transcribe(self, chunks: list[np.ndarray]) -> None:
        assert self._model is not None
        audio = np.concatenate(chunks)
        text = transcribe(self._model, audio)
        if text:
            log.info("Transcribed: %s", text)
            self._text_queue.put(text)
