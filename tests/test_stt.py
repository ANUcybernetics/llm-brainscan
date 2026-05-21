import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()

from brainscan.stt import (
    CHUNK_SECONDS,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    SpeechConfig,
    SpeechListener,
    is_speech,
    transcribe,
)


@pytest.fixture
def listener():
    return SpeechListener(model_size="tiny", device="cpu")


def _mock_model(text: str = "hello") -> MagicMock:
    seg = MagicMock()
    seg.text = f" {text} "
    model = MagicMock()
    model.transcribe.return_value = ([seg], None)
    return model


def _drain_partials(listener: SpeechListener, timeout: float = 1.0) -> None:
    """Wait for any in-flight partial worker thread to finish."""
    deadline = time.time() + timeout
    while listener._partial_pending.is_set() and time.time() < deadline:
        time.sleep(0.005)


class TestSpeechDetection:
    def test_silence_not_detected(self):
        silence = np.zeros(1600, dtype=np.float32)
        assert not is_speech(silence, SILENCE_THRESHOLD)

    def test_loud_audio_detected(self):
        loud = np.full(1600, 0.5, dtype=np.float32)
        assert is_speech(loud, SILENCE_THRESHOLD)

    def test_threshold_boundary(self):
        just_below = np.full(1600, SILENCE_THRESHOLD * 0.9, dtype=np.float32)
        assert not is_speech(just_below, SILENCE_THRESHOLD)

        just_above = np.full(1600, SILENCE_THRESHOLD * 1.1, dtype=np.float32)
        assert is_speech(just_above, SILENCE_THRESHOLD)


class TestTranscribe:
    def test_transcribes_audio(self):
        audio = np.random.randn(3200).astype(np.float32)
        result = transcribe(_mock_model("hello world"), audio)
        assert result == "hello world"

    def test_empty_transcription(self):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        result = transcribe(mock_model, np.zeros(3200, dtype=np.float32))
        assert result == ""

    def test_multiple_segments_joined(self):
        seg1 = MagicMock(); seg1.text = " hello "
        seg2 = MagicMock(); seg2.text = " world "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)

        result = transcribe(mock_model, np.zeros(3200, dtype=np.float32))
        assert result == "hello world"

    def test_whitespace_only_segments_return_empty(self):
        # Whisper returns empty segments for silence; joined with " " they
        # form a truthy run of spaces unless the result is stripped.
        seg1 = MagicMock(); seg1.text = " "
        seg2 = MagicMock(); seg2.text = "  "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        result = transcribe(mock_model, np.zeros(3200, dtype=np.float32))
        assert result == ""


class TestSpeechConfig:
    def test_defaults(self):
        cfg = SpeechConfig()
        assert cfg.sample_rate == SAMPLE_RATE
        assert cfg.chunk_seconds == CHUNK_SECONDS
        assert cfg.silence_threshold == SILENCE_THRESHOLD

    def test_computed_properties(self):
        cfg = SpeechConfig(min_speech_seconds=0.5, max_speech_seconds=30.0)
        assert cfg.min_samples == int(0.5 * SAMPLE_RATE)
        assert cfg.max_samples == int(30.0 * SAMPLE_RATE)
        assert cfg.chunk_samples == int(CHUNK_SECONDS * SAMPLE_RATE)

    def test_audio_device_default_none(self):
        assert SpeechConfig().audio_device is None
        assert SpeechConfig(audio_device=3).audio_device == 3


class TestGetText:
    def test_empty_queue(self, listener):
        assert listener.get_text() == []

    def test_drains_queue(self, listener):
        listener._text_queue.put("hello")
        listener._text_queue.put("world")
        result = listener.get_text()
        assert result == ["hello", "world"]
        assert listener.get_text() == []

    def test_successive_drains(self, listener):
        listener._text_queue.put("first")
        assert listener.get_text() == ["first"]
        listener._text_queue.put("second")
        assert listener.get_text() == ["second"]


class TestDoTranscribe:
    def test_transcribes_and_queues(self, listener):
        listener._model = _mock_model("hello world")
        listener._do_transcribe([np.random.randn(3200).astype(np.float32)])
        assert listener.get_text() == ["hello world"]

    def test_empty_transcription_not_queued(self, listener):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        listener._model = mock_model
        listener._do_transcribe([np.random.randn(3200).astype(np.float32)])
        assert listener.get_text() == []

    def test_whitespace_transcription_not_queued(self, listener):
        # Silence transcribes to empty/whitespace segments; nothing should
        # reach the text queue (no spurious "> mic > " tag downstream).
        seg1 = MagicMock(); seg1.text = " "
        seg2 = MagicMock(); seg2.text = "  "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        listener._model = mock_model
        listener._do_transcribe([np.random.randn(3200).astype(np.float32)])
        assert listener.get_text() == []


class TestStartStop:
    def test_start_creates_both_threads(self, listener):
        gate = threading.Event()

        def blocking() -> None:
            gate.wait(timeout=5.0)

        with patch.object(listener, "_run_audio", side_effect=blocking), \
             patch.object(listener, "_run_processor", side_effect=blocking):
            listener.start()
            assert listener._audio_thread is not None
            assert listener._processor_thread is not None
            assert listener._audio_thread.is_alive()
            assert listener._processor_thread.is_alive()
            gate.set()
            listener.stop()
            assert listener._audio_thread is None
            assert listener._processor_thread is None

    def test_start_idempotent(self, listener):
        gate = threading.Event()

        def blocking() -> None:
            gate.wait(timeout=5.0)

        with patch.object(listener, "_run_audio", side_effect=blocking), \
             patch.object(listener, "_run_processor", side_effect=blocking):
            listener.start()
            audio_t = listener._audio_thread
            proc_t = listener._processor_thread
            listener.start()
            assert listener._audio_thread is audio_t
            assert listener._processor_thread is proc_t
            gate.set()
            listener.stop()

    def test_stop_without_start(self, listener):
        listener.stop()


class TestRunAudio:
    """Verify the audio thread reads the configured chunk size and pushes to the queue."""

    def _drive_audio(self, listener: SpeechListener, chunks: list[np.ndarray]) -> MagicMock:
        counter = [0]

        def mock_read(n):
            assert n == listener.config.chunk_samples
            i = counter[0]
            counter[0] += 1
            if i < len(chunks):
                return chunks[i], False
            listener._stop_event.set()
            return chunks[-1], False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        input_stream = MagicMock(return_value=mock_stream)
        setattr(sys.modules["sounddevice"], "InputStream", input_stream)

        listener._run_audio()
        return input_stream

    def test_default_chunk_seconds_used(self):
        listener = SpeechListener(model_size="tiny")
        assert listener.config.chunk_seconds == CHUNK_SECONDS

    def test_custom_chunk_seconds_drives_stream_read(self):
        listener = SpeechListener(model_size="tiny", chunk_seconds=1.0)
        chunk_samples = listener.config.chunk_samples
        speech = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        input_stream = self._drive_audio(listener, [speech, speech])

        assert listener._audio_queue.qsize() >= 2
        input_stream.assert_called_once_with(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=None,
        )

    def test_audio_device_passed_through(self):
        listener = SpeechListener(model_size="tiny", audio_device=7)
        chunk_samples = listener.config.chunk_samples
        chunk = np.zeros((chunk_samples, 1), dtype=np.float32)
        input_stream = self._drive_audio(listener, [chunk])
        _, kwargs = input_stream.call_args
        assert kwargs["device"] == 7


class TestProcessChunk:
    """Drive the per-chunk VAD/buffering state machine deterministically."""

    def _make_listener(self, **kwargs) -> SpeechListener:
        listener = SpeechListener(model_size="tiny", **kwargs)
        listener._model = _mock_model("transcribed")
        return listener

    def _speech(self, listener: SpeechListener) -> np.ndarray:
        return np.full(listener.config.chunk_samples, 0.5, dtype=np.float32)

    def _silence(self, listener: SpeechListener) -> np.ndarray:
        return np.zeros(listener.config.chunk_samples, dtype=np.float32)

    def test_silence_alone_does_nothing(self):
        listener = self._make_listener()
        listener._process_chunk(self._silence(listener), now=0.0)
        assert listener.get_text() == []
        assert not listener._in_speech

    def test_speech_then_silence_commits_transcription(self):
        listener = self._make_listener()
        # Need ≥ min_speech_seconds (0.5s = 8000 samples). At 0.2s chunks
        # that's 3 chunks.
        for i in range(3):
            listener._process_chunk(self._speech(listener), now=0.2 * i)
        listener._process_chunk(self._silence(listener), now=0.6)
        assert listener.get_text() == ["transcribed"]

    def test_speech_too_brief_is_discarded(self):
        listener = self._make_listener()
        # 1 chunk = 0.2s = 3200 samples < min_samples (8000) → no transcribe
        listener._process_chunk(self._speech(listener), now=0.0)
        listener._process_chunk(self._silence(listener), now=0.2)
        assert listener.get_text() == []

    def test_partial_callback_invoked_during_speech(self):
        partials: list[str] = []
        listener = self._make_listener(partial_interval_seconds=0.0)
        listener.partial_callback = partials.append

        listener._process_chunk(self._speech(listener), now=0.0)
        _drain_partials(listener)
        listener._process_chunk(self._speech(listener), now=0.5)
        _drain_partials(listener)
        listener._process_chunk(self._speech(listener), now=1.0)
        _drain_partials(listener)

        assert len(partials) >= 1
        assert all(p == "transcribed" for p in partials)

    def test_partial_callback_skipped_while_worker_in_flight(self):
        listener = self._make_listener(partial_interval_seconds=0.0)
        listener.partial_callback = lambda _t: None
        listener._partial_pending.set()  # simulate worker in flight

        with patch.object(listener, "_kick_partial") as kick:
            listener._process_chunk(self._speech(listener), now=0.0)
            kick.assert_not_called()

    def test_speech_end_callback_fires_on_silence(self):
        ended: list[bool] = []
        listener = self._make_listener()
        listener.speech_end_callback = lambda: ended.append(True)

        for i in range(3):
            listener._process_chunk(self._speech(listener), now=0.2 * i)
        listener._process_chunk(self._silence(listener), now=0.6)

        assert ended == [True]

    def test_speech_end_callback_fires_on_max_samples(self):
        """Regression: previously the max_samples cap reset the buffer
        without firing speech_end_callback, leaving partial UI state."""
        ended: list[bool] = []
        listener = self._make_listener(max_speech_seconds=1.0)
        listener.speech_end_callback = lambda: ended.append(True)

        speech = self._speech(listener)
        # 5 chunks of 0.2s = 1.0s = max_samples (cap fires on the 5th).
        for i in range(5):
            listener._process_chunk(speech, now=0.2 * i)

        assert ended == [True], "speech_end_callback must fire on max_samples cap"
        assert listener.get_text() == ["transcribed"]
        assert not listener._in_speech, "max_samples reset should clear in_speech"

    def test_no_callbacks_does_not_raise(self):
        listener = self._make_listener()
        # No partial_callback, no speech_end_callback.
        for i in range(3):
            listener._process_chunk(self._speech(listener), now=0.2 * i)
        listener._process_chunk(self._silence(listener), now=0.6)
        assert listener.get_text() == ["transcribed"]


class TestPipelineIntegration:
    """End-to-end test: scripted input stream → audio thread → processor → text queue."""

    def test_pipeline_produces_transcription(self):
        listener = SpeechListener(model_size="tiny")
        listener._model = _mock_model("integrated")

        chunk_samples = listener.config.chunk_samples
        speech = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        silence = np.zeros((chunk_samples, 1), dtype=np.float32)
        # 3 speech chunks (≥ min_samples) then silence to commit.
        chunks = [speech, speech, speech, silence]
        counter = [0]

        def mock_read(n):
            i = counter[0]
            counter[0] += 1
            if i < len(chunks):
                return chunks[i], False
            listener._stop_event.set()
            return silence, False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        setattr(
            sys.modules["sounddevice"],
            "InputStream",
            MagicMock(return_value=mock_stream),
        )

        listener.start()
        deadline = time.time() + 3.0
        while not listener._stop_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        # let processor drain remaining chunks
        time.sleep(0.2)
        listener.stop()

        assert listener.get_text() == ["integrated"]
