import queue
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
    SpeechListener,
)


@pytest.fixture
def listener():
    return SpeechListener(model_size="tiny", device="cpu")


class TestSpeechDetection:
    def test_silence_not_detected(self, listener):
        silence = np.zeros(1600, dtype=np.float32)
        assert not listener._is_speech(silence)

    def test_loud_audio_detected(self, listener):
        loud = np.full(1600, 0.5, dtype=np.float32)
        assert listener._is_speech(loud)

    def test_threshold_boundary(self, listener):
        just_below = np.full(1600, SILENCE_THRESHOLD * 0.9, dtype=np.float32)
        assert not listener._is_speech(just_below)

        just_above = np.full(1600, SILENCE_THRESHOLD * 1.1, dtype=np.float32)
        assert listener._is_speech(just_above)


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


class TestTranscribe:
    def test_transcribes_and_queues(self, listener):
        mock_segment = MagicMock()
        mock_segment.text = " hello world "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)
        listener._model = mock_model

        chunks = [np.random.randn(3200).astype(np.float32)]
        listener._transcribe(chunks)

        result = listener.get_text()
        assert result == ["hello world"]

    def test_empty_transcription_not_queued(self, listener):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], None)
        listener._model = mock_model

        chunks = [np.random.randn(3200).astype(np.float32)]
        listener._transcribe(chunks)

        assert listener.get_text() == []

    def test_multiple_segments_joined(self, listener):
        seg1 = MagicMock()
        seg1.text = " hello "
        seg2 = MagicMock()
        seg2.text = " world "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg1, seg2], None)
        listener._model = mock_model

        listener._transcribe([np.zeros(3200, dtype=np.float32)])
        assert listener.get_text() == ["hello world"]

    def test_callback_fired(self, listener):
        received = []
        listener._on_transcription = received.append

        mock_segment = MagicMock()
        mock_segment.text = "test"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)
        listener._model = mock_model

        listener._transcribe([np.zeros(3200, dtype=np.float32)])
        assert received == ["test"]


class TestStartStop:
    def test_start_creates_thread(self, listener):
        gate = threading.Event()

        def blocking_run():
            gate.wait(timeout=5.0)

        with patch.object(listener, "_run", side_effect=blocking_run):
            listener.start()
            assert listener._thread is not None
            assert listener._thread.is_alive()
            gate.set()
            listener.stop()
            assert listener._thread is None

    def test_start_idempotent(self, listener):
        gate = threading.Event()

        def blocking_run():
            gate.wait(timeout=5.0)

        with patch.object(listener, "_run", side_effect=blocking_run):
            listener.start()
            thread1 = listener._thread
            listener.start()
            assert listener._thread is thread1
            gate.set()
            listener.stop()

    def test_stop_without_start(self, listener):
        listener.stop()


class TestChunkSeconds:
    def test_default_chunk_seconds(self):
        listener = SpeechListener(model_size="tiny")
        assert listener._chunk_seconds == CHUNK_SECONDS

    def test_custom_chunk_seconds(self):
        listener = SpeechListener(model_size="tiny", chunk_seconds=1.0)
        assert listener._chunk_seconds == 1.0

    def test_chunk_seconds_affects_audio_read_size(self):
        chunk_sec = 1.0
        chunk_samples = int(chunk_sec * SAMPLE_RATE)
        speech_chunk = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        silence_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)

        read_count = 0

        def mock_read(n):
            nonlocal read_count
            read_count += 1
            assert n == chunk_samples
            if read_count == 1:
                return speech_chunk, False
            if read_count == 2:
                return silence_chunk, False
            listener._stop_event.set()
            return silence_chunk, False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        sd_mock = sys.modules["sounddevice"]
        sd_mock.InputStream = MagicMock(return_value=mock_stream)

        listener = SpeechListener(
            model_size="tiny", chunk_seconds=chunk_sec
        )

        mock_segment = MagicMock()
        mock_segment.text = "short chunk"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch.object(listener, "_load_model"):
            listener._model = mock_model
            listener._run()

        assert listener.get_text() == ["short chunk"]
        sd_mock.InputStream.assert_called_once_with(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            device=None,
        )


class TestAudioLoop:
    def test_speech_then_silence_triggers_transcription(self):
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        speech_chunk = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        silence_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)

        read_count = 0

        def mock_read(n):
            nonlocal read_count
            read_count += 1
            if read_count <= 2:
                return speech_chunk, False
            if read_count == 3:
                return silence_chunk, False
            listener._stop_event.set()
            return silence_chunk, False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        sd_mock = sys.modules["sounddevice"]
        sd_mock.InputStream = MagicMock(return_value=mock_stream)

        listener = SpeechListener(model_size="tiny", device="cpu")

        mock_segment = MagicMock()
        mock_segment.text = "transcribed"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch.object(listener, "_load_model"):
            listener._model = mock_model
            listener._run()

        result = listener.get_text()
        assert result == ["transcribed"]
        assert mock_model.transcribe.call_count == 1
