from unittest.mock import MagicMock, patch

from brainscan.tts import TTSEngine


class TestTTSEngineDisabled:
    def test_disabled_speak_is_noop(self):
        eng = TTSEngine(enabled=False)
        duration = eng.speak("hello")
        assert duration == 0.0

    def test_disabled_does_not_load(self):
        eng = TTSEngine(enabled=False)
        assert eng._voice is None


class TestTTSEngineEnabled:
    def test_speak_returns_estimated_duration(self):
        with patch("brainscan.tts._load_voice") as mock_load:
            mock_voice = MagicMock()
            mock_voice.config.sample_rate = 22050
            audio = MagicMock()
            audio.audio_int16_bytes = b"\x00\x00" * 22050  # 1.0s
            mock_voice.synthesize.return_value = iter([audio])
            mock_load.return_value = mock_voice

            with patch("brainscan.tts.sd") as mock_sd:
                eng = TTSEngine(enabled=True, voice="en_AU-fitch-medium")
                duration = eng.speak("hello world")
                assert duration > 0.0
                assert mock_sd.play.called

    def test_speak_returns_zero_for_empty(self):
        with patch("brainscan.tts._load_voice"):
            eng = TTSEngine(enabled=True, voice="en_AU-fitch-medium")
            eng._voice = MagicMock()
            assert eng.speak("") == 0.0
            assert eng.speak("   ") == 0.0
