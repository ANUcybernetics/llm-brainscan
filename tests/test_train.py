import sys
from unittest.mock import MagicMock, patch

import pytest


class TestSTTCliArgs:
    def _parse_args(self, cli_args: list[str]):
        with patch("sys.argv", ["train"] + cli_args):
            from brainscan.train import main

            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("--silence-threshold", type=float, default=0.01)
            parser.add_argument("--chunk-seconds", type=float, default=2.0)
            parser.add_argument("--min-speech-seconds", type=float, default=0.5)
            parser.add_argument("--max-speech-seconds", type=float, default=30.0)
            return parser.parse_args(cli_args)

    def test_default_silence_threshold(self):
        args = self._parse_args([])
        assert args.silence_threshold == 0.01

    def test_custom_silence_threshold(self):
        args = self._parse_args(["--silence-threshold", "0.05"])
        assert args.silence_threshold == 0.05

    def test_default_chunk_seconds(self):
        args = self._parse_args([])
        assert args.chunk_seconds == 2.0

    def test_custom_chunk_seconds(self):
        args = self._parse_args(["--chunk-seconds", "1.0"])
        assert args.chunk_seconds == 1.0

    def test_default_min_speech_seconds(self):
        args = self._parse_args([])
        assert args.min_speech_seconds == 0.5

    def test_custom_min_speech_seconds(self):
        args = self._parse_args(["--min-speech-seconds", "0.3"])
        assert args.min_speech_seconds == 0.3

    def test_default_max_speech_seconds(self):
        args = self._parse_args([])
        assert args.max_speech_seconds == 30.0

    def test_custom_max_speech_seconds(self):
        args = self._parse_args(["--max-speech-seconds", "15.0"])
        assert args.max_speech_seconds == 15.0


class TestSTTArgsPassthrough:
    def test_listener_receives_all_stt_args(self):
        if "sounddevice" not in sys.modules:
            sys.modules["sounddevice"] = MagicMock()

        from brainscan.stt import SpeechListener

        listener = SpeechListener(
            model_size="tiny",
            device="cpu",
            chunk_seconds=1.5,
            silence_threshold=0.05,
            min_speech_seconds=0.3,
            max_speech_seconds=15.0,
        )

        assert listener._chunk_seconds == 1.5
        assert listener._silence_threshold == 0.05
        assert listener._min_samples == int(0.3 * 16000)
        assert listener._max_samples == int(15.0 * 16000)
