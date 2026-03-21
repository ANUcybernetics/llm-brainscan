import sys
from unittest.mock import MagicMock

import numpy as np
import torch
from conftest import SMALL_CONFIG

from brainscan.data import prepare_batches
from brainscan.model import GPT
from brainscan.renderer import OffscreenRenderer
from brainscan.snapshot import capture_weights
from brainscan.train import prepare_display_buffers, render_frame


class TestTrainingLoop:
    def test_loss_decreases(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pattern = b"abcabc" * 200

        losses = []
        for _ in range(50):
            x, y = prepare_batches(
                pattern,
                batch_size=8,
                sequence_len=SMALL_CONFIG["sequence_len"],
                device=device,
            )
            _, loss = model(x, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_snapshot_during_training(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = b"hello world " * 200

        prev = capture_weights(model)
        snapshots = [prev]
        for _step in range(5):
            x, y = prepare_batches(
                data,
                batch_size=4,
                sequence_len=SMALL_CONFIG["sequence_len"],
                device=device,
            )
            _, loss = model(x, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            current = capture_weights(model)
            snapshots.append(current)
            prev = current
        assert len(snapshots) == 6
        first_param = next(iter(snapshots[0].keys()))
        assert not torch.equal(snapshots[0][first_param], snapshots[-1][first_param])


class TestPrepareDisplayBuffers:
    def _canvas_for(self, model):
        total = sum(p.numel() for p in model.parameters())
        return total + 1024

    def test_returns_correct_shapes(self, small_model):
        weights = capture_weights(small_model)
        canvas = self._canvas_for(small_model)
        buf, chars, probs = prepare_display_buffers(
            weights,
            list(weights.keys()),
            canvas,
            text_chars=[65, 66],
            text_probs=[1.0, 0.5],
        )
        assert buf.shape == (canvas,)
        assert buf.dtype == np.float32
        assert chars is not None
        assert probs is not None
        assert chars.dtype == np.uint32
        assert probs.dtype == np.float32

    def test_none_text_returns_none(self, small_model):
        weights = capture_weights(small_model)
        canvas = self._canvas_for(small_model)
        buf, chars, probs = prepare_display_buffers(
            weights, list(weights.keys()), canvas
        )
        assert chars is None
        assert probs is None

    def test_buffer_size_matches_canvas(self, small_model):
        weights = capture_weights(small_model)
        total_params = sum(p.numel() for p in small_model.parameters())
        canvas = total_params + 1024
        buf, _, _ = prepare_display_buffers(
            weights, list(weights.keys()), canvas
        )
        assert len(buf) == canvas
        assert np.count_nonzero(buf) > 0
        assert np.count_nonzero(buf) <= total_params


class TestRenderFrame:
    def test_produces_image(self, small_model):
        weights = capture_weights(small_model)
        renderer = OffscreenRenderer(32, 32)
        img = render_frame(renderer, weights, list(weights.keys()))
        assert img.shape == (32, 32, 4)
        assert img.dtype == np.uint8

    def test_with_text(self, small_model):
        weights = capture_weights(small_model)
        renderer = OffscreenRenderer(64, 64, text_strip_height=16, text_scale=1)
        img = render_frame(
            renderer,
            weights,
            list(weights.keys()),
            text_chars=[ord("A"), ord("B")],
            text_probs=[1.0, 0.5],
        )
        assert img.shape == (64, 64, 4)


class TestEndToEndTrainAndRender:
    def test_train_snapshot_render_cycle(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        data = b"the quick brown fox " * 100

        x, y = prepare_batches(
            data, batch_size=4, sequence_len=SMALL_CONFIG["sequence_len"], device=device
        )
        _, loss = model(x, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        weights = capture_weights(model)
        tokens, probs = model.generate(b"the ", max_tokens=10)

        renderer = OffscreenRenderer(32, 32, text_strip_height=16, text_scale=1)
        img = render_frame(
            renderer,
            weights,
            list(weights.keys()),
            text_chars=tokens,
            text_probs=probs,
        )
        assert img.shape == (32, 32, 4)
        assert np.any(img[:, :, :3] > 20)


class TestSTTArgsPassthrough:
    def test_listener_receives_all_stt_args(self):
        if "sounddevice" not in sys.modules:
            sys.modules["sounddevice"] = MagicMock()

        from brainscan.stt import SpeechConfig, SpeechListener

        cfg = SpeechConfig(
            model_size="tiny",
            device="cpu",
            chunk_seconds=1.5,
            silence_threshold=0.05,
            min_speech_seconds=0.3,
            max_speech_seconds=15.0,
        )
        listener = SpeechListener(config=cfg)

        assert listener.config.chunk_seconds == 1.5
        assert listener.config.silence_threshold == 0.05
        assert listener.config.min_samples == int(0.3 * 16000)
        assert listener.config.max_samples == int(15.0 * 16000)
