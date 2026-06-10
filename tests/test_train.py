import sys
from unittest.mock import MagicMock

import numpy as np
import torch
from conftest import SMALL_CONFIG

from brainscan.conversation import Conversation, ListenerSnapshot
from brainscan.data import prepare_batches
from brainscan.lanes import LaneBuffer
from brainscan.model import GPT
from brainscan.renderer import LaneFrame, OffscreenRenderer, flatten_weights
from brainscan.snapshot import capture_weights


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


class TestConversationFrameWiring:
    def _make_renderer(self):
        return OffscreenRenderer(
            64, 144, audience_height=48, model_height=48
        )

    def test_render_with_lane_frames(self, small_model):
        weights = capture_weights(small_model)
        renderer = self._make_renderer()

        a_buf = LaneBuffer(capacity=renderer.config.lane_capacity)
        a_buf.push_text("hi", attrs=0)
        m_buf = LaneBuffer(capacity=renderer.config.lane_capacity)
        m_buf.push_text("yo", prob=0.8)

        a_chars, a_attrs, _ = a_buf.snapshot()
        m_chars, _, m_probs = m_buf.snapshot()

        canvas_pixels = renderer.width * renderer.height
        np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
        flat, count = flatten_weights(np_weights)
        buf = np.zeros(canvas_pixels, dtype=np.float32)
        n = min(count, canvas_pixels)
        buf[:n] = flat[:n]

        img = renderer.render(
            buf,
            audience=LaneFrame(chars=a_chars, attrs_or_probs=a_attrs, count=a_buf.count),
            model=LaneFrame(chars=m_chars, attrs_or_probs=m_probs, count=m_buf.count),
        )
        assert img.shape == (144, 64, 4)


class TestTrainLoopUsesConversation:
    def test_committed_input_appended_to_corpus(self, device):
        from brainscan.data import TextBuffer
        from brainscan.train import _process_committed

        buf = TextBuffer(b"seed")
        listener = ListenerSnapshot(committed=["from-mic"])
        _process_committed(listener, buf)
        assert b"from-mic" in buf.data


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


def test_daily_seed_is_reproducible():
    """Same date + same base seed must produce the same daily seed across runs."""
    import datetime as dt
    import hashlib

    def daily_seed(date: dt.date, base: int) -> int:
        return int.from_bytes(
            hashlib.sha256(
                f"{date.isoformat()}-{base}".encode()
            ).digest()[:4],
            "big",
        )

    a = daily_seed(dt.date(2026, 4, 26), 42)
    b = daily_seed(dt.date(2026, 4, 26), 42)
    c = daily_seed(dt.date(2026, 4, 27), 42)
    d = daily_seed(dt.date(2026, 4, 26), 99)

    assert a == b
    assert a != c
    assert a != d


def test_train_main_smoke(tmp_path, monkeypatch):
    """Run main() for a few steps and verify a frame is saved."""
    import sys

    from brainscan import train as train_mod

    # tiny model so the test runs in seconds
    args = [
        "train",
        "--no-mic",
        "--steps", "5",
        "--snapshot-every", "1",
        "--n-layer", "1",
        "--n-head", "1",
        "--n-embd", "16",
        "--sequence-len", "16",
        "--batch-size", "2",
        "--gen-tokens", "4",
        "--save-images",
        "--output-dir", str(tmp_path),
        "--data", str(tmp_path / "tiny.txt"),
    ]
    (tmp_path / "tiny.txt").write_bytes(b"abcdefghij" * 200)

    monkeypatch.setattr(sys, "argv", args)
    train_mod.main()
    frames = sorted((tmp_path / "frames").glob("*.png"))
    assert len(frames) >= 1




class TestBuildDeltaBuffer:
    def test_identical_buffers_give_zero_delta(self):
        from brainscan.train import _build_delta_buffer

        buf = np.random.default_rng(0).normal(size=64).astype(np.float32)
        delta = _build_delta_buffer(buf, buf.copy())
        assert delta.shape == buf.shape
        assert (delta == 0.0).all()

    def test_uniform_motion_is_no_signal(self):
        # Differential semantics: when every parameter moved by the same
        # amount there is no exceptional motion, so nothing lights up.
        from brainscan.train import _build_delta_buffer

        prev = np.zeros(100, dtype=np.float32)
        buf = np.full(100, 0.5, dtype=np.float32)
        delta = _build_delta_buffer(buf, prev)
        assert (delta == 0.0).all()

    def test_only_above_median_motion_lights_up(self):
        from brainscan.train import _build_delta_buffer

        prev = np.zeros(1000, dtype=np.float32)
        buf = np.full(1000, 0.1, dtype=np.float32)  # gradient-noise floor
        buf[:20] = 1.0  # exceptional motion
        delta = _build_delta_buffer(buf, prev)
        assert delta.dtype == np.float32
        assert delta.max() <= 1.0
        assert (delta[20:] == 0.0).all(), "median-level motion stays dark"
        assert delta[:20].min() > 0.9, "exceptional motion flashes bright"
