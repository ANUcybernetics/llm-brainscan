import pytest
import torch
from conftest import SMALL_CONFIG

from brainscan.data import decode
from brainscan.layout import LAYOUT_HEIGHT, WIDTH
from brainscan.model import GPT


class TestGPTArchitecture:
    def test_creates_with_defaults(self):
        model = GPT()
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
        assert total <= WIDTH * LAYOUT_HEIGHT

    def test_creates_with_small_config(self, small_model):
        assert small_model is not None

    def test_default_model_fits_layout(self):
        model = GPT()
        total = sum(p.numel() for p in model.parameters())
        layout_pixels = WIDTH * LAYOUT_HEIGHT
        assert total <= layout_pixels, f"{total} params exceeds layout area {layout_pixels}"

    def test_has_expected_layers(self, small_model):
        param_names = {name for name, _ in small_model.named_parameters()}
        for i in range(SMALL_CONFIG["n_layer"]):
            prefix = f"blocks.{i}"
            assert any(n.startswith(f"{prefix}.attn") for n in param_names)
            assert any(n.startswith(f"{prefix}.mlp") for n in param_names)
            assert any(n.startswith(f"{prefix}.ln_1") for n in param_names)
            assert any(n.startswith(f"{prefix}.ln_2") for n in param_names)


class TestGPTForward:
    def test_forward_logits_shape(self, small_model, device):
        x = torch.randint(0, 256, (2, 16), device=device)
        logits, loss = small_model(x)
        assert logits.shape == (2, 16, 256)
        assert loss is None

    def test_forward_with_targets(self, small_model, device):
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        logits, loss = small_model(x, y)
        assert logits.shape == (2, 16, 256)
        assert loss is not None
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_forward_single_token(self, small_model, device):
        x = torch.randint(0, 256, (1, 1), device=device)
        logits, _ = small_model(x)
        assert logits.shape == (1, 1, 256)

    def test_forward_max_sequence_length(self, small_model, device):
        seq_len = SMALL_CONFIG["sequence_len"]
        x = torch.randint(0, 256, (1, seq_len), device=device)
        logits, _ = small_model(x)
        assert logits.shape == (1, seq_len, 256)

    def test_forward_exceeding_sequence_length_raises(self, small_model, device):
        seq_len = SMALL_CONFIG["sequence_len"]
        x = torch.randint(0, 256, (1, seq_len + 1), device=device)
        with pytest.raises(AssertionError):
            small_model(x)

    def test_gradients_flow(self, small_model, device):
        x = torch.randint(0, 256, (2, 16), device=device)
        y = torch.randint(0, 256, (2, 16), device=device)
        _, loss = small_model(x, y)
        loss.backward()
        for name, param in small_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestGPTGenerate:
    def test_returns_tokens_and_probs(self, small_model, device):
        tokens, probs = small_model.generate(b"hello", max_tokens=10)
        assert len(tokens) == 15  # 5 prompt + 10 generated
        assert len(probs) == 15
        assert all(0 <= t < 256 for t in tokens)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_prompt_probs_are_one(self, small_model, device):
        _, probs = small_model.generate(b"ab", max_tokens=5)
        assert probs[0] == 1.0
        assert probs[1] == 1.0

    def test_output_is_decodable(self, small_model, device):
        tokens, _ = small_model.generate(b"test", max_tokens=20)
        result = decode(tokens)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_restores_training_mode(self, small_model, device):
        small_model.train()
        small_model.generate(b"x", max_tokens=5)
        assert small_model.training

    def test_works_in_eval_mode(self, small_model, device):
        small_model.eval()
        tokens, _ = small_model.generate(b"x", max_tokens=5)
        assert len(tokens) == 6
        assert not small_model.training

    def test_respects_max_tokens(self, small_model, device):
        tokens, _ = small_model.generate(b"a", max_tokens=3)
        assert len(tokens) == 4  # 1 prompt + 3 generated

    def test_infers_device(self, small_model):
        tokens, _ = small_model.generate(b"x", max_tokens=3)
        assert len(tokens) == 4


class TestGPTGenerationLegacy:
    def test_greedy_generation(self, small_model, device):
        small_model.eval()
        context = torch.randint(0, 256, (1, 4), device=device)
        with torch.no_grad():
            for _ in range(10):
                logits, _ = small_model(context[:, -SMALL_CONFIG["sequence_len"] :])
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                context = torch.cat([context, next_token], dim=1)
        assert context.shape == (1, 14)

    def test_sampling_generation(self, small_model, device):
        small_model.eval()
        context = torch.randint(0, 256, (1, 4), device=device)
        with torch.no_grad():
            for _ in range(10):
                logits, _ = small_model(context[:, -SMALL_CONFIG["sequence_len"] :])
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
        assert context.shape == (1, 14)
        assert (context >= 0).all() and (context < 256).all()
