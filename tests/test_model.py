import pytest
import torch
from conftest import SMALL_CONFIG

from brainscan.model import GPT


class TestGPTArchitecture:
    def test_creates_with_defaults(self):
        model = GPT()
        total = model.param_count()["total"]
        assert total > 0
        assert total <= 7680 * 4320

    def test_creates_with_small_config(self, small_model):
        assert small_model is not None

    def test_param_count_groups_sum_to_total(self, small_model):
        groups = small_model.param_count()
        total = groups.pop("total")
        assert sum(groups.values()) == total

    def test_param_count_matches_pytorch(self, small_model):
        groups = small_model.param_count()
        pytorch_total = sum(p.numel() for p in small_model.parameters())
        assert groups["total"] == pytorch_total

    def test_default_model_fits_8k(self):
        model = GPT()
        total = model.param_count()["total"]
        assert total <= 7680 * 4320, f"{total} params exceeds 8K pixel count"

    def test_has_expected_blocks(self, small_model):
        groups = small_model.param_count()
        for i in range(SMALL_CONFIG["n_layer"]):
            assert f"block_{i}/attn" in groups
            assert f"block_{i}/mlp" in groups
            assert f"block_{i}/ln_1" in groups
            assert f"block_{i}/ln_2" in groups


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


class TestGPTGeneration:
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
