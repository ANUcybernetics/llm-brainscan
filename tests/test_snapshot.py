import torch
from conftest import SMALL_CONFIG

from brainscan.model import GPT
from brainscan.snapshot import capture_weights


class TestCaptureWeights:
    def test_captures_all_parameters(self, small_model):
        weights = capture_weights(small_model)
        param_names = {name for name, _ in small_model.named_parameters()}
        assert set(weights.keys()) == param_names

    def test_weights_are_detached_clones(self, small_model):
        weights = capture_weights(small_model)
        for name, tensor in weights.items():
            assert not tensor.requires_grad
            original = dict(small_model.named_parameters())[name].detach().clone()
            tensor.zero_()
            current = dict(small_model.named_parameters())[name]
            assert torch.equal(current.detach(), original)
