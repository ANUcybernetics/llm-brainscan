"""Capture weight snapshots and activation maps from the model."""

import torch
import torch.nn as nn

from brainscan.model import GPT


def capture_weights(model: GPT) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def capture_weight_deltas(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: current[name] - previous[name] for name in current}


class ActivationCapture:
    def __init__(self, model: GPT):
        self.model = model
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def install(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self.activations[name] = output.detach()

        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        self.activations.clear()
