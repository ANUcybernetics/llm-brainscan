"""Capture weight snapshots from the model."""

import torch

from brainscan.model import GPT


def capture_weights(model: GPT) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}
