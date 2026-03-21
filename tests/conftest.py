import pytest
import torch

from brainscan.model import GPT

SMALL_CONFIG = {
    "vocab_size": 256,
    "sequence_len": 32,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
}


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model(device):
    return GPT(**SMALL_CONFIG).to(device)


@pytest.fixture
def sample_text():
    return b"To be, or not to be, that is the question. " * 100
