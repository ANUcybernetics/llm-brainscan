from pathlib import Path

import numpy as np
import torch


def load_text_dataset(path: Path) -> bytes:
    return path.read_bytes()


def encode(text: bytes) -> list[int]:
    return list(text)


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def prepare_batches(
    data: bytes,
    batch_size: int,
    sequence_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    n = len(arr)
    ix = torch.randint(n - sequence_len, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(arr[i : i + sequence_len].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(arr[i + 1 : i + 1 + sequence_len].astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)
