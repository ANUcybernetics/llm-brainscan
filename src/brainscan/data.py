from pathlib import Path
from threading import Lock

import numpy as np
import torch


def load_text_dataset(path: Path) -> bytes:
    return path.read_bytes()


def encode(text: bytes) -> list[int]:
    return list(text)


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


class TextBuffer:
    def __init__(self, initial: bytes, persist_path: Path | None = None):
        self._buf = bytearray(initial)
        self._persist_path = persist_path
        self._lock = Lock()
        if persist_path and persist_path.exists():
            self._buf.extend(persist_path.read_bytes())

    def append(self, text: str) -> None:
        encoded = text.encode("utf-8")
        with self._lock:
            self._buf.extend(encoded)
            if self._persist_path:
                with open(self._persist_path, "ab") as f:
                    f.write(encoded)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def data(self) -> bytes:
        with self._lock:
            return bytes(self._buf)


def prepare_batches(
    data: bytes | TextBuffer,
    batch_size: int,
    sequence_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = data.data if isinstance(data, TextBuffer) else data
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
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
