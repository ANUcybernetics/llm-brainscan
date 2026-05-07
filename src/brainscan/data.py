from pathlib import Path
from threading import Lock

import numpy as np
import torch


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

    def rotate(self, target: Path) -> None:
        if self._persist_path is None:
            raise ValueError("TextBuffer has no persist path to rotate")
        with self._lock:
            if not self._persist_path.exists():
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.replace(target)

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
    *,
    seed_corpus: np.ndarray | None = None,
    seed_weight: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a training batch.

    Single-source mode (seed_corpus=None): all rows sampled from data.
    Mixed mode: seed_weight of rows sampled from seed_corpus, rest from data
    (the audience). Falls back to seed-only when audience is shorter than
    sequence_len + 1.
    """
    raw = data.data if isinstance(data, TextBuffer) else data
    arr = np.frombuffer(raw, dtype=np.uint8)

    if seed_corpus is None:
        return _sample_single(arr, batch_size, sequence_len, device)

    if len(arr) < sequence_len + 1:
        return _sample_single(seed_corpus, batch_size, sequence_len, device)

    n_seed_rows = round(batch_size * seed_weight)
    n_aud_rows = batch_size - n_seed_rows

    rows_x: list[torch.Tensor] = []
    rows_y: list[torch.Tensor] = []
    if n_seed_rows > 0:
        _append_rows(seed_corpus, n_seed_rows, sequence_len, rows_x, rows_y)
    if n_aud_rows > 0:
        _append_rows(arr, n_aud_rows, sequence_len, rows_x, rows_y)
    return torch.stack(rows_x).to(device), torch.stack(rows_y).to(device)


def _sample_single(
    arr: np.ndarray,
    batch_size: int,
    sequence_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_x: list[torch.Tensor] = []
    rows_y: list[torch.Tensor] = []
    _append_rows(arr, batch_size, sequence_len, rows_x, rows_y)
    return torch.stack(rows_x).to(device), torch.stack(rows_y).to(device)


def _append_rows(
    arr: np.ndarray,
    n_rows: int,
    sequence_len: int,
    rows_x: list[torch.Tensor],
    rows_y: list[torch.Tensor],
) -> None:
    ix = torch.randint(len(arr) - sequence_len, (n_rows,))
    for i in ix:
        i_int = int(i)
        rows_x.append(
            torch.from_numpy(
                arr[i_int : i_int + sequence_len].astype(np.int64)
            )
        )
        rows_y.append(
            torch.from_numpy(
                arr[i_int + 1 : i_int + 1 + sequence_len].astype(np.int64)
            )
        )
