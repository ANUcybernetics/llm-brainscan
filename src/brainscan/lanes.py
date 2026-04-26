"""Circular character buffer for the dual-lane text strip.

Each lane holds chars + per-char attribute bits (partial / source-tag) + a
per-char probability (used by the model lane for brightness; harmless for the
audience lane). ``snapshot()`` returns three numpy arrays already shaped for
upload to a wgpu storage buffer in display order (oldest at index 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATTR_PARTIAL = 1 << 0
ATTR_SOURCE_TAG = 1 << 1


def _encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


@dataclass
class LaneBuffer:
    capacity: int = 320
    _chars: list[int] = field(default_factory=list)
    _attrs: list[int] = field(default_factory=list)
    _probs: list[float] = field(default_factory=list)
    _committed: int = 0

    @property
    def count(self) -> int:
        return len(self._chars)

    def push(self, byte: int, prob: float = 1.0, attrs: int = 0) -> None:
        self._chars.append(byte & 0xFF)
        self._attrs.append(attrs)
        self._probs.append(prob)
        self._trim()
        if not (attrs & ATTR_PARTIAL):
            self._committed = len(self._chars)

    def push_text(self, text: str, prob: float = 1.0, attrs: int = 0) -> None:
        for b in _encode_text(text):
            self.push(b, prob=prob, attrs=attrs)

    def replace_tail(
        self, text: str, prob: float = 1.0, attrs: int = ATTR_PARTIAL
    ) -> None:
        self._chars = self._chars[: self._committed]
        self._attrs = self._attrs[: self._committed]
        self._probs = self._probs[: self._committed]
        for b in _encode_text(text):
            self._chars.append(b & 0xFF)
            self._attrs.append(attrs)
            self._probs.append(prob)
        self._trim()

    def commit_partial(self, prefix: str = "", attrs: int = 0) -> None:
        partial_chars = self._chars[self._committed :]
        partial_probs = self._probs[self._committed :]

        self._chars = self._chars[: self._committed]
        self._attrs = self._attrs[: self._committed]
        self._probs = self._probs[: self._committed]

        for b in _encode_text(prefix):
            self._chars.append(b & 0xFF)
            self._attrs.append(attrs | ATTR_SOURCE_TAG)
            self._probs.append(1.0)
        for b, p in zip(partial_chars, partial_probs, strict=True):
            self._chars.append(b)
            self._attrs.append(attrs & ~ATTR_PARTIAL)
            self._probs.append(p)
        self._trim()
        self._committed = len(self._chars)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chars = np.zeros(self.capacity, dtype=np.uint32)
        attrs = np.zeros(self.capacity, dtype=np.uint32)
        probs = np.zeros(self.capacity, dtype=np.float32)
        n = len(self._chars)
        if n:
            chars[:n] = self._chars
            attrs[:n] = self._attrs
            probs[:n] = self._probs
        return chars, attrs, probs

    def _trim(self) -> None:
        excess = len(self._chars) - self.capacity
        if excess > 0:
            self._chars = self._chars[excess:]
            self._attrs = self._attrs[excess:]
            self._probs = self._probs[excess:]
            self._committed = max(0, self._committed - excess)
