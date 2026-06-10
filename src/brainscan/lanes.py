"""Circular character buffer for the dual-lane text strip.

Each lane holds chars + per-char attribute bits (partial / source-tag) + a
per-char probability (used by the model lane for brightness; harmless for the
audience lane). ``snapshot()`` returns three numpy arrays already shaped for
upload to a wgpu storage buffer in display order (oldest at index 0).

A lane can also act as a wall-clock conveyor: ``advance()`` drifts the
display offset left over time (dropping chars that have fully rolled off)
and ``pad_to_now()`` inserts committed spaces so the next char lands at the
right edge --- together they turn the lane into a timeline where silence
reads as a gap proportional to its duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATTR_PARTIAL = 1 << 0
ATTR_SOURCE_TAG = 1 << 1

# Display width of one lane cell in canvas px (the 32x64 lane glyph; see
# font.LANE_GLYPH_W). Used to convert between drift px and char columns.
CELL_W = 32


def _encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


@dataclass
class LaneBuffer:
    capacity: int = 240
    _chars: list[int] = field(default_factory=list, init=False, repr=False)
    _attrs: list[int] = field(default_factory=list, init=False, repr=False)
    _probs: list[float] = field(default_factory=list, init=False, repr=False)
    _committed: int = field(default=0, init=False, repr=False)
    offset_px: float = field(default=0.0, init=False)

    @property
    def count(self) -> int:
        return len(self._chars)

    def advance(self, dt_seconds: float, px_per_second: float) -> None:
        """Drift the lane left by wall-clock time (conveyor motion).

        Chars that have fully rolled off the left edge are dropped and the
        offset rebased so it stays within one cell --- a display-invariant
        change (char ``i`` at screen px ``i * CELL_W - offset_px``). An
        empty lane resets to offset 0.
        """
        if dt_seconds <= 0.0 or px_per_second <= 0.0 or not self._chars:
            return
        self.offset_px += dt_seconds * px_per_second
        drop = min(int(self.offset_px // CELL_W), len(self._chars))
        if drop:
            self._chars = self._chars[drop:]
            self._attrs = self._attrs[drop:]
            self._probs = self._probs[drop:]
            self._committed = max(0, self._committed - drop)
            self.offset_px -= drop * CELL_W
        if not self._chars:
            self.offset_px = 0.0

    def pad_to_now(self) -> None:
        """Insert committed spaces so the next char lands at the right edge.

        With ``offset_px`` rebased below one cell by ``advance()``, the
        right-edge slot is index ``capacity - 1``; the gap left by however
        far the old text has drifted is filled with blank committed cells,
        making silence literally visible as empty timeline. No-op while the
        lane is already full to the right edge or a partial tail is active
        (padding belongs at utterance start, before any partial text).
        """
        if self._committed < len(self._chars):
            return
        while len(self._chars) < self.capacity - 1:
            self._chars.append(0x20)
            self._attrs.append(0)
            self._probs.append(1.0)
        self._committed = len(self._chars)

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
        """Promote the partial tail to committed; prefix gets ATTR_SOURCE_TAG, body has ATTR_PARTIAL cleared. ``attrs`` is applied to both."""
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
