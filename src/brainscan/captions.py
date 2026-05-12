"""Pure caption composition for the 12px footer band."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAPTIONS_COLS = 480
LEFT_ZONE = CAPTIONS_COLS // 2 - 1
RIGHT_ZONE = CAPTIONS_COLS // 2 - 1


@dataclass(frozen=True)
class CaptionsState:
    state_label: str = ""
    cursor_label: str = ""
    event_line: str = ""


def compose_caption(s: CaptionsState) -> np.ndarray:
    row = bytearray(b" " * CAPTIONS_COLS)

    left = s.state_label.encode("ascii", errors="replace")[:LEFT_ZONE]
    row[: len(left)] = left

    right = s.cursor_label.encode("ascii", errors="replace")[:RIGHT_ZONE]
    if right:
        row[CAPTIONS_COLS - len(right) :] = right

    if s.event_line:
        event = s.event_line.encode("ascii", errors="replace")
        if len(event) <= CAPTIONS_COLS // 4:
            start = (CAPTIONS_COLS - len(event)) // 2
            row[start : start + len(event)] = event

    return np.frombuffer(bytes(row), dtype=np.uint8).astype(np.uint32)
