"""Daily-rebirth helpers: rotate audience log, reset model, schedule."""

from __future__ import annotations

import datetime as dt
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

from brainscan.model import GPT

log = logging.getLogger(__name__)


class RebirthPhase(Enum):
    IDLE = "idle"
    FADING_OUT = "fading_out"
    FADING_IN = "fading_in"


@dataclass
class RebirthFade:
    phase: RebirthPhase = RebirthPhase.IDLE
    started_at: float = 0.0


def step_rebirth_phase(
    state: RebirthFade,
    now_t: float,
    is_due: bool,
    fade_duration: float = 2.0,
) -> tuple[RebirthFade, float, bool]:
    """Advance the rebirth fade state machine.

    Returns: (new_state, global_brightness, should_perform_rebirth_now)

    Phases:
        IDLE        — global_brightness=1.0; transitions to FADING_OUT when is_due.
        FADING_OUT  — brightness ramps 1.0→0.0 over fade_duration; at end transitions to
                       FADING_IN and signals should_perform_rebirth_now=True for one tick.
        FADING_IN   — brightness ramps 0.0→1.0 over fade_duration; at end → IDLE.
    """
    if state.phase == RebirthPhase.IDLE:
        if is_due:
            return (
                RebirthFade(phase=RebirthPhase.FADING_OUT, started_at=now_t),
                1.0,
                False,
            )
        return state, 1.0, False

    elapsed = now_t - state.started_at

    if state.phase == RebirthPhase.FADING_OUT:
        brightness = max(0.0, 1.0 - elapsed / fade_duration)
        if elapsed >= fade_duration:
            return (
                RebirthFade(phase=RebirthPhase.FADING_IN, started_at=now_t),
                0.0,
                True,
            )
        return state, brightness, False

    if state.phase == RebirthPhase.FADING_IN:
        brightness = min(1.0, elapsed / fade_duration)
        if elapsed >= fade_duration:
            return RebirthFade(), 1.0, False
        return state, brightness, False

    return state, 1.0, False  # unreachable


def rotate_audience_log(
    source: Path, target_dir: Path, date: dt.date
) -> Path | None:
    if not source.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{date.isoformat()}.txt"
    source.replace(target)
    return target


@dataclass
class RebirthResult:
    corpus: bytes
    seed: int


def rebirth(
    model: GPT,
    seed_corpus: bytes,
    audience_dir: Path,
    persist_days: int,
    seed: int,
) -> RebirthResult:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.apply(model._init_weights)

    persisted = _load_recent_audience(audience_dir, persist_days)
    corpus = persisted + seed_corpus
    return RebirthResult(corpus=corpus, seed=seed)


def _load_recent_audience(audience_dir: Path, persist_days: int) -> bytes:
    if persist_days <= 0 or not audience_dir.exists():
        return b""
    files = sorted(audience_dir.glob("*.txt"), reverse=True)[:persist_days]
    files.reverse()
    return b"".join(f.read_bytes() for f in files)


@dataclass
class RebirthScheduler:
    at_hh_mm: str | None
    state_path: Path | None = None
    _last_fired_date: dt.date | None = field(default=None, init=False, repr=False)
    _hour: int = field(default=0, init=False, repr=False)
    _minute: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.at_hh_mm is not None:
            try:
                h, m = self.at_hh_mm.split(":", 1)
                self._hour = int(h)
                self._minute = int(m)
                if not (0 <= self._hour < 24 and 0 <= self._minute < 60):
                    raise ValueError
            except ValueError as e:
                raise ValueError(
                    f"--rebirth-at must be HH:MM, got {self.at_hh_mm!r}"
                ) from e

        if self.state_path is not None and self.state_path.exists():
            try:
                stored = self.state_path.read_text().strip()
                self._last_fired_date = dt.date.fromisoformat(stored)
            except (ValueError, OSError):
                pass

    def due(self, now: dt.datetime) -> bool:
        if self.at_hh_mm is None:
            return False
        target = now.replace(
            hour=self._hour, minute=self._minute, second=0, microsecond=0
        )
        if now < target:
            return False
        if self._last_fired_date == now.date():
            return False
        return True

    def mark_fired(self, when: dt.datetime) -> None:
        self._last_fired_date = when.date()
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(when.date().isoformat())
