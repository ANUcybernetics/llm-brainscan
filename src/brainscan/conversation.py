"""Pure conversational state machine.

Drives the dual-lane text strip from a four-state model:
MUSE (slow self-talk), LISTENING (partial transcription visible),
RESPONDING (fast generation seeded from audience input), then back to MUSE
after a cooldown. The driver is pure: ``step()`` returns events the caller
should act on (TTS, training-corpus append). All I/O lives outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from brainscan import tuning
from brainscan.lanes import (
    ATTR_PARTIAL,
    ATTR_SOURCE_TAG,
    LaneBuffer,
)


class ConversationState(Enum):
    MUSE = "muse"
    LISTENING = "listening"
    RESPONDING = "responding"


@dataclass
class ListenerSnapshot:
    committed: list[str] = field(default_factory=list)
    partial: str | None = None
    in_speech: bool = False


@dataclass(frozen=True)
class SpeakEvent:
    text: str


@dataclass
class StepEvents:
    token_count: int = 0
    speak_events: list[SpeakEvent] = field(default_factory=list)
    new_corpus_text: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    muse_token_interval: float = tuning.MUSE_TOKEN_INTERVAL
    listening_token_interval: float = tuning.LISTENING_TOKEN_INTERVAL
    response_token_interval: float = tuning.RESPONDING_TOKEN_INTERVAL
    response_token_count: int = tuning.RESPONSE_TOKEN_COUNT
    cooldown_seconds: float = tuning.COOLDOWN_SECONDS
    tts_enabled: bool = False
    source_tag: str = "> mic > "

    state: ConversationState = ConversationState.MUSE
    audience: LaneBuffer = field(default_factory=LaneBuffer)
    model_lane: LaneBuffer = field(default_factory=LaneBuffer)

    _last_token_t: float = -1.0
    _response_remaining: int = 0
    _response_text: list[int] = field(default_factory=list)
    _cooldown_until: float = -1.0

    def step(
        self,
        now: float,
        listener: ListenerSnapshot,
        token_fn: Callable[[float], tuple[int, float]],
    ) -> StepEvents:
        events = StepEvents()

        for text in listener.committed:
            events.new_corpus_text.append(text)

        self._maybe_enter_listening(now, listener)
        self._maybe_handle_partial(listener)
        self._maybe_enter_responding(now, listener)

        token_count = self._tick_generation(now, token_fn, events)
        events.token_count = token_count
        return events

    def _maybe_enter_listening(self, now: float, listener: ListenerSnapshot) -> None:
        if self.state == ConversationState.MUSE and listener.in_speech:
            if now >= self._cooldown_until:
                self.state = ConversationState.LISTENING

    def _maybe_handle_partial(self, listener: ListenerSnapshot) -> None:
        if self.state != ConversationState.LISTENING:
            return
        if listener.partial is not None:
            self.audience.replace_tail(
                listener.partial, prob=1.0, attrs=ATTR_PARTIAL
            )

    def _maybe_enter_responding(self, now: float, listener: ListenerSnapshot) -> None:
        if not listener.committed:
            return
        self.audience.commit_partial(prefix=self.source_tag, attrs=ATTR_SOURCE_TAG)
        self.state = ConversationState.RESPONDING
        self._last_token_t = -1.0
        self._response_remaining = self.response_token_count
        self._response_text = []

    def _tick_generation(
        self,
        now: float,
        token_fn: Callable[[float], tuple[int, float]],
        events: StepEvents,
    ) -> int:
        MAX_CATCHUP = 8
        emitted = 0
        while emitted < MAX_CATCHUP:
            interval = self._current_interval()
            if self._last_token_t < 0.0:
                should_emit = True
            else:
                should_emit = (now - self._last_token_t) >= interval

            if not should_emit:
                break

            tok, prob = token_fn(now)
            self.model_lane.push(tok, prob=prob)
            if self._last_token_t < 0.0:
                self._last_token_t = now
            else:
                self._last_token_t += interval

            emitted += 1

            if self.state == ConversationState.RESPONDING:
                self._response_text.append(tok)
                self._response_remaining -= 1
                if self._response_remaining <= 0:
                    text = bytes(self._response_text).decode(
                        "utf-8", errors="replace"
                    )
                    if self.tts_enabled and text.strip():
                        events.speak_events.append(SpeakEvent(text=text))
                    self.state = ConversationState.MUSE
                    self._cooldown_until = now + self.cooldown_seconds
                    self._response_text = []
                    break  # state changed; stop catching up under old interval
        return emitted

    def _current_interval(self) -> float:
        if self.state == ConversationState.LISTENING:
            return self.listening_token_interval
        if self.state == ConversationState.RESPONDING:
            return self.response_token_interval
        return self.muse_token_interval
