"""Integration tests that drive train.main() with a fake listener."""
import sys
from dataclasses import dataclass, field
from typing import Callable

from brainscan import train


@dataclass
class ScriptedListener:
    """Fake SpeechListener: emits scripted partial/commit events at specific iterations."""

    script: list[tuple[int, str, str]] = field(default_factory=list)
    """List of (iteration_no, action, payload). action is 'partial' or 'commit'."""

    _iter: int = 0
    _committed: list[str] = field(default_factory=list)
    partial_callback: Callable[[str], None] | None = None
    speech_end_callback: Callable[[], None] | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_text(self) -> list[str]:
        # Process script entries scheduled for this iteration
        for it, action, payload in self.script:
            if it == self._iter:
                if action == "partial" and self.partial_callback is not None:
                    self.partial_callback(payload)
                elif action == "commit":
                    self._committed.append(payload)
                    if self.speech_end_callback is not None:
                        self.speech_end_callback()
        self._iter += 1
        items = list(self._committed)
        self._committed.clear()
        return items


def test_listening_then_responding_transition(tmp_path, monkeypatch):
    """Drive train.main() through a partial → commit sequence and verify
    the conversation enters LISTENING and RESPONDING, the corpus receives
    the committed text, and at least one frame is written."""

    fake = ScriptedListener(script=[
        (1, "partial", "hel"),
        (2, "partial", "hello"),
        (3, "commit", "hello"),
    ])

    state_history: list = []
    monkeypatch.setattr(train, "_build_listener", lambda args: fake)
    monkeypatch.setattr(train, "_train_state_history", state_history)

    args = [
        "train",
        "--steps", "8",
        "--snapshot-every", "1",
        "--n-layer", "1",
        "--n-head", "1",
        "--n-embd", "16",
        "--sequence-len", "16",
        "--batch-size", "2",
        "--gen-tokens", "4",
        "--save-images",
        "--output-dir", str(tmp_path),
        "--data", str(tmp_path / "tiny.txt"),
    ]
    (tmp_path / "tiny.txt").write_bytes(b"abcdefghij" * 200)

    monkeypatch.setattr(sys, "argv", args)
    train.main()

    from brainscan.conversation import ConversationState
    assert ConversationState.LISTENING in state_history, (
        f"never entered LISTENING; states: {state_history}"
    )
    assert ConversationState.RESPONDING in state_history, (
        f"never entered RESPONDING; states: {state_history}"
    )

    audience_log = tmp_path / "audience_input.txt"
    assert audience_log.exists()
    assert b"hello" in audience_log.read_bytes()

    frames = sorted((tmp_path / "frames").glob("*.png"))
    assert len(frames) >= 1
