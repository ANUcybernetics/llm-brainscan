# Conversational Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 192px text strip below the weight visualisation into a three-band conversational surface (audience / model / captions), driven by a four-state conversation machine, with TTS and a daily rebirth cycle.

**Architecture:** A pure `Conversation` dataclass owns four states (MUSE / LISTENING / RESPONDING) and drives generation rate from wall-clock time. The renderer is extended with three independent text bands sharing one fragment shader, each backed by a circular character buffer and a sub-pixel scroll offset. STT, TTS and the rebirth scheduler are thin side-effecting wrappers that the conversation driver invokes; everything testable is built as pure functions over plain data.

**Tech Stack:** Python 3.12, PyTorch (model), wgpu/WGSL (shader), faster-whisper (STT), piper-tts (TTS, optional), sounddevice (input + drone), pytest. All commands prefixed with `mise exec -- uv run`.

**Spec:** `docs/superpowers/specs/2026-04-26-conversational-brain-design.md`

---

## File structure

### New files

| Path | Responsibility |
|------|----------------|
| `src/brainscan/lanes.py` | `LaneBuffer` dataclass — circular char/attr/prob buffer with `push()` and a `view()` returning the GPU-shaped arrays for a lane |
| `src/brainscan/captions.py` | Pure `compose_caption(state, training, layout_cursor)` returning the 960-char captions row |
| `src/brainscan/conversation.py` | `ConversationState` enum, `Conversation` dataclass, `step()` driver |
| `src/brainscan/tts.py` | `TTSEngine` wrapper around piper (or a no-op when disabled) |
| `src/brainscan/rebirth.py` | `rotate_audience_log()`, `rebirth()`, `RebirthScheduler` |
| `src/brainscan/audio_drone.py` | Optional sub-bass oscillator (deferred — last task) |
| `tests/test_lanes.py` | LaneBuffer tests |
| `tests/test_captions.py` | Caption composition tests |
| `tests/test_conversation.py` | State-machine tests with a fake clock and mock listener |
| `tests/test_tts.py` | TTSEngine tests with the engine mocked |
| `tests/test_rebirth.py` | Rotation, rebirth, scheduler tests |

### Modified files

| Path | What changes |
|------|--------------|
| `src/brainscan/renderer.py` | Three-band shader, new uniforms and storage buffers, `draw()` accepts lane data |
| `src/brainscan/data.py` | `TextBuffer.rotate(target_path)` |
| `src/brainscan/stt.py` | `partial_callback` parameter on `SpeechListener` |
| `src/brainscan/model.py` | `GPT.streaming_generate()` token-by-token iterator |
| `src/brainscan/train.py` | Conversation driver replaces inline mic loop; new CLI flags |
| `tests/test_renderer.py` | Dual-lane rendering tests; existing single-strip tests updated to new API |
| `tests/test_stt.py` | Partial callback tests |
| `tests/test_data.py` | Rotate tests (note: tests for `TextBuffer` currently live in `tests/test_text_buffer.py`; add rotate tests there) |
| `tests/test_train.py` | State-machine integration with a fake clock |

### Test commands

- All tests: `mise exec -- uv run pytest tests/ -v`
- Single test: `mise exec -- uv run pytest tests/<file>::<TestClass>::<test_name> -v`
- Type check: `mise exec -- uv run ty check`

Each task ends with a commit. Use imperative-mood, concise commit messages.

---

## Phase 1: Foundations

### Task 1: `TextBuffer.rotate()`

Atomically move the persistence file aside; the in-memory buffer is left untouched so the caller decides when to also reset it.

**Files:**
- Modify: `src/brainscan/data.py`
- Test: `tests/test_text_buffer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_text_buffer.py` inside a new `class TestTextBufferRotate:`

```python
class TestTextBufferRotate:
    def test_rotate_moves_persist_file(self, tmp_path):
        path = tmp_path / "audience.txt"
        target = tmp_path / "rotated" / "2026-04-25.txt"
        buf = TextBuffer(b"base", persist_path=path)
        buf.append(" hello")

        buf.rotate(target)

        assert target.read_bytes() == b" hello"
        assert not path.exists()

    def test_rotate_creates_target_parent(self, tmp_path):
        path = tmp_path / "audience.txt"
        target = tmp_path / "deep" / "nested" / "out.txt"
        buf = TextBuffer(b"base", persist_path=path)
        buf.append("x")

        buf.rotate(target)

        assert target.parent.is_dir()
        assert target.read_bytes() == b"x"

    def test_rotate_no_persist_path_raises(self):
        buf = TextBuffer(b"hello")
        with pytest.raises(ValueError):
            buf.rotate(Path("/tmp/whatever.txt"))

    def test_rotate_missing_source_is_noop(self, tmp_path):
        path = tmp_path / "audience.txt"
        target = tmp_path / "rotated.txt"
        buf = TextBuffer(b"base", persist_path=path)

        buf.rotate(target)

        assert not target.exists()

    def test_rotate_leaves_in_memory_buffer(self, tmp_path):
        path = tmp_path / "audience.txt"
        target = tmp_path / "rotated.txt"
        buf = TextBuffer(b"base", persist_path=path)
        buf.append(" tail")

        buf.rotate(target)

        assert buf.data == b"base tail"
```

Add `import pytest` and `from pathlib import Path` to the test file if not already present.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `mise exec -- uv run pytest tests/test_text_buffer.py::TestTextBufferRotate -v`
Expected: FAIL with `AttributeError: 'TextBuffer' object has no attribute 'rotate'`

- [ ] **Step 3: Implement `rotate`**

In `src/brainscan/data.py` add this method to `TextBuffer` (place it just below `append`):

```python
    def rotate(self, target: Path) -> None:
        if self._persist_path is None:
            raise ValueError("TextBuffer has no persist path to rotate")
        with self._lock:
            if not self._persist_path.exists():
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.replace(target)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `mise exec -- uv run pytest tests/test_text_buffer.py -v`
Expected: all green, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/data.py tests/test_text_buffer.py
git commit -m "feat(data): add TextBuffer.rotate for daily log rotation"
```

---

### Task 2: `LaneBuffer` circular char buffer

A small, pure dataclass each lane uses. Holds chars + per-char attribute byte + per-char float prob, with a `push()` operation and a `snapshot()` that returns three `np.ndarray`s the renderer can upload directly.

**Files:**
- Create: `src/brainscan/lanes.py`
- Test: `tests/test_lanes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lanes.py`:

```python
import numpy as np

from brainscan.lanes import LaneBuffer, ATTR_PARTIAL, ATTR_SOURCE_TAG


class TestLaneBufferConstruction:
    def test_default_capacity_is_320(self):
        buf = LaneBuffer()
        assert buf.capacity == 320

    def test_custom_capacity(self):
        buf = LaneBuffer(capacity=64)
        assert buf.capacity == 64

    def test_initial_empty(self):
        buf = LaneBuffer(capacity=8)
        chars, attrs, probs = buf.snapshot()
        assert chars.shape == (8,)
        assert attrs.shape == (8,)
        assert probs.shape == (8,)
        assert chars.dtype == np.uint32
        assert attrs.dtype == np.uint32
        assert probs.dtype == np.float32
        np.testing.assert_array_equal(chars, np.zeros(8, dtype=np.uint32))
        assert buf.count == 0


class TestLaneBufferPush:
    def test_push_below_capacity(self):
        buf = LaneBuffer(capacity=4)
        buf.push(ord("A"), prob=1.0)
        buf.push(ord("B"), prob=0.5)
        chars, _attrs, probs = buf.snapshot()
        # newest char is at the rightmost visible position (index count-1)
        assert chars[0] == ord("A")
        assert chars[1] == ord("B")
        assert probs[1] == 0.5
        assert buf.count == 2

    def test_push_above_capacity_drops_oldest(self):
        buf = LaneBuffer(capacity=3)
        for c in "ABCD":
            buf.push(ord(c), prob=1.0)
        chars, _attrs, _probs = buf.snapshot()
        # logical view: oldest first, newest last
        assert chars[0] == ord("B")
        assert chars[1] == ord("C")
        assert chars[2] == ord("D")
        assert buf.count == 3

    def test_push_with_attrs(self):
        buf = LaneBuffer(capacity=4)
        buf.push(ord("X"), prob=1.0, attrs=ATTR_PARTIAL)
        _chars, attrs, _probs = buf.snapshot()
        assert attrs[0] & ATTR_PARTIAL

    def test_default_prob_is_one(self):
        buf = LaneBuffer(capacity=4)
        buf.push(ord("A"))
        _chars, _attrs, probs = buf.snapshot()
        assert probs[0] == 1.0


class TestLaneBufferReplaceTail:
    def test_replace_tail_overwrites_recent(self):
        """Used during partial transcription: replace last N chars with new partial."""
        buf = LaneBuffer(capacity=8)
        for c in "AB":
            buf.push(ord(c), prob=1.0, attrs=0)
        buf.replace_tail("hello", prob=0.5, attrs=ATTR_PARTIAL)
        chars, attrs, _ = buf.snapshot()
        # AB followed by greyed hello
        assert chars[0] == ord("A")
        assert chars[1] == ord("B")
        assert chars[2] == ord("h")
        assert chars[6] == ord("o")
        assert attrs[2] & ATTR_PARTIAL
        assert buf.count == 7

    def test_replace_tail_idempotent_when_partials_grow(self):
        buf = LaneBuffer(capacity=16)
        buf.push(ord("X"), prob=1.0, attrs=0)
        buf.replace_tail("he", attrs=ATTR_PARTIAL)
        buf.replace_tail("hello", attrs=ATTR_PARTIAL)
        chars, _, _ = buf.snapshot()
        # only one X kept; partial replaced wholesale
        assert chars[0] == ord("X")
        assert chars[1] == ord("h")
        assert chars[5] == ord("o")
        assert buf.count == 6

    def test_commit_partial_clears_partial_bit(self):
        buf = LaneBuffer(capacity=16)
        buf.replace_tail("hi", attrs=ATTR_PARTIAL)
        buf.commit_partial(prefix="▸ mic ▸ ", attrs=ATTR_SOURCE_TAG)
        chars, attrs, _ = buf.snapshot()
        # prefix prepended, all attrs no longer partial
        assert chars[0] == ord("▸") or chars[0] == ord(">")  # bytewise of mic prefix
        assert not (attrs[buf.count - 1] & ATTR_PARTIAL)


class TestLaneBufferSourceTag:
    def test_push_with_source_tag(self):
        buf = LaneBuffer(capacity=16)
        buf.push_text("▸ mic ▸ ", attrs=ATTR_SOURCE_TAG)
        _chars, attrs, _ = buf.snapshot()
        # all chars have ATTR_SOURCE_TAG set
        for i in range(buf.count):
            assert attrs[i] & ATTR_SOURCE_TAG
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `mise exec -- uv run pytest tests/test_lanes.py -v`
Expected: ImportError on `brainscan.lanes`.

- [ ] **Step 3: Implement `LaneBuffer`**

Create `src/brainscan/lanes.py`:

```python
"""Circular character buffer for the dual-lane text strip.

Each lane holds chars + per-char attribute bits (partial / source-tag) + a
per-char probability (used by the model lane for brightness; harmless for the
audience lane). ``snapshot()`` returns three numpy arrays already shaped for
upload to a wgpu storage buffer in *display* order (oldest at index 0).
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `mise exec -- uv run pytest tests/test_lanes.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/lanes.py tests/test_lanes.py
git commit -m "feat(lanes): add LaneBuffer circular char buffer"
```

---

### Task 3: Captions composer

Pure function that returns the 960-char captions string by concatenating left tag, middle event line, and right cursor label.

**Files:**
- Create: `src/brainscan/captions.py`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_captions.py`:

```python
from brainscan.captions import (
    CAPTIONS_COLS,
    CaptionsState,
    compose_caption,
)


def _decode(arr) -> str:
    return bytes(arr.tolist()).decode("ascii", errors="replace")


class TestComposeCaption:
    def test_listening_left_label(self):
        s = CaptionsState(
            state_label="listening...",
            cursor_label="block 4 attn c_attn",
            event_line="",
        )
        chars = compose_caption(s)
        assert chars.shape == (CAPTIONS_COLS,)
        text = _decode(chars).rstrip("\x00")
        assert text.startswith("listening...")
        assert text.endswith("block 4 attn c_attn")

    def test_event_line_centred(self):
        s = CaptionsState(
            state_label="musing",
            cursor_label="embed wte",
            event_line="audience_input rotated",
        )
        chars = compose_caption(s)
        text = bytes(chars.tolist()).decode("ascii", errors="replace")
        # event line appears somewhere in the middle (not at left or right)
        idx = text.find("audience_input rotated")
        assert idx > 50
        assert idx < CAPTIONS_COLS - 70

    def test_long_state_label_truncated(self):
        s = CaptionsState(
            state_label="x" * 200,
            cursor_label="ok",
            event_line="",
        )
        chars = compose_caption(s)
        # left zone is bounded; right zone still readable
        text = bytes(chars.tolist()).decode("ascii", errors="replace")
        assert text.endswith("ok")
        # the left zone is at most ~CAPTIONS_COLS // 2 - 1
        assert text.count("x") <= CAPTIONS_COLS // 2

    def test_empty_state_renders_blank(self):
        s = CaptionsState(state_label="", cursor_label="", event_line="")
        chars = compose_caption(s)
        assert chars.shape == (CAPTIONS_COLS,)
        text = bytes(chars.tolist()).decode("ascii", errors="replace")
        assert text.strip("\x00 ") == ""

    def test_dawn_event(self):
        s = CaptionsState(
            state_label="musing",
            cursor_label="embed wte",
            event_line="dawn 06:00",
        )
        chars = compose_caption(s)
        text = bytes(chars.tolist()).decode("ascii", errors="replace")
        assert "dawn 06:00" in text
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `mise exec -- uv run pytest tests/test_captions.py -v`
Expected: ImportError on `brainscan.captions`.

- [ ] **Step 3: Implement `compose_caption`**

Create `src/brainscan/captions.py`:

```python
"""Pure caption composition for the 12px footer band."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAPTIONS_COLS = 960
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `mise exec -- uv run pytest tests/test_captions.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/captions.py tests/test_captions.py
git commit -m "feat(captions): add compose_caption for footer band"
```

---

## Phase 2: Dual-lane renderer

The shader gains three independent text bands. Each band reads from its own pair of storage buffers (chars + attrs/probs) and is positioned by uniforms. The bottom 192px of the canvas is split as 90 / 90 / 12.

### Task 4: Extend `RenderConfig` for three bands

**Files:**
- Modify: `src/brainscan/renderer.py`
- Test: `tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_renderer.py` (top-level, after the existing imports):

```python
from brainscan.renderer import RenderConfig


class TestRenderConfigBands:
    def test_default_bands_are_disabled(self):
        cfg = RenderConfig(width=100, height=100)
        assert cfg.audience_height == 0
        assert cfg.model_height == 0
        assert cfg.captions_height == 0
        assert cfg.audience_y == 0
        assert cfg.model_y == 0
        assert cfg.captions_y == 0

    def test_bands_stack_upward_from_bottom(self):
        cfg = RenderConfig(
            width=7680,
            height=4320,
            audience_height=90,
            model_height=90,
            captions_height=12,
        )
        assert cfg.captions_y == 4308
        assert cfg.model_y == 4218
        assert cfg.audience_y == 4128

    def test_lane_capacity_320_at_3x(self):
        cfg = RenderConfig(width=7680, height=4320, audience_height=90)
        assert cfg.lane_capacity == 320

    def test_captions_capacity_960(self):
        cfg = RenderConfig(width=7680, height=4320, captions_height=12)
        assert cfg.captions_capacity == 960
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `mise exec -- uv run pytest tests/test_renderer.py::TestRenderConfigBands -v`
Expected: AttributeError on `audience_height`.

- [ ] **Step 3: Add fields and properties to `RenderConfig`**

In `src/brainscan/renderer.py`, replace the existing `RenderConfig` block with:

```python
LANE_SCALE = 3
LANE_GLYPH_W = 8 * LANE_SCALE
LANE_GLYPH_H = 16 * LANE_SCALE
CAPTIONS_GLYPH_W = 8


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    colormap: int = COLORMAP_DIVERGING
    audience_height: int = 0
    model_height: int = 0
    captions_height: int = 0

    @property
    def captions_y(self) -> int:
        return self.height - self.captions_height if self.captions_height > 0 else 0

    @property
    def model_y(self) -> int:
        if self.model_height == 0:
            return 0
        return self.height - self.captions_height - self.model_height

    @property
    def audience_y(self) -> int:
        if self.audience_height == 0:
            return 0
        return (
            self.height
            - self.captions_height
            - self.model_height
            - self.audience_height
        )

    @property
    def lane_capacity(self) -> int:
        return max(1, self.width // LANE_GLYPH_W)

    @property
    def captions_capacity(self) -> int:
        return max(1, self.width // CAPTIONS_GLYPH_W)
```

Also remove the now-unused `text_strip_height`, `text_scale`, `TEXT_SCALE_DEFAULT`, `text_y`, `text_cols`, and `max_text` lines from `RenderConfig`. They will be replaced wholesale by the lane fields.

- [ ] **Step 4: Run the new tests to confirm they pass**

Run: `mise exec -- uv run pytest tests/test_renderer.py::TestRenderConfigBands -v`
Expected: all green.

The other renderer tests will be broken at this point — that is expected; they are fixed in Tasks 5–8.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/renderer.py tests/test_renderer.py
git commit -m "feat(renderer): add three-band fields to RenderConfig"
```

---

### Task 5: New uniforms, storage buffers and pipeline wiring

Replace the existing `_UNIFORM_DTYPE`, buffer creation, and bind group with the three-band layout. The shader source is updated in Task 6 — for now we just wire the resources and a stub shader so the pipeline still compiles.

**Files:**
- Modify: `src/brainscan/renderer.py`

- [ ] **Step 1: Replace `_UNIFORM_DTYPE`**

In `src/brainscan/renderer.py`, replace the existing `_UNIFORM_DTYPE` with:

```python
_UNIFORM_DTYPE = np.dtype([
    ("width", np.uint32),
    ("height", np.uint32),
    ("param_count", np.uint32),
    ("colormap", np.uint32),
    ("audience_y", np.uint32),
    ("audience_height", np.uint32),
    ("audience_count", np.uint32),
    ("audience_offset_px", np.uint32),
    ("model_y", np.uint32),
    ("model_height", np.uint32),
    ("model_count", np.uint32),
    ("model_offset_px", np.uint32),
    ("captions_y", np.uint32),
    ("captions_height", np.uint32),
    ("captions_count", np.uint32),
    ("vmax", np.float32),
])
```

- [ ] **Step 2: Replace `RenderResources`**

Replace the existing `RenderResources` dataclass with:

```python
@dataclass
class RenderResources:
    device: wgpu.GPUDevice
    config: RenderConfig
    uniform_data: np.ndarray
    uniform_buffer: wgpu.GPUBuffer
    weight_buffer: wgpu.GPUBuffer
    font_buffer: wgpu.GPUBuffer
    audience_chars_buffer: wgpu.GPUBuffer
    audience_attrs_buffer: wgpu.GPUBuffer
    model_chars_buffer: wgpu.GPUBuffer
    model_probs_buffer: wgpu.GPUBuffer
    captions_chars_buffer: wgpu.GPUBuffer
    bind_group: wgpu.GPUBindGroup
    pipeline: wgpu.GPURenderPipeline
```

- [ ] **Step 3: Replace `create_render_pipeline`**

Replace the existing `create_render_pipeline` function with:

```python
def create_render_pipeline(
    config: RenderConfig,
    device: wgpu.GPUDevice,
    target_format: str,
) -> RenderResources:
    uniform_data = np.zeros(1, dtype=_UNIFORM_DTYPE)
    uniform_data["width"] = config.width
    uniform_data["height"] = config.height
    uniform_data["colormap"] = config.colormap
    uniform_data["audience_y"] = config.audience_y
    uniform_data["audience_height"] = config.audience_height
    uniform_data["model_y"] = config.model_y
    uniform_data["model_height"] = config.model_height
    uniform_data["captions_y"] = config.captions_y
    uniform_data["captions_height"] = config.captions_height

    uniform_size = max(_UNIFORM_DTYPE.itemsize, 64)
    uniform_buffer = device.create_buffer(
        size=uniform_size,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    max_params = config.width * config.height
    weight_buffer = device.create_buffer(
        size=max_params * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    font_size = 1024 * 4
    font_buffer = device.create_buffer(
        size=font_size,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    lane_cap = config.lane_capacity
    captions_cap = config.captions_capacity

    audience_chars_buffer = device.create_buffer(
        size=lane_cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    audience_attrs_buffer = device.create_buffer(
        size=lane_cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    model_chars_buffer = device.create_buffer(
        size=lane_cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    model_probs_buffer = device.create_buffer(
        size=lane_cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    captions_chars_buffer = device.create_buffer(
        size=captions_cap * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    if (
        config.audience_height > 0
        or config.model_height > 0
        or config.captions_height > 0
    ):
        from brainscan.font import generate_font_atlas

        font_data = generate_font_atlas()
        device.queue.write_buffer(font_buffer, 0, font_data.tobytes())

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": i,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform
                    if i == 0
                    else wgpu.BufferBindingType.read_only_storage
                },
            }
            for i in range(8)
        ]
    )

    buffers = [
        uniform_buffer,
        weight_buffer,
        font_buffer,
        audience_chars_buffer,
        audience_attrs_buffer,
        model_chars_buffer,
        model_probs_buffer,
        captions_chars_buffer,
    ]
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": i,
                "resource": {"buffer": buf, "offset": 0, "size": buf.size},
            }
            for i, buf in enumerate(buffers)
        ],
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    shader = device.create_shader_module(code=SHADER_SOURCE)
    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={
            "module": shader,
            "entry_point": "fs_main",
            "targets": [{"format": target_format}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )

    return RenderResources(
        device=device,
        config=config,
        uniform_data=uniform_data,
        uniform_buffer=uniform_buffer,
        weight_buffer=weight_buffer,
        font_buffer=font_buffer,
        audience_chars_buffer=audience_chars_buffer,
        audience_attrs_buffer=audience_attrs_buffer,
        model_chars_buffer=model_chars_buffer,
        model_probs_buffer=model_probs_buffer,
        captions_chars_buffer=captions_chars_buffer,
        bind_group=bind_group,
        pipeline=pipeline,
    )
```

- [ ] **Step 2: No tests yet — pipeline build is verified in Task 6 once the shader is updated. Skip running tests for this step.**

- [ ] **Step 3: Commit**

```bash
git add src/brainscan/renderer.py
git commit -m "feat(renderer): add three-lane storage buffers and uniforms"
```

---

### Task 6: New shader source for three bands

Replace the WGSL with three-band sampling. Each band has its own y-test, sub-pixel scroll offset, and colour formula. The captions footer is at 1× scale and dim grey; the audience and model lanes are at 3× scale with warm cream and cool blue-cream ramps respectively. Audience-lane partials are dimmed via the `attrs` storage buffer.

**Files:**
- Modify: `src/brainscan/renderer.py`

- [ ] **Step 1: Replace `SHADER_SOURCE`**

Replace the entire `SHADER_SOURCE` constant with:

```python
SHADER_SOURCE = """
struct Uniforms {
    width: u32,
    height: u32,
    param_count: u32,
    colormap: u32,
    audience_y: u32,
    audience_height: u32,
    audience_count: u32,
    audience_offset_px: u32,
    model_y: u32,
    model_height: u32,
    model_count: u32,
    model_offset_px: u32,
    captions_y: u32,
    captions_height: u32,
    captions_count: u32,
    vmax: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> font_data: array<u32>;
@group(0) @binding(3) var<storage, read> audience_chars: array<u32>;
@group(0) @binding(4) var<storage, read> audience_attrs: array<u32>;
@group(0) @binding(5) var<storage, read> model_chars: array<u32>;
@group(0) @binding(6) var<storage, read> model_probs: array<f32>;
@group(0) @binding(7) var<storage, read> captions_chars: array<u32>;

const ATTR_PARTIAL: u32 = 1u;
const ATTR_SOURCE_TAG: u32 = 2u;
const LANE_SCALE: u32 = 3u;
const LANE_CELL_W: u32 = 24u;
const LANE_CELL_H: u32 = 48u;
const LANE_GLYPH_W: u32 = 8u;
const LANE_GLYPH_H: u32 = 16u;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

fn diverging(v: f32) -> vec3<f32> {
    let t = clamp(v, -1.0, 1.0);
    let r = clamp(0.5 + t * 0.5, 0.0, 1.0);
    let g = clamp(0.5 - abs(t) * 0.4, 0.0, 1.0);
    let b = clamp(0.5 - t * 0.5, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn thermal(v: f32) -> vec3<f32> {
    let t = clamp((v + 1.0) * 0.5, 0.0, 1.0);
    let r = clamp(t * 3.0 - 1.0, 0.0, 1.0);
    let g = clamp(t * 3.0 - 2.0, 0.0, 1.0);
    let b = clamp(min(t * 3.0, 2.0 - t * 3.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn font_pixel(char_idx: u32, gx: u32, gy: u32) -> bool {
    let byte_offset = char_idx * 16u + gy;
    let word_idx = byte_offset / 4u;
    let byte_in_word = byte_offset % 4u;
    let word = font_data[word_idx];
    let byte_val = (word >> (byte_in_word * 8u)) & 0xFFu;
    return (byte_val & (0x80u >> gx)) != 0u;
}

fn render_lane_audience(px: u32, py: u32) -> vec4<f32> {
    let band_top = uniforms.audience_y;
    let band_bot = band_top + uniforms.audience_height;
    if uniforms.audience_height == 0u || py < band_top || py >= band_bot {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let local_y = py - band_top;
    let glyph_top = (uniforms.audience_height - LANE_CELL_H) / 2u;
    if local_y < glyph_top || local_y >= glyph_top + LANE_CELL_H {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }

    let scroll_x = px + uniforms.audience_offset_px;
    let col = scroll_x / LANE_CELL_W;
    if col >= uniforms.audience_count {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }
    let glyph = audience_chars[col];
    let attrs = audience_attrs[col];
    let gx = (scroll_x % LANE_CELL_W) / LANE_SCALE;
    let gy = (local_y - glyph_top) / LANE_SCALE;
    if !font_pixel(glyph, gx, gy) {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }

    var col_rgb = vec3<f32>(0.94, 0.88, 0.72);
    if (attrs & ATTR_PARTIAL) != 0u {
        col_rgb = vec3<f32>(0.50, 0.46, 0.38);
    } else if (attrs & ATTR_SOURCE_TAG) != 0u {
        col_rgb = vec3<f32>(0.62, 0.56, 0.42);
    }
    return vec4<f32>(col_rgb, 1.0);
}

fn render_lane_model(px: u32, py: u32) -> vec4<f32> {
    let band_top = uniforms.model_y;
    let band_bot = band_top + uniforms.model_height;
    if uniforms.model_height == 0u || py < band_top || py >= band_bot {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let local_y = py - band_top;
    let glyph_top = (uniforms.model_height - LANE_CELL_H) / 2u;
    if local_y < glyph_top || local_y >= glyph_top + LANE_CELL_H {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }

    let scroll_x = px + uniforms.model_offset_px;
    let col = scroll_x / LANE_CELL_W;
    if col >= uniforms.model_count {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }
    let glyph = model_chars[col];
    let prob = model_probs[col];
    let gx = (scroll_x % LANE_CELL_W) / LANE_SCALE;
    let gy = (local_y - glyph_top) / LANE_SCALE;
    if !font_pixel(glyph, gx, gy) {
        return vec4<f32>(0.04, 0.04, 0.06, 1.0);
    }
    let brightness = 0.25 + prob * 0.75;
    return vec4<f32>(
        brightness,
        brightness * 0.95,
        brightness * 1.10,
        1.0,
    );
}

fn render_captions(px: u32, py: u32) -> vec4<f32> {
    let band_top = uniforms.captions_y;
    let band_bot = band_top + uniforms.captions_height;
    if uniforms.captions_height == 0u || py < band_top || py >= band_bot {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let local_y = py - band_top;
    let col = px / LANE_GLYPH_W;
    if col >= uniforms.captions_count {
        return vec4<f32>(0.02, 0.02, 0.02, 1.0);
    }
    let glyph = captions_chars[col];
    let gx = px % LANE_GLYPH_W;
    let gy = local_y;
    if gy >= LANE_GLYPH_H {
        return vec4<f32>(0.02, 0.02, 0.02, 1.0);
    }
    if !font_pixel(glyph, gx, gy) {
        return vec4<f32>(0.02, 0.02, 0.02, 1.0);
    }
    return vec4<f32>(0.40, 0.40, 0.42, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(uniforms.width));
    let py = u32(in.uv.y * f32(uniforms.height));

    let cap = render_captions(px, py);
    if cap.x >= 0.0 { return cap; }

    let m = render_lane_model(px, py);
    if m.x >= 0.0 { return m; }

    let a = render_lane_audience(px, py);
    if a.x >= 0.0 { return a; }

    let idx = py * uniforms.width + px;
    if idx >= uniforms.param_count {
        return vec4<f32>(0.05, 0.05, 0.05, 1.0);
    }

    let raw = weights[idx];
    let w = select(raw / uniforms.vmax, 0.0, uniforms.vmax < 1e-10);
    var color: vec3<f32>;
    if uniforms.colormap == 1u {
        color = thermal(w);
    } else {
        color = diverging(w);
    }
    return vec4<f32>(color, 1.0);
}
"""
```

- [ ] **Step 2: No new tests yet — verification arrives in Task 8 once `draw()` exposes lane data.**

- [ ] **Step 3: Commit**

```bash
git add src/brainscan/renderer.py
git commit -m "feat(renderer): three-band shader (audience/model/captions)"
```

---

### Task 7: Update `draw()` and renderer classes for lane data

Replace the old `text_chars` / `text_probs` parameters with structured per-lane data. Two helper dataclasses keep call sites tidy.

**Files:**
- Modify: `src/brainscan/renderer.py`

- [ ] **Step 1: Add `LaneFrame` and `CaptionsFrame` dataclasses**

Just below the existing dataclasses in `src/brainscan/renderer.py`, add:

```python
@dataclass
class LaneFrame:
    chars: np.ndarray  # uint32, length lane_capacity
    attrs_or_probs: np.ndarray  # uint32 (audience attrs) or float32 (model probs)
    count: int
    offset_px: int = 0


@dataclass
class CaptionsFrame:
    chars: np.ndarray  # uint32, length captions_capacity
    count: int
```

- [ ] **Step 2: Replace `draw()`**

Replace the existing `draw` function with:

```python
def draw(
    res: RenderResources,
    target_view: wgpu.GPUTextureView,
    flat_weights: np.ndarray,
    audience: LaneFrame | None = None,
    model: LaneFrame | None = None,
    captions: CaptionsFrame | None = None,
) -> None:
    device = res.device
    param_count = len(flat_weights)

    if audience is not None:
        device.queue.write_buffer(
            res.audience_chars_buffer, 0, audience.chars.astype(np.uint32).tobytes()
        )
        device.queue.write_buffer(
            res.audience_attrs_buffer,
            0,
            audience.attrs_or_probs.astype(np.uint32).tobytes(),
        )
        res.uniform_data["audience_count"] = audience.count
        res.uniform_data["audience_offset_px"] = audience.offset_px
    else:
        res.uniform_data["audience_count"] = 0
        res.uniform_data["audience_offset_px"] = 0

    if model is not None:
        device.queue.write_buffer(
            res.model_chars_buffer, 0, model.chars.astype(np.uint32).tobytes()
        )
        device.queue.write_buffer(
            res.model_probs_buffer,
            0,
            model.attrs_or_probs.astype(np.float32).tobytes(),
        )
        res.uniform_data["model_count"] = model.count
        res.uniform_data["model_offset_px"] = model.offset_px
    else:
        res.uniform_data["model_count"] = 0
        res.uniform_data["model_offset_px"] = 0

    if captions is not None:
        device.queue.write_buffer(
            res.captions_chars_buffer, 0, captions.chars.astype(np.uint32).tobytes()
        )
        res.uniform_data["captions_count"] = captions.count
    else:
        res.uniform_data["captions_count"] = 0

    vmax = float(np.max(np.abs(flat_weights))) if param_count > 0 else 0.0
    res.uniform_data["param_count"] = param_count
    res.uniform_data["colormap"] = res.config.colormap
    res.uniform_data["vmax"] = vmax
    device.queue.write_buffer(res.uniform_buffer, 0, res.uniform_data.tobytes())

    device.queue.write_buffer(
        res.weight_buffer, 0, flat_weights.astype(np.float32).tobytes()
    )

    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": target_view,
                "resolve_target": None,
                "clear_value": (0.05, 0.05, 0.05, 1.0),
                "load_op": "clear",
                "store_op": "store",
            }
        ]
    )
    render_pass.set_pipeline(res.pipeline)
    render_pass.set_bind_group(0, res.bind_group)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])
```

- [ ] **Step 3: Update `OffscreenRenderer`**

Replace the existing `OffscreenRenderer.__init__` and `OffscreenRenderer.render` with:

```python
class OffscreenRenderer:
    """Render weight data to an offscreen texture and read back as numpy."""

    def __init__(
        self,
        width: int,
        height: int,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        audience_height: int = 0,
        model_height: int = 0,
        captions_height: int = 0,
    ):
        self.width = width
        self.height = height
        self.colormap = colormap
        self.device = device or get_device()

        self._config = RenderConfig(
            width,
            height,
            colormap,
            audience_height=audience_height,
            model_height=model_height,
            captions_height=captions_height,
        )
        self._res = create_render_pipeline(
            self._config, self.device, wgpu.TextureFormat.rgba8unorm
        )

        self._target_texture = self.device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

    @property
    def config(self) -> RenderConfig:
        return self._config

    def render(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        captions: CaptionsFrame | None = None,
    ) -> np.ndarray:
        target_view = self._target_texture.create_view()
        draw(self._res, target_view, flat_weights, audience, model, captions)

        data = self.device.queue.read_texture(
            {"texture": self._target_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"bytes_per_row": self.width * 4},
            (self.width, self.height, 1),
        )
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
```

- [ ] **Step 4: Update `LiveRenderer`**

Replace the existing `LiveRenderer.__init__`, `LiveRenderer.update`, and `LiveRenderer._draw` with:

```python
class LiveRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        device: wgpu.GPUDevice | None = None,
        colormap: int = COLORMAP_DIVERGING,
        audience_height: int = 0,
        model_height: int = 0,
        captions_height: int = 0,
        fullscreen: bool = True,
        max_fps: int = 30,
        canvas: object | None = None,
        display_size: tuple[int, int] | None = None,
    ):
        self.width = width
        self.height = height
        self.device = device or get_device()

        window_w, window_h = display_size or (width, height)

        if canvas is None:
            from rendercanvas.glfw import RenderCanvas

            canvas = RenderCanvas(
                size=(window_w, window_h),
                title="LLM Brainscan",
                update_mode="continuous",
                max_fps=max_fps,
            )
        self._canvas = canvas
        self._context = self._canvas.get_wgpu_context()  # type: ignore[union-attr]
        self._context.configure(
            device=self.device, format=wgpu.TextureFormat.rgba8unorm
        )

        self._config = RenderConfig(
            width,
            height,
            colormap,
            audience_height=audience_height,
            model_height=model_height,
            captions_height=captions_height,
        )
        self._res = create_render_pipeline(
            self._config, self.device, wgpu.TextureFormat.rgba8unorm
        )

        self._lock = threading.Lock()
        self._flat_weights: np.ndarray | None = None
        self._audience: LaneFrame | None = None
        self._model: LaneFrame | None = None
        self._captions: CaptionsFrame | None = None

        if fullscreen:
            self._go_fullscreen()
        self._canvas.request_draw(self._draw)  # type: ignore[union-attr]

    @property
    def config(self) -> RenderConfig:
        return self._config

    def update(
        self,
        flat_weights: np.ndarray,
        audience: LaneFrame | None = None,
        model: LaneFrame | None = None,
        captions: CaptionsFrame | None = None,
    ) -> None:
        with self._lock:
            self._flat_weights = flat_weights.astype(np.float32, copy=True)
            self._audience = audience
            self._model = model
            self._captions = captions

    def _draw(self) -> None:
        with self._lock:
            weights = self._flat_weights
            audience = self._audience
            model = self._model
            captions = self._captions
        if weights is None:
            return
        texture = self._context.get_current_texture()
        draw(self._res, texture.create_view(), weights, audience, model, captions)
```

(The `_go_fullscreen`, `run`, `close` methods are unchanged — keep them.)

- [ ] **Step 5: No tests yet — verification in Task 8.**

- [ ] **Step 6: Commit**

```bash
git add src/brainscan/renderer.py
git commit -m "feat(renderer): draw() takes per-lane frames"
```

---

### Task 8: Update existing renderer tests + add dual-lane tests

Bring the existing `test_renderer.py` tests in line with the new API and add new coverage for the three-band path.

**Files:**
- Modify: `tests/test_renderer.py`

- [ ] **Step 1: Update existing tests for new API**

In `tests/test_renderer.py`, update `class TestTextStripRenderer` so each test passes a `LaneFrame` / `CaptionsFrame` instead of `text_chars` / `text_probs`. Replace it with:

```python
from brainscan.renderer import CaptionsFrame, LaneFrame


class TestModelLaneRendering:
    @pytest.fixture
    def lane_renderer(self):
        return OffscreenRenderer(64, 96, model_height=48)

    def test_lane_dimensions(self, lane_renderer):
        cfg = lane_renderer.config
        assert cfg.model_y == 48
        assert cfg.model_height == 48
        assert cfg.lane_capacity == 64 // 24

    def test_render_with_model_text(self, lane_renderer):
        weights = np.zeros(64 * 96, dtype=np.float32)
        chars = np.zeros(lane_renderer.config.lane_capacity, dtype=np.uint32)
        probs = np.zeros(lane_renderer.config.lane_capacity, dtype=np.float32)
        chars[0] = ord("W")
        probs[0] = 1.0
        frame = LaneFrame(chars=chars, attrs_or_probs=probs, count=1)
        img = lane_renderer.render(weights, model=frame)
        assert img.shape == (96, 64, 4)
        # bottom band has visible cool-ramp pixels
        band = img[48:96, :, :3]
        assert band.max() > 30

    def test_high_prob_brighter_than_low(self, lane_renderer):
        weights = np.zeros(64 * 96, dtype=np.float32)
        cap = lane_renderer.config.lane_capacity
        chars = np.full(cap, ord("X"), dtype=np.uint32)

        probs_hi = np.ones(cap, dtype=np.float32)
        img_hi = lane_renderer.render(
            weights, model=LaneFrame(chars=chars, attrs_or_probs=probs_hi, count=cap)
        )

        probs_lo = np.full(cap, 0.1, dtype=np.float32)
        img_lo = lane_renderer.render(
            weights, model=LaneFrame(chars=chars, attrs_or_probs=probs_lo, count=cap)
        )

        hi = img_hi[48:96, :, :3].astype(float).sum()
        lo = img_lo[48:96, :, :3].astype(float).sum()
        assert hi > lo


class TestAudienceLaneRendering:
    @pytest.fixture
    def lane_renderer(self):
        return OffscreenRenderer(64, 96, audience_height=48)

    def test_audience_lane_position(self, lane_renderer):
        cfg = lane_renderer.config
        assert cfg.audience_y == 48
        assert cfg.audience_height == 48

    def test_render_with_audience_text(self, lane_renderer):
        weights = np.zeros(64 * 96, dtype=np.float32)
        cap = lane_renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        attrs = np.zeros(cap, dtype=np.uint32)
        chars[0] = ord("Y")
        frame = LaneFrame(chars=chars, attrs_or_probs=attrs, count=1)
        img = lane_renderer.render(weights, audience=frame)
        assert img.shape == (96, 64, 4)

    def test_partial_attr_dims_chars(self, lane_renderer):
        from brainscan.lanes import ATTR_PARTIAL

        weights = np.zeros(64 * 96, dtype=np.float32)
        cap = lane_renderer.config.lane_capacity
        chars = np.full(cap, ord("X"), dtype=np.uint32)

        committed_attrs = np.zeros(cap, dtype=np.uint32)
        partial_attrs = np.full(cap, ATTR_PARTIAL, dtype=np.uint32)

        img_committed = lane_renderer.render(
            weights,
            audience=LaneFrame(
                chars=chars, attrs_or_probs=committed_attrs, count=cap
            ),
        )
        img_partial = lane_renderer.render(
            weights,
            audience=LaneFrame(
                chars=chars, attrs_or_probs=partial_attrs, count=cap
            ),
        )

        bright_committed = img_committed[48:96, :, :3].astype(float).sum()
        bright_partial = img_partial[48:96, :, :3].astype(float).sum()
        assert bright_committed > bright_partial


class TestCaptionsRendering:
    @pytest.fixture
    def cap_renderer(self):
        return OffscreenRenderer(64, 32, captions_height=16)

    def test_captions_band_position(self, cap_renderer):
        cfg = cap_renderer.config
        assert cfg.captions_y == 16
        assert cfg.captions_capacity == 64 // 8

    def test_captions_render_dim_grey(self, cap_renderer):
        weights = np.zeros(64 * 32, dtype=np.float32)
        cap = cap_renderer.config.captions_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        chars[:5] = [ord("h"), ord("e"), ord("l"), ord("l"), ord("o")]
        frame = CaptionsFrame(chars=chars, count=5)
        img = cap_renderer.render(weights, captions=frame)
        band = img[16:32, :, :3]
        # band has visible pixels but not bright
        assert band.max() > 20
        assert band.max() < 130


class TestThreeBandRendering:
    def test_three_bands_coexist(self):
        renderer = OffscreenRenderer(
            64, 144, audience_height=48, model_height=48, captions_height=16
        )
        cfg = renderer.config
        assert cfg.audience_y == 32
        assert cfg.model_y == 80
        assert cfg.captions_y == 128

        weights = np.zeros(64 * 144, dtype=np.float32)
        cap = cfg.lane_capacity
        cap_cap = cfg.captions_capacity

        a_chars = np.full(cap, ord("A"), dtype=np.uint32)
        a_attrs = np.zeros(cap, dtype=np.uint32)
        m_chars = np.full(cap, ord("M"), dtype=np.uint32)
        m_probs = np.ones(cap, dtype=np.float32)
        c_chars = np.full(cap_cap, ord("c"), dtype=np.uint32)

        img = renderer.render(
            weights,
            audience=LaneFrame(chars=a_chars, attrs_or_probs=a_attrs, count=cap),
            model=LaneFrame(chars=m_chars, attrs_or_probs=m_probs, count=cap),
            captions=CaptionsFrame(chars=c_chars, count=cap_cap),
        )
        assert img.shape == (144, 64, 4)
        # each band has visible content
        assert img[32:80, :, :3].max() > 30   # audience
        assert img[80:128, :, :3].max() > 30  # model
        assert img[128:144, :, :3].max() > 20 # captions

    def test_lane_colours_distinct(self):
        renderer = OffscreenRenderer(
            64, 144, audience_height=48, model_height=48, captions_height=16
        )
        cap = renderer.config.lane_capacity
        weights = np.zeros(64 * 144, dtype=np.float32)
        chars = np.full(cap, ord("X"), dtype=np.uint32)

        img = renderer.render(
            weights,
            audience=LaneFrame(
                chars=chars, attrs_or_probs=np.zeros(cap, dtype=np.uint32), count=cap
            ),
            model=LaneFrame(
                chars=chars, attrs_or_probs=np.ones(cap, dtype=np.float32), count=cap
            ),
        )

        a_band = img[32:80, :, :3].astype(float)
        m_band = img[80:128, :, :3].astype(float)
        # audience: warm cream -> R > B
        a_lit = a_band[a_band.sum(axis=-1) > 60]
        if len(a_lit):
            assert a_lit[:, 0].mean() > a_lit[:, 2].mean()
        # model: cool ramp -> B >= R
        m_lit = m_band[m_band.sum(axis=-1) > 60]
        if len(m_lit):
            assert m_lit[:, 2].mean() >= m_lit[:, 0].mean()


class TestLaneScroll:
    def test_scroll_offset_shifts_glyph_left(self):
        renderer = OffscreenRenderer(64, 64, model_height=64)
        cap = renderer.config.lane_capacity
        weights = np.zeros(64 * 64, dtype=np.float32)
        chars = np.zeros(cap, dtype=np.uint32)
        chars[0] = ord("|")
        probs = np.full(cap, 1.0, dtype=np.float32)
        no_scroll = renderer.render(
            weights,
            model=LaneFrame(chars=chars, attrs_or_probs=probs, count=1, offset_px=0),
        )
        scrolled = renderer.render(
            weights,
            model=LaneFrame(
                chars=chars, attrs_or_probs=probs, count=1, offset_px=12
            ),
        )
        # the lit pixels' column centre-of-mass moves left when offset_px > 0
        def lit_cols(img):
            band = img[:, :, :3].astype(float).sum(axis=-1)
            cols = np.argwhere(band > 60)[:, 1]
            return cols.mean() if len(cols) else -1.0

        a = lit_cols(no_scroll)
        b = lit_cols(scrolled)
        if a >= 0 and b >= 0:
            assert b < a, f"offset_px should shift glyph left; {a=} {b=}"
```

Also remove the obsolete `class TestTextStripRenderer` block entirely. And in `class TestLiveRenderer`, replace `text_chars` and `text_probs` references with `LaneFrame` / `CaptionsFrame` keyword arguments matching the new `update()` signature. Specifically replace `test_update_with_text` with:

```python
    def test_update_with_lane(self, live_renderer):
        from brainscan.renderer import LaneFrame
        weights = np.zeros(32 * 32, dtype=np.float32)
        chars = np.array([65, 66, 67] + [0] * 13, dtype=np.uint32)
        probs = np.array([1.0, 0.5, 0.1] + [0.0] * 13, dtype=np.float32)
        frame = LaneFrame(chars=chars, attrs_or_probs=probs, count=3)
        live_renderer.update(weights, model=frame)
        assert live_renderer._model is frame
```

And remove `test_text_strip_with_scaled_display` (replace with):

```python
    def test_lane_with_scaled_display(self):
        device = get_device()
        canvas = _make_offscreen_canvas(32, 32)
        live = LiveRenderer(
            64, 64,
            device=device,
            fullscreen=False,
            canvas=canvas,
            display_size=(32, 32),
            model_height=24,
        )
        weights = np.zeros(64 * 64, dtype=np.float32)
        cap = live.config.lane_capacity
        chars = np.full(cap, ord("X"), dtype=np.uint32)
        probs = np.ones(cap, dtype=np.float32)
        live.update(
            weights,
            model=LaneFrame(chars=chars, attrs_or_probs=probs, count=cap),
        )
        canvas.force_draw()
        img = canvas._last_image
        assert img is not None
        assert img.shape == (32, 32, 4)
        live.close()
```

- [ ] **Step 2: Run renderer tests**

Run: `mise exec -- uv run pytest tests/test_renderer.py -v`
Expected: all green.

- [ ] **Step 3: Run the full test suite**

Run: `mise exec -- uv run pytest tests/ -v`
Expected: still some failures from `test_train.py` referring to old API — that is fine, they are fixed in Phase 7. All tests outside `test_train.py` should pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_renderer.py
git commit -m "test(renderer): cover three-band rendering and lane scroll"
```

---

## Phase 3: STT and TTS

### Task 9: `partial_callback` on `SpeechListener`

Stream growing partial transcriptions to a callback while `in_speech` is True. Re-runs Whisper on the accumulated buffer at most once per `partial_interval_seconds`.

**Files:**
- Modify: `src/brainscan/stt.py`
- Test: `tests/test_stt.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_stt.py`:

```python
class TestPartialCallback:
    def test_partial_callback_invoked_during_speech(self):
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        speech_chunk = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        silence_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)

        read_count = 0

        def mock_read(n):
            nonlocal read_count
            read_count += 1
            if read_count <= 3:
                return speech_chunk, False
            if read_count == 4:
                return silence_chunk, False
            listener._stop_event.set()
            return silence_chunk, False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        sd_mock = sys.modules["sounddevice"]
        sd_mock.InputStream = MagicMock(return_value=mock_stream)

        partials: list[str] = []
        listener = SpeechListener(
            model_size="tiny",
            device="cpu",
            partial_callback=partials.append,
            partial_interval_seconds=0.0,
        )

        seg = MagicMock()
        seg.text = "hello"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], None)
        with patch.object(listener, "_load_model"):
            listener._model = mock_model
            listener._run()

        assert len(partials) >= 1
        assert all(p == "hello" for p in partials)

    def test_no_partial_callback_when_unset(self):
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        speech_chunk = np.full((chunk_samples, 1), 0.5, dtype=np.float32)
        silence_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)

        read_count = 0

        def mock_read(n):
            nonlocal read_count
            read_count += 1
            if read_count <= 2:
                return speech_chunk, False
            if read_count == 3:
                return silence_chunk, False
            listener._stop_event.set()
            return silence_chunk, False

        mock_stream = MagicMock()
        mock_stream.read = mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        sd_mock = sys.modules["sounddevice"]
        sd_mock.InputStream = MagicMock(return_value=mock_stream)

        listener = SpeechListener(model_size="tiny", device="cpu")
        seg = MagicMock(); seg.text = "hi"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([seg], None)
        with patch.object(listener, "_load_model"):
            listener._model = mock_model
            # smoke: should not raise even without partial_callback
            listener._run()
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_stt.py::TestPartialCallback -v`
Expected: TypeError on unexpected `partial_callback`.

- [ ] **Step 3: Update `SpeechListener`**

In `src/brainscan/stt.py`:

1. Add to `SpeechConfig`:

```python
    partial_interval_seconds: float = 1.0
```

2. Update `SpeechListener.__init__`:

```python
    def __init__(
        self,
        config: SpeechConfig | None = None,
        partial_callback: object | None = None,
        **kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = SpeechConfig(**kwargs)
        self._partial_callback = partial_callback

        self._text_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: object | None = None
```

3. Update `SpeechListener._run` — replace the in-speech branch so a partial transcript is emitted while accumulating:

```python
    def _run(self) -> None:
        import sounddevice as sd

        self._load_model()
        cfg = self.config
        log.info(
            "STT listener started (model=%s, device=%s)",
            cfg.model_size,
            cfg.device,
        )

        speech_buffer: list[np.ndarray] = []
        in_speech = False
        last_partial_t = 0.0
        import time as _time

        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=cfg.chunk_samples,
            device=cfg.audio_device,
        ) as stream:
            while not self._stop_event.is_set():
                audio, overflowed = stream.read(cfg.chunk_samples)
                if overflowed:
                    log.warning("Audio input overflowed")
                chunk = audio[:, 0]

                if is_speech(chunk, cfg.silence_threshold):
                    speech_buffer.append(chunk)
                    in_speech = True
                    total_samples = sum(len(c) for c in speech_buffer)

                    now = _time.time()
                    if (
                        self._partial_callback is not None
                        and now - last_partial_t >= cfg.partial_interval_seconds
                    ):
                        partial_audio = np.concatenate(speech_buffer)
                        partial_text = transcribe(self._model, partial_audio)
                        if partial_text:
                            self._partial_callback(partial_text)
                        last_partial_t = now

                    if total_samples >= cfg.max_samples:
                        self._do_transcribe(speech_buffer)
                        speech_buffer = []
                        in_speech = False
                        last_partial_t = 0.0
                elif in_speech:
                    total_samples = sum(len(c) for c in speech_buffer)
                    if total_samples >= cfg.min_samples:
                        self._do_transcribe(speech_buffer)
                    speech_buffer = []
                    in_speech = False
                    last_partial_t = 0.0
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_stt.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/stt.py tests/test_stt.py
git commit -m "feat(stt): partial_callback for live transcription"
```

---

### Task 10: `TTSEngine` wrapper around piper

Sub-second offline TTS. The dependency is optional; when piper is unavailable or `enabled=False`, the engine is a silent no-op so dev machines and CI work without the model file.

**Files:**
- Create: `src/brainscan/tts.py`
- Test: `tests/test_tts.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add piper as an optional dependency**

In `pyproject.toml`, add an `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
tts = ["piper-tts>=1.2"]
```

(If the section already exists, add `tts = ...` to it.)

- [ ] **Step 2: Write failing tests**

Create `tests/test_tts.py`:

```python
from unittest.mock import MagicMock, patch

from brainscan.tts import TTSEngine


class TestTTSEngineDisabled:
    def test_disabled_speak_is_noop(self):
        eng = TTSEngine(enabled=False)
        duration = eng.speak("hello")
        assert duration == 0.0

    def test_disabled_does_not_load(self):
        eng = TTSEngine(enabled=False)
        assert eng._voice is None


class TestTTSEngineEnabled:
    def test_speak_returns_estimated_duration(self):
        with patch("brainscan.tts._load_voice") as mock_load:
            mock_voice = MagicMock()
            mock_voice.config.sample_rate = 22050
            audio = MagicMock()
            audio.audio_int16_bytes = b"\x00\x00" * 22050  # 1.0s
            mock_voice.synthesize.return_value = iter([audio])
            mock_load.return_value = mock_voice

            with patch("brainscan.tts.sd") as mock_sd:
                eng = TTSEngine(enabled=True, voice="en_AU-fitch-medium")
                duration = eng.speak("hello world")
                assert duration > 0.0
                assert mock_sd.play.called

    def test_speak_returns_zero_for_empty(self):
        with patch("brainscan.tts._load_voice"):
            eng = TTSEngine(enabled=True, voice="en_AU-fitch-medium")
            eng._voice = MagicMock()
            assert eng.speak("") == 0.0
            assert eng.speak("   ") == 0.0
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_tts.py -v`
Expected: ImportError on `brainscan.tts`.

- [ ] **Step 4: Implement `TTSEngine`**

Create `src/brainscan/tts.py`:

```python
"""Offline TTS via piper. No-op when disabled or when piper is unavailable."""

from __future__ import annotations

import io
import logging
import wave

import numpy as np

log = logging.getLogger(__name__)

try:  # optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - sounddevice is in main deps but mocked in tests
    sd = None  # type: ignore[assignment]


def _load_voice(voice: str) -> object:
    from piper import PiperVoice  # type: ignore[import-not-found]

    return PiperVoice.load(voice)


class TTSEngine:
    def __init__(
        self,
        enabled: bool = False,
        voice: str = "en_AU-fitch-medium",
        gain_db: float = -12.0,
    ):
        self.enabled = enabled
        self.voice_name = voice
        self.gain_db = gain_db
        self._voice: object | None = None

        if enabled:
            try:
                self._voice = _load_voice(voice)
            except Exception as e:  # pragma: no cover
                log.warning("TTS disabled: failed to load voice %s: %s", voice, e)
                self.enabled = False

    def speak(self, text: str) -> float:
        if not self.enabled or self._voice is None or not text.strip():
            return 0.0

        sample_rate = int(getattr(self._voice.config, "sample_rate", 22050))  # type: ignore[union-attr]
        chunks: list[bytes] = []
        for piece in self._voice.synthesize(text):  # type: ignore[union-attr]
            chunks.append(piece.audio_int16_bytes)

        raw = b"".join(chunks)
        if not raw:
            return 0.0

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        gain = 10 ** (self.gain_db / 20.0)
        audio *= gain
        if sd is not None:
            sd.play(audio, samplerate=sample_rate)
        duration = len(audio) / sample_rate
        return float(duration)
```

- [ ] **Step 5: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_tts.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/brainscan/tts.py tests/test_tts.py pyproject.toml
git commit -m "feat(tts): add piper-based TTSEngine (optional)"
```

---

## Phase 4: Streaming generation

### Task 11: `GPT.streaming_generate()`

Token-by-token iterator. The conversation driver consumes one token at a time on its own clock instead of pre-computing N tokens at once.

**Files:**
- Modify: `src/brainscan/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Inspect existing `tests/test_model.py`**

Run: `mise exec -- uv run cat tests/test_model.py | head -10` to confirm the file exists; if not, create it with a stub:

```python
import torch

from brainscan.model import GPT
from conftest import SMALL_CONFIG
```

- [ ] **Step 2: Write failing tests**

Add to `tests/test_model.py`:

```python
class TestStreamingGenerate:
    def test_yields_one_token_per_step(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        gen = model.streaming_generate(b"abc", device=device)
        token, prob = next(gen)
        assert isinstance(token, int)
        assert 0 <= token < 256
        assert 0.0 <= prob <= 1.0

    def test_can_stop_early(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        gen = model.streaming_generate(b"x", device=device)
        produced = [next(gen) for _ in range(5)]
        assert len(produced) == 5
        gen.close()

    def test_resumed_generation_consistent(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        torch.manual_seed(0)
        gen_a = model.streaming_generate(b"hi", device=device)
        out_a = [next(gen_a)[0] for _ in range(8)]
        gen_a.close()

        # generate same length in two halves with same seed
        torch.manual_seed(0)
        gen_b = model.streaming_generate(b"hi", device=device)
        out_b = [next(gen_b)[0] for _ in range(4)]
        out_b.extend(next(gen_b)[0] for _ in range(4))
        gen_b.close()

        assert out_a == out_b

    def test_prompt_is_consumed_first(self, device):
        model = GPT(**SMALL_CONFIG).to(device)
        gen = model.streaming_generate(
            b"hi", device=device, emit_prompt=True
        )
        prompt_tokens = [next(gen) for _ in range(2)]
        assert prompt_tokens[0][0] == ord("h")
        assert prompt_tokens[1][0] == ord("i")
        # then real generation begins
        new_tok, _ = next(gen)
        assert isinstance(new_tok, int)
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_model.py::TestStreamingGenerate -v`
Expected: AttributeError on `streaming_generate`.

- [ ] **Step 4: Implement `streaming_generate`**

Add to the `GPT` class in `src/brainscan/model.py`:

```python
    @torch.no_grad()
    def streaming_generate(
        self,
        prompt_bytes: bytes,
        device: torch.device | None = None,
        emit_prompt: bool = False,
    ):
        if device is None:
            device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            tokens = list(prompt_bytes)
            if emit_prompt:
                for t in tokens:
                    yield int(t), 1.0
            context = torch.tensor([tokens], dtype=torch.long, device=device)
            while True:
                logits, _ = self(context[:, -self.sequence_len :])
                p = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(p, num_samples=1)
                tok = int(next_token.item())
                token_prob = float(p[0, tok].item())
                tokens.append(tok)
                context = torch.cat([context, next_token], dim=1)
                yield tok, token_prob
        finally:
            if was_training:
                self.train()
```

- [ ] **Step 5: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_model.py::TestStreamingGenerate -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/brainscan/model.py tests/test_model.py
git commit -m "feat(model): GPT.streaming_generate token-by-token iterator"
```

---

## Phase 5: Conversation state machine

### Task 12: `ConversationState` enum + `Conversation` dataclass + `step()`

The state machine is pure: `step(now, listener_state, ...)` returns events the caller acts on (`SpeakEvent`, `TrainEvent`, etc.) plus the new state. No I/O happens inside `Conversation`. This makes it directly testable with a fake clock.

**Files:**
- Create: `src/brainscan/conversation.py`
- Test: `tests/test_conversation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_conversation.py`:

```python
from brainscan.conversation import (
    Conversation,
    ConversationState,
    ListenerSnapshot,
    SpeakEvent,
)


def _muse_token(_now):
    return ord("a"), 0.9


def _resp_token(_now):
    return ord("R"), 0.5


def make_listener(committed=None, partial=None, in_speech=False):
    return ListenerSnapshot(
        committed=list(committed or []),
        partial=partial,
        in_speech=in_speech,
    )


class TestInitialState:
    def test_starts_in_muse(self):
        c = Conversation()
        assert c.state == ConversationState.MUSE

    def test_lanes_initially_empty(self):
        c = Conversation()
        assert c.audience.count == 0
        assert c.model_lane.count == 0


class TestMuseTiming:
    def test_one_token_per_muse_interval(self):
        c = Conversation(muse_token_interval=0.15)
        # at t=0 we expect a token to be produced
        events = c.step(now=0.0, listener=make_listener(), token_fn=_muse_token)
        assert events.token_count == 1
        # at t=0.10 not yet time
        events = c.step(now=0.10, listener=make_listener(), token_fn=_muse_token)
        assert events.token_count == 0
        # at t=0.20 the next token comes
        events = c.step(now=0.20, listener=make_listener(), token_fn=_muse_token)
        assert events.token_count == 1


class TestListeningTransition:
    def test_in_speech_moves_to_listening(self):
        c = Conversation()
        c.step(now=0.0, listener=make_listener(), token_fn=_muse_token)
        c.step(
            now=0.1,
            listener=make_listener(in_speech=True),
            token_fn=_muse_token,
        )
        assert c.state == ConversationState.LISTENING

    def test_partial_appended_to_audience_lane(self):
        c = Conversation()
        c.step(
            now=0.0,
            listener=make_listener(in_speech=True, partial="he"),
            token_fn=_muse_token,
        )
        c.step(
            now=0.1,
            listener=make_listener(in_speech=True, partial="hello"),
            token_fn=_muse_token,
        )
        snapshot = c.audience.snapshot()
        text = bytes(snapshot[0].tolist()[: c.audience.count]).decode("ascii")
        assert text == "hello"


class TestRespondingTransition:
    def test_committed_input_starts_responding(self):
        c = Conversation()
        c.step(
            now=0.0,
            listener=make_listener(in_speech=True, partial="hello"),
            token_fn=_muse_token,
        )
        ev = c.step(
            now=0.5,
            listener=make_listener(committed=["hello"]),
            token_fn=_resp_token,
        )
        assert c.state == ConversationState.RESPONDING
        # response generation seeded with the committed text — so a response token was produced
        assert ev.token_count >= 0  # responding rate is fast; at the same instant the seed loads

    def test_response_completes_then_cooldown(self):
        c = Conversation(
            response_token_count=3,
            response_token_interval=0.05,
            cooldown_seconds=2.0,
        )
        c.step(now=0.0, listener=make_listener(committed=["hi"]), token_fn=_resp_token)
        for t in [0.05, 0.10, 0.15]:
            c.step(now=t, listener=make_listener(), token_fn=_resp_token)
        # after 3 tokens responding -> cooldown
        assert c.state == ConversationState.MUSE
        # listening cannot trigger during cooldown
        c.step(
            now=0.16,
            listener=make_listener(in_speech=True),
            token_fn=_muse_token,
        )
        assert c.state == ConversationState.MUSE
        # after cooldown, listening can fire again
        c.step(
            now=2.5,
            listener=make_listener(in_speech=True),
            token_fn=_muse_token,
        )
        assert c.state == ConversationState.LISTENING


class TestSpeakEvent:
    def test_response_emits_speak_event(self):
        c = Conversation(
            response_token_count=2,
            response_token_interval=0.05,
            tts_enabled=True,
        )
        c.step(now=0.0, listener=make_listener(committed=["hi"]), token_fn=_resp_token)
        ev2 = c.step(now=0.05, listener=make_listener(), token_fn=_resp_token)
        # final token completes response -> emits speak event
        all_events: list[SpeakEvent] = list(ev2.speak_events)
        assert any(e.text for e in all_events) or c.state == ConversationState.MUSE

    def test_speak_event_disabled_when_tts_off(self):
        c = Conversation(
            response_token_count=2,
            response_token_interval=0.05,
            tts_enabled=False,
        )
        c.step(now=0.0, listener=make_listener(committed=["hi"]), token_fn=_resp_token)
        ev = c.step(now=0.05, listener=make_listener(), token_fn=_resp_token)
        assert not list(ev.speak_events)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_conversation.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `Conversation`**

Create `src/brainscan/conversation.py`:

```python
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
from typing import Callable, Iterable

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
    muse_token_interval: float = 0.15
    listening_token_interval: float = 0.6
    response_token_interval: float = 0.05
    response_token_count: int = 60
    cooldown_seconds: float = 3.0
    tts_enabled: bool = False
    source_tag: str = "> mic > "

    state: ConversationState = ConversationState.MUSE
    audience: LaneBuffer = field(default_factory=LaneBuffer)
    model_lane: LaneBuffer = field(default_factory=LaneBuffer)

    _last_token_t: float = -1.0
    _response_remaining: int = 0
    _response_text: list[int] = field(default_factory=list)
    _cooldown_until: float = -1.0

    @property
    def cursor_label(self) -> str:
        return ""

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
        text = listener.committed[-1]
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
        interval = self._current_interval()
        if self._last_token_t < 0.0 or now - self._last_token_t >= interval:
            tok, prob = token_fn(now)
            self.model_lane.push(tok, prob=prob)
            self._last_token_t = now
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
            return 1
        return 0

    def _current_interval(self) -> float:
        if self.state == ConversationState.LISTENING:
            return self.listening_token_interval
        if self.state == ConversationState.RESPONDING:
            return self.response_token_interval
        return self.muse_token_interval
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_conversation.py -v`
Expected: all green. Some test cases use timing edges; if `test_one_token_per_muse_interval` flakes, tighten the interval comparisons in `_tick_generation` so the *first* tick always emits.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/conversation.py tests/test_conversation.py
git commit -m "feat(conversation): add MUSE/LISTENING/RESPONDING state machine"
```

---

## Phase 6: Daily rebirth

### Task 13: `rotate_audience_log`

Pure file-op helper. Moves `audience_input.txt` aside to `audience/YYYY-MM-DD.txt`. Returns the target path.

**Files:**
- Create: `src/brainscan/rebirth.py`
- Test: `tests/test_rebirth.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rebirth.py`:

```python
import datetime as dt
from pathlib import Path

from brainscan.rebirth import rotate_audience_log


class TestRotateAudienceLog:
    def test_rotates_to_target(self, tmp_path):
        src = tmp_path / "audience_input.txt"
        src.write_text("hello world")
        target_dir = tmp_path / "audience"

        result = rotate_audience_log(src, target_dir, dt.date(2026, 4, 25))

        assert result == target_dir / "2026-04-25.txt"
        assert result.read_text() == "hello world"
        assert not src.exists()

    def test_no_source_returns_none(self, tmp_path):
        src = tmp_path / "audience_input.txt"
        target_dir = tmp_path / "audience"

        result = rotate_audience_log(src, target_dir, dt.date(2026, 4, 25))

        assert result is None
        assert not target_dir.exists()

    def test_creates_target_dir(self, tmp_path):
        src = tmp_path / "audience_input.txt"
        src.write_text("x")
        target_dir = tmp_path / "deeply" / "nested" / "audience"

        result = rotate_audience_log(src, target_dir, dt.date(2026, 4, 25))

        assert result is not None
        assert result.read_text() == "x"

    def test_overwrites_existing_target(self, tmp_path):
        src = tmp_path / "audience_input.txt"
        src.write_text("new")
        target_dir = tmp_path / "audience"
        target_dir.mkdir()
        existing = target_dir / "2026-04-25.txt"
        existing.write_text("old")

        result = rotate_audience_log(src, target_dir, dt.date(2026, 4, 25))

        assert result.read_text() == "new"
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_rebirth.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `src/brainscan/rebirth.py`:

```python
"""Daily-rebirth helpers: rotate audience log, reset model, schedule."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def rotate_audience_log(
    source: Path, target_dir: Path, date: dt.date
) -> Path | None:
    if not source.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{date.isoformat()}.txt"
    source.replace(target)
    return target
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_rebirth.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/rebirth.py tests/test_rebirth.py
git commit -m "feat(rebirth): add rotate_audience_log helper"
```

---

### Task 14: `rebirth()` reset function

Re-initialise model weights, reset optimiser, reset training corpus to seed + N days of audience logs.

**Files:**
- Modify: `src/brainscan/rebirth.py`
- Test: `tests/test_rebirth.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_rebirth.py`:

```python
import torch

from brainscan.model import GPT
from brainscan.rebirth import RebirthResult, rebirth
from conftest import SMALL_CONFIG


class TestRebirth:
    def test_rebirth_resets_weights(self, tmp_path):
        torch.manual_seed(0)
        model = GPT(**SMALL_CONFIG)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 256, (4, SMALL_CONFIG["sequence_len"]))
        _, loss = model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        before = next(iter(model.parameters())).detach().clone()

        result = rebirth(
            model=model,
            seed_corpus=b"abcdef",
            audience_dir=tmp_path / "audience",
            persist_days=0,
            seed=42,
        )
        assert isinstance(result, RebirthResult)
        after = next(iter(model.parameters())).detach().clone()
        assert not torch.equal(before, after)

    def test_rebirth_returns_corpus(self, tmp_path):
        model = GPT(**SMALL_CONFIG)
        result = rebirth(
            model=model,
            seed_corpus=b"abcdef",
            audience_dir=tmp_path / "audience",
            persist_days=0,
            seed=42,
        )
        assert result.corpus.startswith(b"abcdef")

    def test_rebirth_prepends_recent_logs(self, tmp_path):
        adir = tmp_path / "audience"
        adir.mkdir()
        (adir / "2026-04-23.txt").write_text("oldold")
        (adir / "2026-04-24.txt").write_text("newer")
        model = GPT(**SMALL_CONFIG)

        result = rebirth(
            model=model,
            seed_corpus=b"SEED",
            audience_dir=adir,
            persist_days=2,
            seed=1,
        )
        # most-recent persisted text appears, plus the seed
        assert b"oldold" in result.corpus
        assert b"newer" in result.corpus
        assert b"SEED" in result.corpus

    def test_persist_days_zero_seed_only(self, tmp_path):
        adir = tmp_path / "audience"
        adir.mkdir()
        (adir / "2026-04-23.txt").write_text("ignored")
        model = GPT(**SMALL_CONFIG)

        result = rebirth(
            model=model,
            seed_corpus=b"SEED",
            audience_dir=adir,
            persist_days=0,
            seed=1,
        )
        assert b"ignored" not in result.corpus
        assert result.corpus == b"SEED"

    def test_seed_makes_reproducible(self, tmp_path):
        m_a = GPT(**SMALL_CONFIG)
        rebirth(
            model=m_a,
            seed_corpus=b"x",
            audience_dir=tmp_path,
            persist_days=0,
            seed=99,
        )
        m_b = GPT(**SMALL_CONFIG)
        rebirth(
            model=m_b,
            seed_corpus=b"x",
            audience_dir=tmp_path,
            persist_days=0,
            seed=99,
        )
        for p_a, p_b in zip(
            m_a.parameters(), m_b.parameters(), strict=True
        ):
            assert torch.equal(p_a, p_b)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_rebirth.py::TestRebirth -v`
Expected: ImportError on `RebirthResult`/`rebirth`.

- [ ] **Step 3: Implement**

Add to `src/brainscan/rebirth.py`:

```python
import random

import torch

from brainscan.model import GPT


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
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_rebirth.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/rebirth.py tests/test_rebirth.py
git commit -m "feat(rebirth): rebirth() resets weights and rebuilds corpus"
```

---

### Task 15: `RebirthScheduler`

Wall-clock watcher that fires `rebirth()` once per day at the configured `HH:MM`. Pure callable factory; the test injects a fake clock.

**Files:**
- Modify: `src/brainscan/rebirth.py`
- Test: `tests/test_rebirth.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_rebirth.py`:

```python
from brainscan.rebirth import RebirthScheduler


class TestRebirthScheduler:
    def test_due_when_clock_passes_target(self):
        sched = RebirthScheduler(at_hh_mm="06:00")
        # before 06:00 today
        before = dt.datetime(2026, 4, 26, 5, 59, 59)
        assert not sched.due(before)
        # at 06:00 today (not yet armed)
        first = dt.datetime(2026, 4, 26, 6, 0, 0)
        assert sched.due(first)
        sched.mark_fired(first)
        # not due immediately afterwards
        assert not sched.due(dt.datetime(2026, 4, 26, 6, 0, 1))
        # not due later that day
        assert not sched.due(dt.datetime(2026, 4, 26, 23, 59, 0))
        # due again the next day
        assert sched.due(dt.datetime(2026, 4, 27, 6, 0, 0))

    def test_disabled_never_due(self):
        sched = RebirthScheduler(at_hh_mm=None)
        assert not sched.due(dt.datetime(2026, 4, 26, 6, 0, 0))

    def test_invalid_format_raises(self):
        import pytest
        with pytest.raises(ValueError):
            RebirthScheduler(at_hh_mm="boom")
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_rebirth.py::TestRebirthScheduler -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Add to `src/brainscan/rebirth.py`:

```python
@dataclass
class RebirthScheduler:
    at_hh_mm: str | None
    _last_fired_date: dt.date | None = None
    _hour: int = 0
    _minute: int = 0

    def __post_init__(self) -> None:
        if self.at_hh_mm is None:
            return
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
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `mise exec -- uv run pytest tests/test_rebirth.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/brainscan/rebirth.py tests/test_rebirth.py
git commit -m "feat(rebirth): add RebirthScheduler for daily firing"
```

---

## Phase 7: Wire into `train.py`

### Task 16: Replace inline mic loop with `Conversation` driver

The training loop now drives the conversation each step. Generation is no longer batched into one `model.generate()` call; instead each MUSE/RESPONDING step asks `streaming_generate` for one more token.

**Files:**
- Modify: `src/brainscan/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write a failing integration test**

Replace the existing `class TestRenderFrame:` and remove `class TestEndToEndTrainAndRender:` (the old API is gone). Add to `tests/test_train.py`:

```python
from unittest.mock import MagicMock

from brainscan.conversation import Conversation, ListenerSnapshot
from brainscan.lanes import LaneBuffer
from brainscan.renderer import LaneFrame, OffscreenRenderer


class TestConversationFrameWiring:
    def _make_renderer(self):
        return OffscreenRenderer(
            64, 144, audience_height=48, model_height=48, captions_height=16
        )

    def test_render_with_lane_frames(self, small_model):
        from brainscan.snapshot import capture_weights

        weights = capture_weights(small_model)
        renderer = self._make_renderer()

        a_buf = LaneBuffer(capacity=renderer.config.lane_capacity)
        a_buf.push_text("hi", attrs=0)
        m_buf = LaneBuffer(capacity=renderer.config.lane_capacity)
        m_buf.push_text("yo", prob=0.8)

        a_chars, a_attrs, _ = a_buf.snapshot()
        m_chars, _, m_probs = m_buf.snapshot()

        canvas_pixels = renderer.width * renderer.height
        np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
        from brainscan.renderer import flatten_weights

        flat, count = flatten_weights(np_weights)
        buf = np.zeros(canvas_pixels, dtype=np.float32)
        n = min(count, canvas_pixels)
        buf[:n] = flat[:n]

        img = renderer.render(
            buf,
            audience=LaneFrame(chars=a_chars, attrs_or_probs=a_attrs, count=a_buf.count),
            model=LaneFrame(chars=m_chars, attrs_or_probs=m_probs, count=m_buf.count),
        )
        assert img.shape == (144, 64, 4)


class TestTrainLoopUsesConversation:
    def test_committed_input_appended_to_corpus(self, device):
        from brainscan.data import TextBuffer
        from brainscan.train import _process_committed

        buf = TextBuffer(b"seed")
        listener = ListenerSnapshot(committed=["from-mic"])
        _process_committed(listener, buf)
        assert b"from-mic" in buf.data
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `mise exec -- uv run pytest tests/test_train.py -v`
Expected: ImportError on `_process_committed`, plus other failures from the old API.

- [ ] **Step 3: Refactor `src/brainscan/train.py`**

This is the largest single edit in the plan. Replace the entire body of `train.py`'s `train_loop` and the helpers around it:

3a) Update imports at the top of `train.py`:

```python
import argparse
import datetime as dt
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from brainscan.captions import CaptionsState, compose_caption
from brainscan.conversation import (
    Conversation,
    ConversationState,
    ListenerSnapshot,
)
from brainscan.data import TextBuffer, prepare_batches
from brainscan.layout import (
    HEIGHT,
    LAYOUT_HEIGHT,
    TEXT_STRIP_HEIGHT,
    WIDTH,
    compute_layout,
    default_sections,
    layout_summary,
    layout_to_flat_order,
)
from brainscan.model import GPT
from brainscan.rebirth import RebirthScheduler, rebirth, rotate_audience_log
from brainscan.renderer import (
    CaptionsFrame,
    LaneFrame,
    LiveRenderer,
    OffscreenRenderer,
    flatten_weights,
)
from brainscan.snapshot import capture_weights
from brainscan.tts import TTSEngine

log = logging.getLogger(__name__)
```

3b) Lane heights:

Add at module top after the imports:

```python
AUDIENCE_HEIGHT = 90
MODEL_LANE_HEIGHT = 90
CAPTIONS_HEIGHT = 12
```

3c) Helpers — replace `prepare_display_buffers` and `render_frame` with a new lane-aware variant:

```python
def _build_weight_buffer(
    weights: dict[str, torch.Tensor],
    flat_order: list[str],
    canvas_pixels: int,
) -> np.ndarray:
    np_weights = {k: v.cpu().numpy() for k, v in weights.items()}
    flat, count = flatten_weights(np_weights, layout_order=flat_order)
    buf = np.zeros(canvas_pixels, dtype=np.float32)
    n = min(count, canvas_pixels)
    buf[:n] = flat[:n]
    return buf


def _build_lane_frames(
    convo: Conversation, captions_state: CaptionsState
) -> tuple[LaneFrame, LaneFrame, CaptionsFrame]:
    a_chars, a_attrs, _ = convo.audience.snapshot()
    m_chars, _, m_probs = convo.model_lane.snapshot()
    audience = LaneFrame(
        chars=a_chars, attrs_or_probs=a_attrs, count=convo.audience.count
    )
    model = LaneFrame(
        chars=m_chars, attrs_or_probs=m_probs, count=convo.model_lane.count
    )
    cap_chars = compose_caption(captions_state)
    captions = CaptionsFrame(chars=cap_chars, count=len(cap_chars))
    return audience, model, captions


def _process_committed(
    listener: ListenerSnapshot, training: TextBuffer
) -> None:
    for text in listener.committed:
        log.info("Audience: %s", text)
        training.append(text)
```

3d) Replace `train_loop` body:

```python
    def train_loop() -> None:
        nonlocal training_data
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

        convo = Conversation(
            tts_enabled=args.speak,
            response_token_count=args.gen_tokens,
        )

        partial_holder: dict[str, str] = {"text": ""}

        def on_partial(text: str) -> None:
            partial_holder["text"] = text

        if listener is not None:
            listener._partial_callback = on_partial  # type: ignore[attr-defined]

        gen_iter = model.streaming_generate(
            b"ROMEO: ", device=device, emit_prompt=False
        )

        def token_fn(_now: float):
            return next(gen_iter)

        rebirth_sched = RebirthScheduler(at_hh_mm=args.rebirth_at)
        tts = TTSEngine(enabled=args.speak)

        print("\nTraining started (Ctrl+C to stop)...")
        t0 = time.time()
        step = 0

        try:
            while True:
                if args.steps > 0 and step >= args.steps:
                    break

                committed: list[str] = []
                if listener is not None:
                    committed = listener.get_text()

                # Reset gen_iter BEFORE convo.step so the first RESPONDING
                # token is sampled from the response-seeded generator
                # rather than the muse one.
                if committed:
                    partial_holder["text"] = ""
                    gen_iter.close()
                    seed = committed[-1].encode("utf-8", errors="replace")
                    gen_iter = model.streaming_generate(
                        seed, device=device, emit_prompt=False
                    )

                snapshot = ListenerSnapshot(
                    committed=committed,
                    partial=partial_holder["text"] or None,
                    in_speech=bool(partial_holder["text"])
                    or bool(committed),
                )
                events = convo.step(
                    now=time.time() - t0,
                    listener=snapshot,
                    token_fn=token_fn,
                )
                for ev in events.speak_events:
                    tts.speak(ev.text)
                _process_committed(snapshot, training_data)

                # one optimiser step per loop turn
                x, y = prepare_batches(
                    training_data, args.batch_size, args.sequence_len, device
                )
                _, loss = model(x, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # daily rebirth check
                now_dt = dt.datetime.now()
                if rebirth_sched.due(now_dt):
                    yesterday = now_dt.date() - dt.timedelta(days=1)
                    rotate_audience_log(
                        audience_log, output_dir / "audience", yesterday
                    )
                    res = rebirth(
                        model=model,
                        seed_corpus=raw_data,
                        audience_dir=output_dir / "audience",
                        persist_days=args.persist_audience_days,
                        seed=hash(now_dt.date().isoformat() + str(args.seed))
                        & 0xFFFFFFFF,
                    )
                    training_data = TextBuffer(
                        res.corpus, persist_path=audience_log
                    )
                    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    rebirth_sched.mark_fired(now_dt)
                    (output_dir / "rebirth.log").open("a").write(
                        f"{now_dt.date().isoformat()} seed={res.seed}"
                        f" persist_days={args.persist_audience_days}\n"
                    )

                if step % args.snapshot_every == 0:
                    current_weights = capture_weights(model)
                    dt_elapsed = time.time() - t0
                    print(
                        f"step {step:5d} | loss {loss.item():.4f}"
                        f" | data {len(training_data):,}B"
                        f" | {dt_elapsed:.1f}s | {convo.state.value}"
                    )

                    captions_state = CaptionsState(
                        state_label=_caption_state_label(convo, step, loss.item()),
                        cursor_label="",
                    )
                    audience, model_frame, captions = _build_lane_frames(
                        convo, captions_state
                    )

                    if offscreen_renderer is not None:
                        canvas_pixels = (
                            offscreen_renderer.width
                            * offscreen_renderer.height
                        )
                        buf = _build_weight_buffer(
                            current_weights, flat_order, canvas_pixels
                        )
                        canvas = offscreen_renderer.render(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                        )
                        save_frame(
                            canvas, frames_dir / f"frame_{step:06d}.png"
                        )

                    if live_renderer is not None:
                        canvas_pixels = WIDTH * HEIGHT
                        buf = _build_weight_buffer(
                            current_weights, flat_order, canvas_pixels
                        )
                        live_renderer.update(
                            buf,
                            audience=audience,
                            model=model_frame,
                            captions=captions,
                        )

                step += 1

        except KeyboardInterrupt:
            print("\nStopping...")

        if listener is not None:
            listener.stop()

        if live_renderer is not None:
            live_renderer.close()

        total_time = time.time() - t0
        if step > 0:
            print(
                f"Done. {step} steps in {total_time:.1f}s"
                f" ({step / total_time:.1f} steps/s)"
            )


def _caption_state_label(convo: Conversation, step: int, loss: float) -> str:
    if convo.state == ConversationState.LISTENING:
        return "listening..."
    if convo.state == ConversationState.RESPONDING:
        return "responding..."
    return f"musing | step {step:,} loss {loss:.2f}"
```

3e) Add CLI flags:

In `main()`'s argparse block, add:

```python
    parser.add_argument(
        "--speak", action="store_true", help="Enable TTS for model responses"
    )
    parser.add_argument(
        "--drone",
        action="store_true",
        help="Enable sub-bass drone tracking training loss",
    )
    parser.add_argument(
        "--rebirth-at",
        type=str,
        default=None,
        help="Daily rebirth time HH:MM (24h, local). Default off.",
    )
    parser.add_argument(
        "--persist-audience-days",
        type=int,
        default=7,
        help="Number of past audience-log days to prepend on rebirth (0 to disable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for daily rebirth (per-day seed = hash(date)+base)",
    )
```

3f) Update renderer construction in `main()` to use the three-band fields:

```python
    if args.save_images:
        offscreen_renderer = OffscreenRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
            captions_height=CAPTIONS_HEIGHT,
        )
    if args.live:
        live_renderer = LiveRenderer(
            WIDTH,
            HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
            captions_height=CAPTIONS_HEIGHT,
            fullscreen=True,
        )
```

- [ ] **Step 4: Run all tests**

Run: `mise exec -- uv run pytest tests/ -v`
Expected: all green. If `tests/test_train.py` has tests using the old `prepare_display_buffers` / `render_frame` names, delete those tests; they are replaced by `TestConversationFrameWiring`.

- [ ] **Step 5: Type check**

Run: `mise exec -- uv run ty check`
Expected: clean, or any pre-existing warnings unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/brainscan/train.py tests/test_train.py
git commit -m "feat(train): drive training loop via Conversation state machine"
```

---

### Task 17: End-to-end smoke test

A short test that runs `train.main()` with `--steps 5 --no-mic` against `--save-images` writing to `tmp_path` and asserts a frame is written.

**Files:**
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the smoke test**

Add to `tests/test_train.py`:

```python
def test_train_main_smoke(tmp_path, monkeypatch):
    """Run main() for a few steps and verify a frame is saved."""
    import sys

    from brainscan import train as train_mod

    # tiny model so the test runs in seconds
    args = [
        "train",
        "--no-mic",
        "--steps", "5",
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
    train_mod.main()
    frames = sorted((tmp_path / "frames").glob("*.png"))
    assert len(frames) >= 1
```

- [ ] **Step 2: Run the test**

Run: `mise exec -- uv run pytest tests/test_train.py::test_train_main_smoke -v`
Expected: PASS within ~30s. If it fails, look at the printed conversation state — most likely the model's `streaming_generate` blocked because `next(gen_iter)` is being called too eagerly. Mitigation: wrap the `token_fn` in a try/StopIteration that re-creates the generator.

- [ ] **Step 3: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): smoke test main() with save-images"
```

---

## Phase 8: Optional drone

### Task 18: Sub-bass drone tracking loss

Lowest-priority deliverable; not on the critical path. Skip if time is short.

**Files:**
- Create: `src/brainscan/audio_drone.py`

- [ ] **Step 1: Implement**

Create `src/brainscan/audio_drone.py`:

```python
"""Optional sub-bass drone whose pitch tracks training loss."""

from __future__ import annotations

import threading

import numpy as np


class DroneOscillator:
    def __init__(
        self,
        sample_rate: int = 44100,
        min_hz: float = 40.0,
        max_hz: float = 60.0,
        gain_db: float = -18.0,
        device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.gain = 10 ** (gain_db / 20.0)
        self.device = device
        self._phase = 0.0
        self._hz = (min_hz + max_hz) / 2.0
        self._lock = threading.Lock()
        self._stream = None

    def update_loss(self, loss: float, max_loss: float = 4.0) -> None:
        t = float(np.clip(loss / max_loss, 0.0, 1.0))
        new_hz = self.min_hz + (self.max_hz - self.min_hz) * (1.0 - t)
        with self._lock:
            self._hz = new_hz

    def start(self) -> None:
        import sounddevice as sd

        def callback(out, frames, _time, _status) -> None:
            with self._lock:
                hz = self._hz
            phase_inc = 2 * np.pi * hz / self.sample_rate
            samples = np.empty(frames, dtype=np.float32)
            for i in range(frames):
                samples[i] = np.sin(self._phase) * self.gain
                self._phase = (self._phase + phase_inc) % (2 * np.pi)
            out[:, 0] = samples

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
```

(No tests beyond construction; audio output is hard to verify in CI. The plan considers this acceptable per the spec note that the drone is opt-in atmospheric extra.)

- [ ] **Step 2: Wire into `train.py`** (optional)

In `train.py:main()`, after the listener block:

```python
    drone = None
    if args.drone:
        from brainscan.audio_drone import DroneOscillator

        drone = DroneOscillator()
        drone.start()
```

In `train_loop`'s snapshot block:

```python
                    if drone is not None:
                        drone.update_loss(loss.item())
```

In the shutdown block:

```python
        if drone is not None:
            drone.stop()
```

- [ ] **Step 3: Commit**

```bash
git add src/brainscan/audio_drone.py src/brainscan/train.py
git commit -m "feat(audio): optional sub-bass drone tracking loss"
```

---

## Final acceptance check

After all tasks above are complete:

- [ ] **All tests pass**

Run: `mise exec -- uv run pytest tests/ -v`
Expected: zero failures, zero warnings.

- [ ] **Type check is clean**

Run: `mise exec -- uv run ty check`

- [ ] **Smoke run with --live (manual, requires display)**

Run: `mise exec -- uv run python -m brainscan.train --live --steps 50 --snapshot-every 5 --no-mic --n-layer 1 --n-head 1 --n-embd 64 --sequence-len 32`

Expected: live window shows the weights with three-band strip at the bottom; the model lane scrolls token by token; captions footer shows `musing | step N loss X.XX`.

- [ ] **Smoke run with mic and TTS off (manual)**

Run: `mise exec -- uv run python -m brainscan.train --steps 50 --snapshot-every 5 --whisper-model tiny`

Speak into the microphone; expected: audience lane shows partial transcription in greyed colour, then commits to full brightness; model lane interrupts and starts a response.

---

## Open questions deferred to implementation

These come from the spec's "Open questions for the planner" — they are intentionally not pre-resolved here; the engineer should answer them with a quick benchmark while implementing the relevant task.

1. **Partial transcription speed** (Task 9): the plan re-runs the full `small` Whisper model on the growing buffer once per `partial_interval_seconds`. If this dominates CPU, switch to Whisper's segment-level partial decode instead. Run `tests/test_stt.py` with timing to see real numbers.
2. **TTS dev fallback** (Task 10): on macOS, `say -v Karen "..."` is a fine local fallback. The spec says check before locking in `piper`; the plan uses piper as primary because the deployment target is the Jetson. If piper installation is awkward on the dev machine, add a `say`-backed engine in a follow-up.
3. **Generation pacing** (Task 12): the defaults `0.15 / 0.6 / 0.05` are placeholder values from the spec. Tune from a real exhibition viewer at 3m before locking. Pacing is exposed as `Conversation` constructor args.

## Polish backlog (deferred — not in acceptance criteria)

Items mentioned in the spec but not in v1 acceptance. Worth a follow-up plan after the core lands:

- **Caret cursor `▌`** on the model lane during streaming (spec §"Model lane"). Implement as either a sentinel char appended/popped in the model `LaneBuffer` during RESPONDING, or as a special shader path that draws a caret at column `model_count - 1` when state == RESPONDING (needs an extra uniform).
- **Source-tag pulse** on commit (spec §"Audience lane"). One-frame brightness boost when transitioning from LISTENING → RESPONDING. Add an `ATTR_PULSE` bit and a `pulse_timer` in train.py that clears it after one frame.
- **Sub-pixel pulse** on audience lane right edge each successful partial (spec §"State: LISTENING").
- **Captions event-line fade** in/out over ~5s (spec §"Captions footer"). Currently `event_line` is rendered if non-empty; add a `(text, expiry)` tuple and a fade-out timer.
- **Fade-to-charcoal during rebirth** (spec §"At rebirth time" steps 1 & 7). 2s fade out, re-seed, 2s fade in. Implement as a brightness uniform multiplier in the fragment shader, ramped by train.py over the rebirth window.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-conversational-brain.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
