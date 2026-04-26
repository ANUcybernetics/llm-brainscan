"""Tests for the five visual polish features.

Polish 1: caret cursor on model lane
Polish 2: source-tag pulse on commit
Polish 3: audience right-edge pulse on partials
Polish 4: captions event-line shows for 5s then clears
Polish 5: fade-to-charcoal during rebirth (global_brightness)
"""

from __future__ import annotations

import numpy as np
import pytest

from brainscan.conversation import Conversation, ConversationState
from brainscan.captions import CaptionsState
from brainscan.renderer import LaneFrame, OffscreenRenderer, _UNIFORM_DTYPE
from brainscan.train import _build_lane_frames, _current_event_line


# ---------------------------------------------------------------------------
# Polish 1 & 2 & 3 — LaneFrame field defaults
# ---------------------------------------------------------------------------

class TestLaneFrameFields:
    def _make_frame(self) -> LaneFrame:
        chars = np.zeros(4, dtype=np.uint32)
        probs = np.ones(4, dtype=np.float32)
        return LaneFrame(chars=chars, attrs_or_probs=probs, count=4)

    def test_caret_col_defaults_to_minus_one(self):
        frame = self._make_frame()
        assert frame.caret_col == -1

    def test_pulse_defaults_to_zero(self):
        frame = self._make_frame()
        assert frame.pulse == 0.0

    def test_edge_pulse_defaults_to_zero(self):
        frame = self._make_frame()
        assert frame.edge_pulse == 0.0

    def test_fields_can_be_set(self):
        chars = np.zeros(4, dtype=np.uint32)
        probs = np.ones(4, dtype=np.float32)
        frame = LaneFrame(
            chars=chars, attrs_or_probs=probs, count=4,
            caret_col=7, pulse=0.6, edge_pulse=0.3,
        )
        assert frame.caret_col == 7
        assert frame.pulse == pytest.approx(0.6)
        assert frame.edge_pulse == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Polish 1/2/3/5 — draw() writes correct uniform values
# ---------------------------------------------------------------------------

class TestDrawUniforms:
    @pytest.fixture
    def renderer(self):
        return OffscreenRenderer(64, 64, audience_height=48, model_height=0, captions_height=0)

    def _render_and_check_uniforms(self, renderer, **kwargs):
        """Call render() and return the uniform_data after the call."""
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, **kwargs)
        return renderer._res.uniform_data

    def test_no_caret_writes_sentinel(self, renderer):
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        probs = np.zeros(cap, dtype=np.float32)
        # model lane not passed → sentinel
        u = self._render_and_check_uniforms(renderer)
        assert u["model_caret_col"][0] == np.uint32(0xFFFFFFFF)

    def test_caret_col_negative_writes_sentinel(self, renderer):
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        probs = np.zeros(cap, dtype=np.float32)
        model_frame = LaneFrame(
            chars=chars, attrs_or_probs=probs, count=cap, caret_col=-1
        )
        renderer2 = OffscreenRenderer(64, 64, model_height=48, audience_height=0, captions_height=0)
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer2.render(weights, model=model_frame)
        u = renderer2._res.uniform_data
        assert u["model_caret_col"][0] == np.uint32(0xFFFFFFFF)

    def test_caret_col_positive_writes_value(self):
        renderer = OffscreenRenderer(64, 64, model_height=48, audience_height=0, captions_height=0)
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        probs = np.zeros(cap, dtype=np.float32)
        model_frame = LaneFrame(
            chars=chars, attrs_or_probs=probs, count=cap, caret_col=5
        )
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, model=model_frame)
        u = renderer._res.uniform_data
        assert u["model_caret_col"][0] == np.uint32(5)

    def test_audience_pulse_zero_by_default(self, renderer):
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        attrs = np.zeros(cap, dtype=np.uint32)
        audience_frame = LaneFrame(chars=chars, attrs_or_probs=attrs, count=cap)
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, audience=audience_frame)
        u = renderer._res.uniform_data
        assert u["audience_pulse"][0] == pytest.approx(0.0)

    def test_audience_pulse_written(self, renderer):
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        attrs = np.zeros(cap, dtype=np.uint32)
        audience_frame = LaneFrame(
            chars=chars, attrs_or_probs=attrs, count=cap, pulse=0.7
        )
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, audience=audience_frame)
        u = renderer._res.uniform_data
        assert u["audience_pulse"][0] == pytest.approx(0.7, abs=1e-5)

    def test_audience_edge_pulse_written(self, renderer):
        cap = renderer.config.lane_capacity
        chars = np.zeros(cap, dtype=np.uint32)
        attrs = np.zeros(cap, dtype=np.uint32)
        audience_frame = LaneFrame(
            chars=chars, attrs_or_probs=attrs, count=cap, edge_pulse=0.4
        )
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, audience=audience_frame)
        u = renderer._res.uniform_data
        assert u["audience_edge_pulse"][0] == pytest.approx(0.4, abs=1e-5)

    def test_global_brightness_default_one(self, renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights)
        u = renderer._res.uniform_data
        assert u["global_brightness"][0] == pytest.approx(1.0)

    def test_global_brightness_written(self, renderer):
        weights = np.zeros(64 * 64, dtype=np.float32)
        renderer.render(weights, global_brightness=0.3)
        u = renderer._res.uniform_data
        assert u["global_brightness"][0] == pytest.approx(0.3, abs=1e-5)


# ---------------------------------------------------------------------------
# Polish 5 — global_brightness darkens the rendered output
# ---------------------------------------------------------------------------

class TestGlobalBrightnessRender:
    def test_brightness_zero_gives_dark_image(self):
        renderer = OffscreenRenderer(32, 32)
        weights = np.ones(32 * 32, dtype=np.float32)
        img = renderer.render(weights, global_brightness=0.0)
        assert img[:, :, :3].max() < 5, "global_brightness=0 should give near-black output"

    def test_brightness_one_is_normal(self):
        renderer = OffscreenRenderer(32, 32)
        weights = np.ones(32 * 32, dtype=np.float32)
        img_full = renderer.render(weights, global_brightness=1.0)
        img_dark = renderer.render(weights, global_brightness=0.5)
        assert img_full[:, :, :3].astype(float).mean() > img_dark[:, :, :3].astype(float).mean()


# ---------------------------------------------------------------------------
# Polish 4 — _current_event_line show-for-5s-then-clear
# ---------------------------------------------------------------------------

class TestCurrentEventLine:
    def _make_holder(self, text: str = "", expires_at: float = 0.0):
        return {"text": text, "expires_at": expires_at}

    def test_returns_text_before_expiry(self):
        holder = self._make_holder("dawn 09:00", expires_at=10.0)
        assert _current_event_line(5.0, holder) == "dawn 09:00"

    def test_returns_empty_after_expiry(self):
        holder = self._make_holder("dawn 09:00", expires_at=5.0)
        assert _current_event_line(5.0, holder) == ""

    def test_clears_text_on_expiry(self):
        holder = self._make_holder("something", expires_at=3.0)
        _current_event_line(4.0, holder)
        assert holder["text"] == ""

    def test_does_not_clear_before_expiry(self):
        holder = self._make_holder("still visible", expires_at=100.0)
        _current_event_line(50.0, holder)
        assert holder["text"] == "still visible"

    def test_empty_holder_returns_empty(self):
        holder = self._make_holder("", expires_at=0.0)
        assert _current_event_line(1.0, holder) == ""


# ---------------------------------------------------------------------------
# Polish 1 — _build_lane_frames sets caret_col based on convo state
# ---------------------------------------------------------------------------

class TestBuildLaneFramesCaret:
    def _blank_captions(self):
        return CaptionsState(state_label="", cursor_label="")

    def test_muse_state_has_caret(self):
        convo = Conversation()
        assert convo.state == ConversationState.MUSE
        _, model_frame, _ = _build_lane_frames(convo, self._blank_captions())
        assert model_frame.caret_col == convo.model_lane.count

    def test_listening_state_no_caret(self):
        convo = Conversation()
        convo.state = ConversationState.LISTENING
        _, model_frame, _ = _build_lane_frames(convo, self._blank_captions())
        assert model_frame.caret_col == -1

    def test_responding_state_has_caret(self):
        convo = Conversation()
        convo.state = ConversationState.RESPONDING
        _, model_frame, _ = _build_lane_frames(convo, self._blank_captions())
        assert model_frame.caret_col == convo.model_lane.count


# ---------------------------------------------------------------------------
# Polish 2 & 3 — _build_lane_frames passes through pulse values
# ---------------------------------------------------------------------------

class TestBuildLaneFramesPulse:
    def _blank_captions(self):
        return CaptionsState(state_label="", cursor_label="")

    def test_commit_pulse_forwarded_to_audience(self):
        convo = Conversation()
        aud, _, _ = _build_lane_frames(
            convo, self._blank_captions(), commit_pulse=0.8
        )
        assert aud.pulse == pytest.approx(0.8)

    def test_partial_pulse_forwarded_as_edge_pulse(self):
        convo = Conversation()
        aud, _, _ = _build_lane_frames(
            convo, self._blank_captions(), partial_pulse=0.5
        )
        assert aud.edge_pulse == pytest.approx(0.5)

    def test_default_pulses_are_zero(self):
        convo = Conversation()
        aud, _, _ = _build_lane_frames(convo, self._blank_captions())
        assert aud.pulse == 0.0
        assert aud.edge_pulse == 0.0


# ---------------------------------------------------------------------------
# Polish 5 — uniform struct size sanity
# ---------------------------------------------------------------------------

class TestUniformStructSize:
    def test_dtype_has_new_fields(self):
        names = list(_UNIFORM_DTYPE.names or [])
        assert "model_caret_col" in names
        assert "audience_pulse" in names
        assert "audience_edge_pulse" in names
        assert "global_brightness" in names

    def test_dtype_size_is_at_least_80_bytes(self):
        # 16 original fields × 4 bytes = 64; 4 new fields × 4 bytes = 16 → 80
        assert _UNIFORM_DTYPE.itemsize >= 80
