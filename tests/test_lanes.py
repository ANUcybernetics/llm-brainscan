import numpy as np

from brainscan.lanes import LaneBuffer, ATTR_PARTIAL, ATTR_SOURCE_TAG


class TestLaneBufferConstruction:
    def test_default_capacity_is_240(self):
        buf = LaneBuffer()
        assert buf.capacity == 240

    def test_default_capacity_matches_8k_renderer_lane_capacity(self):
        """LaneBuffer.snapshot() writes capacity*4 bytes into the renderer's
        lane buffer, which is sized from RenderConfig.lane_capacity. A
        mismatch silently overflows on the first frame (wgpu Invalid
        data_length); pin them together.
        """
        from brainscan.layout import HEIGHT, WIDTH
        from brainscan.renderer import RenderConfig
        from brainscan.train import AUDIENCE_HEIGHT, MODEL_LANE_HEIGHT

        cfg = RenderConfig(
            width=WIDTH,
            height=HEIGHT,
            audience_height=AUDIENCE_HEIGHT,
            model_height=MODEL_LANE_HEIGHT,
        )
        assert LaneBuffer().capacity == cfg.lane_capacity

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
        assert chars[0] == ord("A")
        assert chars[1] == ord("B")
        assert probs[1] == 0.5
        assert buf.count == 2

    def test_push_above_capacity_drops_oldest(self):
        buf = LaneBuffer(capacity=3)
        for c in "ABCD":
            buf.push(ord(c), prob=1.0)
        chars, _attrs, _probs = buf.snapshot()
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
        assert chars[0] == ord("X")
        assert chars[1] == ord("h")
        assert chars[5] == ord("o")
        assert buf.count == 6

    def test_commit_partial_clears_partial_bit(self):
        buf = LaneBuffer(capacity=16)
        buf.replace_tail("hi", attrs=ATTR_PARTIAL)
        buf.commit_partial(prefix="▸ mic ▸ ", attrs=ATTR_SOURCE_TAG)
        chars, attrs, _ = buf.snapshot()
        expected = list("▸ mic ▸ ".encode("utf-8"))
        assert chars[0] == expected[0]
        assert not (attrs[buf.count - 1] & ATTR_PARTIAL)

    def test_commit_partial_with_no_prior_partial(self):
        buf = LaneBuffer(capacity=8)
        buf.push(ord("A"), prob=1.0)
        buf.commit_partial(prefix="> ", attrs=ATTR_SOURCE_TAG)
        chars, attrs, _ = buf.snapshot()
        assert chars[0] == ord("A")
        assert chars[1] == ord(">")
        assert chars[2] == ord(" ")
        assert attrs[1] & ATTR_SOURCE_TAG
        assert buf.count == 3


class TestLaneBufferOverflow:
    def test_committed_clamps_on_overflow_of_committed_bytes(self):
        # Push A,B,C (all committed, capacity=3 exactly full, _committed=3).
        # Push D: buffer overflows to [B,C,D], _trim clamps _committed to 2,
        # but push then sets _committed = len(_chars) = 3 (D is non-partial).
        # replace_tail("xy") truncates to _committed=3 → [B,C,D], appends xy
        # → [B,C,D,x,y], _trim removes 2 excess → [D,x,y].
        buf = LaneBuffer(capacity=3)
        for c in "ABC":
            buf.push(ord(c), prob=1.0)
        buf.push(ord("D"), prob=1.0)
        buf.replace_tail("xy", attrs=ATTR_PARTIAL)
        chars, _, _ = buf.snapshot()
        assert buf.count == 3
        assert chars[0] == ord("D")
        assert chars[1] == ord("x")
        assert chars[2] == ord("y")


class TestLaneBufferSourceTag:
    def test_push_with_source_tag(self):
        buf = LaneBuffer(capacity=16)
        buf.push_text("▸ mic ▸ ", attrs=ATTR_SOURCE_TAG)
        _chars, attrs, _ = buf.snapshot()
        for i in range(buf.count):
            assert attrs[i] & ATTR_SOURCE_TAG
