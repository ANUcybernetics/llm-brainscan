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
        text = bytes(chars.tolist()).decode("ascii", errors="replace")
        assert text.endswith("ok")
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
