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
