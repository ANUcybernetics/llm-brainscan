"""Smoke test that tuning constants exist and are sane."""
from brainscan import tuning


def test_pacing_intervals_are_positive_and_ordered():
    assert tuning.RESPONDING_TOKEN_INTERVAL < tuning.MUSE_TOKEN_INTERVAL
    assert tuning.MUSE_TOKEN_INTERVAL < tuning.LISTENING_TOKEN_INTERVAL


def test_cooldown_positive():
    assert tuning.COOLDOWN_SECONDS > 0


def test_drone_pitch_range():
    assert tuning.DRONE_MIN_HZ < tuning.DRONE_MAX_HZ


def test_persist_days_default_is_a_week():
    assert tuning.PERSIST_AUDIENCE_DAYS_DEFAULT == 7
