import datetime as dt
from pathlib import Path

import torch

from brainscan.model import GPT
from brainscan.rebirth import (
    RebirthFade,
    RebirthPhase,
    RebirthResult,
    RebirthScheduler,
    rebirth,
    rotate_audience_log,
    step_rebirth_phase,
)
from conftest import SMALL_CONFIG


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
            audience_dir=tmp_path / "audience",
            persist_count=0,
            seed=42,
        )
        assert isinstance(result, RebirthResult)
        after = next(iter(model.parameters())).detach().clone()
        assert not torch.equal(before, after)

    def test_rebirth_returns_empty_audience_when_no_history(self, tmp_path):
        model = GPT(**SMALL_CONFIG)
        result = rebirth(
            model=model,
            audience_dir=tmp_path / "audience",
            persist_count=0,
            seed=42,
        )
        assert result.audience == b""

    def test_rebirth_loads_recent_audience_logs(self, tmp_path):
        adir = tmp_path / "audience"
        adir.mkdir()
        (adir / "2026-04-13.txt").write_text("oldold")
        (adir / "2026-04-20.txt").write_text("newer")
        model = GPT(**SMALL_CONFIG)

        result = rebirth(
            model=model,
            audience_dir=adir,
            persist_count=2,
            seed=1,
        )
        assert b"oldold" in result.audience
        assert b"newer" in result.audience

    def test_persist_count_zero_returns_empty(self, tmp_path):
        adir = tmp_path / "audience"
        adir.mkdir()
        (adir / "2026-04-20.txt").write_text("ignored")
        model = GPT(**SMALL_CONFIG)

        result = rebirth(
            model=model,
            audience_dir=adir,
            persist_count=0,
            seed=1,
        )
        assert result.audience == b""

    def test_persist_count_caps_files_loaded(self, tmp_path):
        adir = tmp_path / "audience"
        adir.mkdir()
        (adir / "2026-04-06.txt").write_text("week14")
        (adir / "2026-04-13.txt").write_text("week15")
        (adir / "2026-04-20.txt").write_text("week16")
        model = GPT(**SMALL_CONFIG)

        result = rebirth(
            model=model,
            audience_dir=adir,
            persist_count=2,
            seed=1,
        )
        # only the two most recent files appear
        assert b"week14" not in result.audience
        assert b"week15" in result.audience
        assert b"week16" in result.audience

    def test_seed_makes_reproducible(self, tmp_path):
        m_a = GPT(**SMALL_CONFIG)
        rebirth(
            model=m_a,
            audience_dir=tmp_path,
            persist_count=0,
            seed=99,
        )
        m_b = GPT(**SMALL_CONFIG)
        rebirth(
            model=m_b,
            audience_dir=tmp_path,
            persist_count=0,
            seed=99,
        )
        for p_a, p_b in zip(
            m_a.parameters(), m_b.parameters(), strict=True
        ):
            assert torch.equal(p_a, p_b)


class TestRebirthScheduler:
    def test_fires_at_target_dow_and_time(self):
        # Apr 27 2026 is Monday, iso-week 2026-W18
        sched = RebirthScheduler(at_dow_hh_mm="MON 02:00")
        before = dt.datetime(2026, 4, 27, 1, 59, 59)
        assert not sched.due(before)
        first = dt.datetime(2026, 4, 27, 2, 0, 0)
        assert sched.due(first)

    def test_not_due_again_in_same_iso_week(self):
        sched = RebirthScheduler(at_dow_hh_mm="MON 02:00")
        first = dt.datetime(2026, 4, 27, 2, 0, 0)  # Mon W18
        assert sched.due(first)
        sched.mark_fired(first)
        assert not sched.due(dt.datetime(2026, 4, 27, 2, 0, 1))
        # Sunday of same iso-week
        assert not sched.due(dt.datetime(2026, 5, 3, 23, 59, 0))

    def test_due_again_next_iso_week(self):
        sched = RebirthScheduler(at_dow_hh_mm="MON 02:00")
        sched.mark_fired(dt.datetime(2026, 4, 27, 2, 0, 0))
        # Mon of next iso-week (W19)
        assert sched.due(dt.datetime(2026, 5, 4, 2, 0, 0))

    def test_due_late_if_missed_within_week(self):
        # System down at MON 02:00; comes up Wed — should still fire this week
        sched = RebirthScheduler(at_dow_hh_mm="MON 02:00")
        wed = dt.datetime(2026, 4, 29, 14, 0, 0)
        assert sched.due(wed)

    def test_target_day_before_target_time_not_due(self):
        sched = RebirthScheduler(at_dow_hh_mm="WED 14:00")
        early_wed = dt.datetime(2026, 4, 29, 13, 59, 59)
        assert not sched.due(early_wed)

    def test_dow_case_insensitive(self):
        sched = RebirthScheduler(at_dow_hh_mm="mon 02:00")
        assert sched.due(dt.datetime(2026, 4, 27, 2, 0, 0))

    def test_disabled_never_due(self):
        sched = RebirthScheduler(at_dow_hh_mm=None)
        assert not sched.due(dt.datetime(2026, 4, 27, 2, 0, 0))

    def test_invalid_dow_raises(self):
        import pytest
        with pytest.raises(ValueError):
            RebirthScheduler(at_dow_hh_mm="XYZ 02:00")

    def test_missing_dow_raises(self):
        import pytest
        with pytest.raises(ValueError):
            RebirthScheduler(at_dow_hh_mm="02:00")

    def test_invalid_time_raises(self):
        import pytest
        with pytest.raises(ValueError):
            RebirthScheduler(at_dow_hh_mm="MON 25:00")

    def test_garbage_raises(self):
        import pytest
        with pytest.raises(ValueError):
            RebirthScheduler(at_dow_hh_mm="boom")


class TestRebirthSchedulerPersistence:
    def test_state_path_persists_last_fired_iso_week(self, tmp_path):
        state_path = tmp_path / "rebirth.last"
        sched_a = RebirthScheduler(
            at_dow_hh_mm="MON 02:00", state_path=state_path
        )
        first = dt.datetime(2026, 4, 27, 2, 0, 0)  # W18
        assert sched_a.due(first)
        sched_a.mark_fired(first)
        # new process: same state path, same iso-week -> not due
        sched_b = RebirthScheduler(
            at_dow_hh_mm="MON 02:00", state_path=state_path
        )
        assert not sched_b.due(dt.datetime(2026, 4, 30, 12, 0, 0))
        # next iso-week -> due again
        assert sched_b.due(dt.datetime(2026, 5, 4, 2, 0, 0))

    def test_no_state_path_falls_back_to_in_memory(self):
        sched = RebirthScheduler(at_dow_hh_mm="MON 02:00")
        assert sched.due(dt.datetime(2026, 4, 27, 2, 0, 0))

    def test_state_path_written_on_mark_fired(self, tmp_path):
        state_path = tmp_path / "rebirth.last"
        sched = RebirthScheduler(
            at_dow_hh_mm="MON 02:00", state_path=state_path
        )
        sched.mark_fired(dt.datetime(2026, 4, 27, 2, 0, 0))
        assert state_path.read_text().strip() == "2026-W18"

    def test_corrupt_state_file_ignored(self, tmp_path):
        state_path = tmp_path / "rebirth.last"
        state_path.write_text("not-a-week")
        sched = RebirthScheduler(
            at_dow_hh_mm="MON 02:00", state_path=state_path
        )
        # should not raise and should treat as never-fired
        assert sched.due(dt.datetime(2026, 4, 27, 2, 0, 0))


class TestRebirthFadeStateMachine:
    def test_idle_stays_idle_when_not_due(self):
        s = RebirthFade()
        new_s, brightness, perform = step_rebirth_phase(s, now_t=0.0, is_due=False)
        assert new_s.phase == RebirthPhase.IDLE
        assert brightness == 1.0
        assert not perform

    def test_idle_to_fading_out_when_due(self):
        s = RebirthFade()
        new_s, brightness, perform = step_rebirth_phase(s, now_t=10.0, is_due=True)
        assert new_s.phase == RebirthPhase.FADING_OUT
        assert new_s.started_at == 10.0
        assert brightness == 1.0
        assert not perform

    def test_fading_out_brightness_ramps_down(self):
        s = RebirthFade(phase=RebirthPhase.FADING_OUT, started_at=10.0)
        _, b1, _ = step_rebirth_phase(s, now_t=10.0, is_due=False, fade_duration=2.0)
        _, b2, _ = step_rebirth_phase(s, now_t=11.0, is_due=False, fade_duration=2.0)
        _, b3, _ = step_rebirth_phase(s, now_t=11.5, is_due=False, fade_duration=2.0)
        assert b1 == 1.0
        assert b2 == 0.5
        assert b3 == 0.25

    def test_fading_out_completes_signals_perform(self):
        s = RebirthFade(phase=RebirthPhase.FADING_OUT, started_at=10.0)
        new_s, brightness, perform = step_rebirth_phase(
            s, now_t=12.0, is_due=False, fade_duration=2.0
        )
        assert new_s.phase == RebirthPhase.FADING_IN
        assert new_s.started_at == 12.0
        assert brightness == 0.0
        assert perform

    def test_fading_in_brightness_ramps_up(self):
        s = RebirthFade(phase=RebirthPhase.FADING_IN, started_at=12.0)
        _, b1, _ = step_rebirth_phase(s, now_t=12.0, is_due=False, fade_duration=2.0)
        _, b2, _ = step_rebirth_phase(s, now_t=13.0, is_due=False, fade_duration=2.0)
        assert b1 == 0.0
        assert b2 == 0.5

    def test_fading_in_completes_returns_to_idle(self):
        s = RebirthFade(phase=RebirthPhase.FADING_IN, started_at=12.0)
        new_s, brightness, _ = step_rebirth_phase(
            s, now_t=14.0, is_due=False, fade_duration=2.0
        )
        assert new_s.phase == RebirthPhase.IDLE
        assert brightness == 1.0

    def test_is_due_during_fade_does_not_re_fire(self):
        # While fading_out / fading_in, is_due must be ignored
        s = RebirthFade(phase=RebirthPhase.FADING_OUT, started_at=10.0)
        new_s, _, perform = step_rebirth_phase(
            s, now_t=11.0, is_due=True, fade_duration=2.0
        )
        # Phase should still be FADING_OUT, no re-trigger
        assert new_s.phase == RebirthPhase.FADING_OUT
        assert not perform
