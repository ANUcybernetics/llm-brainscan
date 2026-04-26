import datetime as dt
from pathlib import Path

import torch

from brainscan.model import GPT
from brainscan.rebirth import RebirthResult, rebirth, rotate_audience_log
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
