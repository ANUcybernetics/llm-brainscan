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
