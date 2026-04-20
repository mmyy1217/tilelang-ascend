from __future__ import annotations

from pathlib import Path

from analyze import main


def test_full_command_builds_latest_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main(["analyze.py", "full", "--repo", str(repo), "--from-cache-only"])

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


def test_pass_command_requires_flag(tmp_path: Path) -> None:
    exit_code = main(["analyze.py", "pass", "--repo", str(tmp_path)])

    assert exit_code == 2
