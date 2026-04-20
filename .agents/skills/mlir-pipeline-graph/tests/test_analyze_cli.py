from __future__ import annotations

from pathlib import Path

import pytest

from analyze import main
from research import promote_json, staging_dir


def test_staging_dir_creates_latest_staging_tree(tmp_path: Path) -> None:
    out_root = tmp_path / ".agent_pipelines"

    path = staging_dir(out_root)

    assert path == out_root / "cache" / "latest" / "_staging"
    assert path.is_dir()
    assert (out_root / "cache" / "latest").is_dir()


def test_promote_json_copies_staged_payload(tmp_path: Path) -> None:
    staged = tmp_path / "_staging" / "pipeline.json"
    target = tmp_path / "cache" / "latest" / "research" / "pipeline.json"
    staged.parent.mkdir(parents=True)
    staged.write_text('{"pipeline":"sample"}')

    promote_json(staged, target)

    assert target.read_text() == staged.read_text()
    assert target.parent.is_dir()


@pytest.mark.parametrize(
    ("argv", "expected_repo_subdir"),
    [
        (["analyze.py", "commit", "--ref", "HEAD"], "repo_commit"),
        (["analyze.py", "pr", "--id", "123"], "repo_pr_id"),
        (["analyze.py", "pr", "--url", "https://example.com/pr/123"], "repo_pr_url"),
        (["analyze.py", "full", "--snapshot", "latest"], "repo_full"),
    ],
)
def test_documented_cli_forms_are_accepted(
    tmp_path: Path, argv: list[str], expected_repo_subdir: str
) -> None:
    repo = tmp_path / expected_repo_subdir
    repo.mkdir()

    argv = argv.copy()
    argv[2:2] = ["--repo", str(repo)]

    exit_code = main(argv)

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


def test_full_command_builds_latest_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main(["analyze.py", "full", "--repo", str(repo), "--from-cache-only"])

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


def test_pass_command_requires_flag(tmp_path: Path) -> None:
    exit_code = main(["analyze.py", "pass", "--repo", str(tmp_path)])

    assert exit_code == 2
