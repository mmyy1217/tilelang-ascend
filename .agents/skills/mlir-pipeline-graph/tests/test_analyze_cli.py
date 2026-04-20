from __future__ import annotations

import json
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


def test_promote_pipeline_research_promotes_valid_example_payload(tmp_path: Path) -> None:
    from analyze import promote_pipeline_research

    staged = tmp_path / "_staging" / "research" / "pipeline.json"
    target = tmp_path / "cache" / "latest" / "research" / "pipeline.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(
        json.dumps(
            {
                "pipeline": "convert-to-hivm-pipeline",
                "passes": [
                    {
                        "flag": "annotation-lowering",
                        "summary": "Erase annotation.mark ops.",
                        "key_options": [],
                        "dialects_touched": ["annotation"],
                        "example": {
                            "test_path": "test/Dialect/Annotation/annotation-lowering.mlir",
                            "run_line": "// RUN: bishengir-opt %s -annotation-lowering | FileCheck %s",
                            "input_ir": "func.func @main() { annotation.mark }",
                            "output_ir": "func.func @main() { return }",
                            "check_lines": ["// CHECK-NOT: annotation.mark"],
                            "evidence_type": "test",
                        },
                    }
                ],
                "helpers": [],
            }
        )
    )

    promote_pipeline_research(staged, target)

    assert json.loads(target.read_text())["passes"][0]["example"]["evidence_type"] == "test"


def test_promote_pipeline_research_promotes_explicit_fallback_payload(
    tmp_path: Path,
) -> None:
    from analyze import promote_pipeline_research

    staged = tmp_path / "_staging" / "research" / "pipeline.json"
    target = tmp_path / "cache" / "latest" / "research" / "pipeline.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(
        json.dumps(
            {
                "pipeline": "convert-to-hivm-pipeline",
                "passes": [
                    {
                        "flag": "annotation-lowering",
                        "summary": "Erase annotation.mark ops.",
                        "key_options": [],
                        "dialects_touched": ["annotation"],
                        "example": None,
                        "example_missing_reason": "No standalone regression test is available yet.",
                        "fallback_evidence": {
                            "notes": ["Observed in downstream pipeline trace"],
                        },
                    }
                ],
                "helpers": [],
            }
        )
    )

    promote_pipeline_research(staged, target)

    promoted = json.loads(target.read_text())
    assert promoted["passes"][0]["example"] is None
    assert promoted["passes"][0]["example_missing_reason"] == "No standalone regression test is available yet."


def test_promote_pipeline_research_rejects_pass_without_example_or_fallback(
    tmp_path: Path,
) -> None:
    from analyze import promote_pipeline_research
    from validators import ValidationError

    staged = tmp_path / "_staging" / "research" / "pipeline.json"
    target = tmp_path / "cache" / "latest" / "research" / "pipeline.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(
        json.dumps(
            {
                "pipeline": "convert-to-hivm-pipeline",
                "passes": [
                    {
                        "flag": "annotation-lowering",
                        "summary": "Erase annotation.mark ops.",
                        "key_options": [],
                        "dialects_touched": ["annotation"],
                    }
                ],
                "helpers": [],
            }
        )
    )

    with pytest.raises(ValidationError) as excinfo:
        promote_pipeline_research(staged, target)

    assert "example" in str(excinfo.value)
    assert not target.exists()


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
    manifest = repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json"
    if argv[1] == "full":
        assert manifest.exists()
    else:
        assert not manifest.exists()


def test_full_command_builds_latest_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main(["analyze.py", "full", "--repo", str(repo), "--from-cache-only"])

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze.py", "full", "--repo", "{repo}", "--name", "x"],
        ["analyze.py", "full", "--repo", "{repo}", "--flag", "x"],
        ["analyze.py", "update", "--repo", "{repo}", "--name", "x"],
        ["analyze.py", "update", "--repo", "{repo}", "--flag", "x"],
    ],
)
def test_full_and_update_reject_command_specific_selectors(
    tmp_path: Path, argv: list[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code == 2


@pytest.mark.parametrize("command", ["pipeline", "dialect", "op"])
def test_pipeline_family_commands_require_name(tmp_path: Path, command: str) -> None:
    repo = tmp_path / f"repo_{command}"
    repo.mkdir()

    exit_code = main(["analyze.py", command, "--repo", str(repo)])

    assert exit_code == 2


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["analyze.py", "pipeline", "--repo", "{repo}", "--name", "sample"], "pipeline"),
        (["analyze.py", "dialect", "--repo", "{repo}", "--name", "sample"], "dialect"),
        (["analyze.py", "op", "--repo", "{repo}", "--name", "sample"], "op"),
        (["analyze.py", "pass", "--repo", "{repo}", "--flag", "sample"], "pass"),
        (["analyze.py", "update", "--repo", "{repo}"], "update"),
        (["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"], "commit"),
        (["analyze.py", "pr", "--repo", "{repo}", "--id", "123"], "pr"),
    ],
)
def test_non_full_commands_preserve_existing_latest_cache(
    tmp_path: Path, argv: list[str], command: str
) -> None:
    repo = tmp_path / f"repo_{command}"
    keep = repo / ".agent_pipelines" / "cache" / "latest" / "research" / "keep.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep")

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code == 0
    assert keep.read_text() == "keep"
    assert not (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


def test_pass_command_requires_flag(tmp_path: Path) -> None:
    exit_code = main(["analyze.py", "pass", "--repo", str(tmp_path)])

    assert exit_code == 2
