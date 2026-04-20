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
    monkeypatch, tmp_path: Path, argv: list[str], expected_repo_subdir: str
) -> None:
    repo = tmp_path / expected_repo_subdir
    repo.mkdir()

    argv = argv.copy()
    argv[2:2] = ["--repo", str(repo)]
    if argv[1] in {"commit", "pr"}:
        monkeypatch.setattr("analyze.git_output", lambda repo_path, *args: "", raising=False)

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
    monkeypatch, tmp_path: Path, argv: list[str], command: str
) -> None:
    repo = tmp_path / f"repo_{command}"
    keep = repo / ".agent_pipelines" / "cache" / "latest" / "research" / "keep.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep")
    if argv[1] in {"update", "commit", "pr"}:
        monkeypatch.setattr("analyze.git_output", lambda repo_path, *args: "", raising=False)

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code == 0
    assert keep.read_text() == "keep"
    assert not (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze.py", "update", "--repo", "{repo}"],
        ["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"],
        ["analyze.py", "pr", "--repo", "{repo}", "--id", "123"],
    ],
)
def test_update_commit_and_pr_create_latest_diffs_dir(
    monkeypatch, tmp_path: Path, argv: list[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("analyze.git_output", lambda repo_path, *args: "", raising=False)

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "diffs").is_dir()


@pytest.mark.parametrize(
    ("argv", "report_name"),
    [
        (["analyze.py", "update", "--repo", "{repo}"], "update.json"),
        (["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"], "commit.json"),
        (["analyze.py", "pr", "--repo", "{repo}", "--id", "123"], "pr.json"),
    ],
)
def test_update_commit_and_pr_write_diff_report(
    monkeypatch, tmp_path: Path, argv: list[str], report_name: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    fixture = json.loads(
        (
            Path(".agents/skills/mlir-pipeline-graph/tests/fixtures/sample_diff.json")
        ).read_text()
    )
    index_path = repo / ".agent_pipelines" / "cache" / "latest" / "index" / "files.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(fixture["file_index"]))

    monkeypatch.setattr(
        "analyze.git_output",
        lambda repo_path, *args: "\n".join(fixture["changed_files"]),
        raising=False,
    )

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code == 0
    report = json.loads(
        (
            repo / ".agent_pipelines" / "cache" / "latest" / "diffs" / report_name
        ).read_text()
    )
    assert report["changed_files"] == fixture["changed_files"]
    assert report["pipelines"] == ["convert-to-hivm-pipeline"]
    assert report["ops"] == ["hfusion.matmul"]
    assert sorted(report["dialects"]) == ["hfusion", "hivm"]


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze.py", "update", "--repo", "{repo}"],
        ["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"],
        ["analyze.py", "pr", "--repo", "{repo}", "--id", "123"],
    ],
)
def test_update_commit_and_pr_reject_nonexistent_repo(
    tmp_path: Path, argv: list[str]
) -> None:
    repo = tmp_path / "missing-repo"

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code != 0
    assert not (repo / ".agent_pipelines").exists()


def test_commit_command_passes_ref_to_git_output(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_git_output(repo_path, *args):
        calls.append((repo_path, args))
        return ""

    monkeypatch.setattr("analyze.git_output", fake_git_output, raising=False)

    exit_code = main(["analyze.py", "commit", "--repo", str(repo), "--ref", "HEAD~3"])

    assert exit_code == 0
    assert calls == [(repo.resolve(), ("show", "--pretty=", "--name-only", "HEAD~3"))]


@pytest.mark.parametrize(
    ("argv", "expected_selector"),
    [
        (["analyze.py", "pr", "--repo", "{repo}", "--id", "123"], "refs/pull/123/head"),
        (
            ["analyze.py", "pr", "--repo", "{repo}", "--url", "https://example.com/pr/123"],
            "refs/pull/123/head",
        ),
    ],
)
def test_pr_command_passes_selector_to_git_output(
    monkeypatch, tmp_path: Path, argv: list[str], expected_selector: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_git_output(repo_path, *args):
        calls.append((repo_path, args))
        return ""

    monkeypatch.setattr("analyze.git_output", fake_git_output, raising=False)

    rendered = [part.format(repo=str(repo)) for part in argv]

    exit_code = main(rendered)

    assert exit_code == 0
    assert calls == [(repo.resolve(), ("diff", "--name-only", expected_selector))]


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze.py", "update", "--repo", "{repo}"],
        ["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"],
        ["analyze.py", "pr", "--repo", "{repo}", "--id", "123"],
    ],
)
def test_update_commit_and_pr_fail_on_git_errors(
    monkeypatch, tmp_path: Path, argv: list[str]
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_git_output(repo_path, *args):
        raise subprocess.CalledProcessError(returncode=128, cmd=["git", *args])

    monkeypatch.setattr("analyze.git_output", fake_git_output, raising=False)

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code != 0


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze.py", "update", "--repo", "{repo}"],
        ["analyze.py", "commit", "--repo", "{repo}", "--ref", "HEAD"],
        ["analyze.py", "pr", "--repo", "{repo}", "--id", "123"],
    ],
)
def test_update_commit_and_pr_reject_plain_non_git_repo(
    tmp_path: Path, argv: list[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main([part.format(repo=str(repo)) for part in argv])

    assert exit_code != 0


def test_pass_command_requires_flag(tmp_path: Path) -> None:
    exit_code = main(["analyze.py", "pass", "--repo", str(tmp_path)])

    assert exit_code == 2


def test_skill_mentions_html_first_and_required_examples() -> None:
    text = Path(".agents/skills/mlir-pipeline-graph/SKILL.md").read_text()

    assert "HTML-first" in text
    assert "example_missing_reason" in text
    assert "analyze.py" in text
