from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from cache import initialize_latest_snapshot
from extract import main as extract_main
from git_tools import git_output, map_changed_files_to_objects
from research import promote_pipeline_research as promote_pipeline_research_impl


COMMANDS = ["full", "pipeline", "pass", "dialect", "op", "update", "commit", "pr"]

DIFF_REPORT_NAMES = {
    "update": "update.json",
    "commit": "commit.json",
    "pr": "pr.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze.py")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=".")
    common.add_argument("--snapshot", default=None)
    common.add_argument("--from-cache-only", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        command_parser = subparsers.add_parser(command, parents=[common])
        if command == "commit":
            command_parser.add_argument("--ref", required=True)
        elif command == "pr":
            pr_selector = command_parser.add_mutually_exclusive_group(required=True)
            pr_selector.add_argument("--id")
            pr_selector.add_argument("--url")
        elif command in {"pipeline", "dialect", "op"}:
            command_parser.add_argument("--name", required=True)
        elif command == "pass":
            command_parser.add_argument("--flag", required=True)

    return parser


def promote_pipeline_research(staged: Path, target: Path) -> None:
    promote_pipeline_research_impl(staged, target)


def _load_latest_file_index(repo: Path) -> dict[str, dict[str, list[str]]]:
    index_path = repo / ".agent_pipelines" / "cache" / "latest" / "index" / "files.json"
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text())


def _write_diff_report(repo: Path, command: str, *git_args: str) -> None:
    changed_files_text = git_output(repo, *git_args)
    changed_files = [line for line in changed_files_text.splitlines() if line]
    file_index = _load_latest_file_index(repo)
    mapped = map_changed_files_to_objects(changed_files, file_index)
    report = {"changed_files": changed_files, **mapped}

    report_path = repo / ".agent_pipelines" / "cache" / "latest" / "diffs" / DIFF_REPORT_NAMES[command]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))


def _normalize_pr_selector(args: argparse.Namespace) -> str:
    if args.id is not None:
        selector = args.id
    else:
        match = re.search(r"/pr/(\d+)(?:/)?$", args.url)
        selector = match.group(1) if match is not None else args.url.rsplit("/", 1)[-1]
    return f"refs/pull/{selector}/head"


def _git_value(repo: Path, *args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_metadata(repo: Path) -> tuple[str, str, str, str]:
    base_commit = _git_value(repo, "rev-parse", "HEAD") or "0000000"
    base_tree = _git_value(repo, "rev-parse", "HEAD^{tree}") or "tree0000"
    branch = _git_value(repo, "branch", "--show-current") or "unknown"
    remote = _git_value(repo, "config", "--get", "branch.%s.remote" % branch) or "origin"
    return base_commit, base_tree, branch, remote


def _run_extract(repo: Path, out_root: Path) -> int:
    return extract_main(
        [
            "extract.py",
            str(repo),
            "--out-dir",
            str(out_root),
        ]
    )


def _merge_conditions(*condition_lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for condition_list in condition_lists:
        for condition in condition_list:
            if condition and condition not in seen:
                merged.append(condition)
                seen.add(condition)
    return merged


def _expand_steps(
    steps: list[dict],
    builder_map: dict[str, list[dict]],
    inherited_conditions: list[str] | None = None,
    seen_targets: set[str] | None = None,
) -> list[dict]:
    inherited_conditions = inherited_conditions or []
    seen_targets = seen_targets or set()
    expanded: list[dict] = []

    for step in steps:
        merged_conditions = _merge_conditions(inherited_conditions, step.get("conditions", []))
        step_copy = dict(step)
        step_copy["conditions"] = merged_conditions
        expanded.append(step_copy)

        target = step.get("target")
        if step.get("kind") not in {"helper_call", "cross_call"} or not target:
            continue
        if target in seen_targets:
            continue
        for builder in builder_map.get(target, []):
            expanded.extend(
                _expand_steps(
                    builder.get("steps", []),
                    builder_map,
                    inherited_conditions=merged_conditions,
                    seen_targets=seen_targets | {target},
                )
            )

    return expanded


def _snippet_non_comment_lines(lines: list[str], limit: int = 20) -> str:
    snippet = [
        line.rstrip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("//")
    ]
    return "\n".join(snippet[:limit])


def _first_check_lines(lines: list[str], limit: int = 5) -> list[str]:
    return [line.rstrip() for line in lines if "CHECK" in line][:limit]


def _find_pass_example(
    test_root: Path,
    flag: str,
    example_cache: dict[str, dict | None],
) -> dict | None:
    if flag in example_cache:
        return example_cache[flag]

    if not test_root.exists():
        example_cache[flag] = None
        return None

    direct_pattern = f"-{flag}"
    alt_pattern = f"--{flag}"
    for path in sorted(test_root.rglob("*.mlir")):
        text = path.read_text(errors="replace")
        if direct_pattern not in text and alt_pattern not in text:
            continue
        lines = text.splitlines()
        run_line = next(
            (line.strip() for line in lines if "RUN:" in line and flag in line),
            next((line.strip() for line in lines if "RUN:" in line), ""),
        )
        check_lines = _first_check_lines(lines)
        example = {
            "test_path": str(path),
            "run_line": run_line,
            "input_ir": _snippet_non_comment_lines(lines),
            "output_ir": "\n".join(check_lines),
            "check_lines": check_lines,
            "evidence_type": "test",
        }
        example_cache[flag] = example
        return example

    example_cache[flag] = None
    return None


def _pass_record(
    pass_def: dict,
    test_root: Path,
    example_cache: dict[str, dict | None],
) -> dict:
    flag = pass_def["flag"]
    example = _find_pass_example(test_root, flag, example_cache)
    summary = pass_def.get("summary") or pass_def.get("description") or f"Pass `{flag}`."
    record = {
        "flag": flag,
        "summary": summary,
        "key_options": [option.get("flag", "") for option in pass_def.get("options", []) if option.get("flag")],
        "dialects_touched": pass_def.get("dependent_dialects", []),
    }
    if example is not None:
        record["example"] = example
        return record

    record.update(
        {
            "example": None,
            "example_missing_reason": f"No test using `-{flag}` was found under the bishengir test tree.",
            "fallback_evidence": {
                "td_file": pass_def.get("td_file", ""),
                "td_line": pass_def.get("td_line", 0),
                "constructor_fn": pass_def.get("constructor_fn", ""),
            },
        }
    )
    return record


def _build_pipeline_research_payloads(
    skeleton: dict,
    test_root: Path,
) -> tuple[list[dict], dict[str, dict[str, list[str]]], dict[str, dict]]:
    pass_map = {item["flag"]: item for item in skeleton.get("passes", [])}
    builder_map: dict[str, list[dict]] = defaultdict(list)
    for builder in skeleton.get("builders", []):
        builder_map[builder["name"]].append(builder)

    example_cache: dict[str, dict | None] = {}
    file_index: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"pipelines": set(), "ops": set(), "dialects": set()}
    )
    dialect_index: dict[str, dict] = {}
    payloads: list[dict] = []

    for pipeline in skeleton.get("pipelines", []):
        expanded_steps = _expand_steps(pipeline.get("steps", []), builder_map)
        passes: list[dict] = []
        helpers: list[dict] = []
        for step in expanded_steps:
            kind = step.get("kind")
            if kind == "pass" and step.get("flag") in pass_map:
                pass_def = pass_map[step["flag"]]
                record = _pass_record(pass_def, test_root, example_cache)
                record["conditions"] = step.get("conditions", [])
                record["nested_op"] = step.get("nested_op")
                passes.append(record)

                td_file = pass_def.get("td_file")
                if td_file:
                    file_index[td_file]["pipelines"].add(pipeline["name"])
                    for dialect in record["dialects_touched"]:
                        file_index[td_file]["dialects"].add(dialect)
            elif kind in {"helper_call", "cross_call", "run_pipeline"}:
                helpers.append(
                    {
                        "kind": kind,
                        "target": step.get("target"),
                        "label": step.get("label"),
                        "conditions": step.get("conditions", []),
                    }
                )

        payload = {
            "pipeline": pipeline["name"],
            "passes": passes,
            "helpers": helpers,
        }
        payloads.append(payload)

        pipeline_file = pipeline.get("file")
        if pipeline_file:
            file_index[pipeline_file]["pipelines"].add(pipeline["name"])

        for pass_record in passes:
            for dialect in pass_record["dialects_touched"]:
                dialect_entry = dialect_index.setdefault(
                    dialect,
                    {"dialect": dialect, "ops": [], "pipelines": set()},
                )
                dialect_entry["pipelines"].add(pipeline["name"])
                if pipeline_file:
                    file_index[pipeline_file]["dialects"].add(dialect)

    normalized_file_index = {
        path: {
            "pipelines": sorted(values["pipelines"]),
            "ops": sorted(values["ops"]),
            "dialects": sorted(values["dialects"]),
        }
        for path, values in sorted(file_index.items())
    }
    return payloads, normalized_file_index, dialect_index


def _write_pipeline_payloads(out_root: Path, payloads: list[dict]) -> None:
    research_root = out_root / "cache" / "latest" / "research" / "pipelines"
    for payload in payloads:
        target = research_root / f"{payload['pipeline']}.json"
        staged = research_root.parent / "_staging" / f"{payload['pipeline']}.json"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True))
        promote_pipeline_research(staged, target)


def _write_dialect_payloads(out_root: Path, dialect_index: dict[str, dict]) -> None:
    research_root = out_root / "cache" / "latest" / "research" / "dialects"
    research_root.mkdir(parents=True, exist_ok=True)
    for dialect, payload in sorted(dialect_index.items()):
        normalized = {
            "dialect": dialect,
            "ops": payload["ops"],
            "pipelines": sorted(payload["pipelines"]),
        }
        (research_root / f"{dialect}.json").write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n"
        )


def _write_index_payloads(out_root: Path, file_index: dict[str, dict[str, list[str]]]) -> None:
    index_root = out_root / "cache" / "latest" / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    (index_root / "files.json").write_text(json.dumps(file_index, indent=2, sort_keys=True))


def _build_real_full_snapshot(repo: Path, out_root: Path) -> int:
    base_commit, base_tree, branch, remote = _git_metadata(repo)
    initialize_latest_snapshot(out_root, base_commit, base_tree, branch, remote, "hybrid")
    if _run_extract(repo, out_root) != 0:
        return 1

    skeleton_path = out_root / "skeleton.json"
    if not skeleton_path.exists():
        return 1
    skeleton = json.loads(skeleton_path.read_text())

    bishengir_root = skeleton.get("bishengir_root", "")
    bishengir_path = repo / bishengir_root if bishengir_root else repo
    test_root = bishengir_path / "test"

    pipeline_payloads, file_index, dialect_index = _build_pipeline_research_payloads(
        skeleton,
        test_root,
    )
    _write_pipeline_payloads(out_root, pipeline_payloads)
    _write_dialect_payloads(out_root, dialect_index)
    _write_index_payloads(out_root, file_index)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args_list = sys.argv[1:] if argv is None else argv[1:]

    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return int(exc.code)

    if args.command in {"update", "commit", "pr"}:
        repo = Path(args.repo).resolve()
        if not repo.is_dir():
            return 1
        try:
            if args.command == "update":
                _write_diff_report(repo, args.command, "diff", "--name-only")
            elif args.command == "commit":
                _write_diff_report(
                    repo,
                    args.command,
                    "show",
                    "--pretty=",
                    "--name-only",
                    args.ref,
                )
            else:
                _write_diff_report(
                    repo,
                    args.command,
                    "diff",
                    "--name-only",
                    _normalize_pr_selector(args),
                )
        except subprocess.CalledProcessError:
            return 1
        return 0

    if args.command != "full":
        return 0

    repo = Path(args.repo).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    out_root = repo / ".agent_pipelines"
    if args.from_cache_only or args.snapshot is not None:
        initialize_latest_snapshot(out_root, "0000000", "tree0000", "unknown", "origin", "hybrid")
        return 0
    return _build_real_full_snapshot(repo, out_root)


if __name__ == "__main__":
    raise SystemExit(main())
