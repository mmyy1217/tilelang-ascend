from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cache import initialize_latest_snapshot
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
    try:
        changed_files_text = git_output(repo, *git_args)
    except subprocess.CalledProcessError as exc:
        error_text = f"{exc.stderr or ''}\n{exc.stdout or ''}".lower()
        if "not a git repository" in error_text:
            changed_files_text = ""
        else:
            raise
    changed_files = [line for line in changed_files_text.splitlines() if line]
    file_index = _load_latest_file_index(repo)
    mapped = map_changed_files_to_objects(changed_files, file_index)
    report = {"changed_files": changed_files, **mapped}

    report_path = repo / ".agent_pipelines" / "cache" / "latest" / "diffs" / DIFF_REPORT_NAMES[command]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))


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
                selector = args.id if args.id is not None else args.url
                _write_diff_report(repo, args.command, "diff", "--name-only", selector)
        except subprocess.CalledProcessError:
            return 1
        return 0

    if args.command != "full":
        return 0

    repo = Path(args.repo).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    out_root = repo / ".agent_pipelines"
    initialize_latest_snapshot(out_root, "0000000", "tree0000", "unknown", "origin", "hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
