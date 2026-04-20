from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cache import initialize_latest_snapshot


COMMANDS = ["full", "pipeline", "pass", "dialect", "op", "update", "commit", "pr"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--repo", default=".")
        command_parser.add_argument("--name")
        command_parser.add_argument("--flag")
        command_parser.add_argument("--from-cache-only", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args_list = sys.argv[1:] if argv is None else argv[1:]

    try:
        args = parser.parse_args(args_list)
        if args.command == "pass" and not args.flag:
            parser.error("--flag is required for pass")
    except SystemExit as exc:
        return int(exc.code)

    repo = Path(args.repo).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    out_root = repo / ".agent_pipelines"
    initialize_latest_snapshot(out_root, "0000000", "tree0000", "unknown", "origin", "hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
