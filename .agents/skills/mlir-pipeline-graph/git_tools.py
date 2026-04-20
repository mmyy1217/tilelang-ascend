from __future__ import annotations

import subprocess
from pathlib import Path


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def map_changed_files_to_objects(
    changed_files: list[str], file_index: dict[str, dict[str, list[str]]]
) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {"pipelines": [], "ops": [], "dialects": []}
    seen: dict[str, set[str]] = {key: set() for key in mapped}

    for path in changed_files:
        file_metadata = file_index.get(path, {})
        for key in mapped:
            for value in file_metadata.get(key, []):
                if value in seen[key]:
                    continue
                seen[key].add(value)
                mapped[key].append(value)

    return mapped
