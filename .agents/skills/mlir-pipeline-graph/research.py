from __future__ import annotations

from pathlib import Path


def staging_dir(out_root: Path) -> Path:
    path = out_root / "cache" / "latest" / "_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def promote_json(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(staged.read_text())
