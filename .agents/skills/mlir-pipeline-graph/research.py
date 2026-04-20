from __future__ import annotations

import json
from pathlib import Path

from validators import validate_pipeline_research


def staging_dir(out_root: Path) -> Path:
    path = out_root / "cache" / "latest" / "_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def promote_json(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(staged.read_text())


def promote_pipeline_research(staged: Path, target: Path) -> None:
    payload = json.loads(staged.read_text())
    validated = validate_pipeline_research(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")
