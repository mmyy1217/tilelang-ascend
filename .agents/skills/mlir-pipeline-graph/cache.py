from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def create_snapshot_id(base_commit: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{base_commit[:7]}"


def initialize_latest_snapshot(
    out_root: Path,
    base_commit: str,
    base_tree: str,
    branch: str,
    remote: str,
    analysis_mode: str,
) -> dict:
    latest_root = out_root / "cache" / "latest"
    (latest_root / "index").mkdir(parents=True, exist_ok=True)

    manifest = {
        "analysis_mode": analysis_mode,
        "base_commit": base_commit,
        "base_tree": base_tree,
        "branch": branch,
        "remote": remote,
        "schema_version": 1,
        "snapshot_id": create_snapshot_id(base_commit),
    }
    _write_json(latest_root / "manifest.json", manifest)
    return manifest


def archive_latest_snapshot(out_root: Path) -> Path:
    latest_root = out_root / "cache" / "latest"
    manifest = json.loads((latest_root / "manifest.json").read_text())
    snapshot_id = manifest["snapshot_id"]
    history_root = out_root / "cache" / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    archive_root = history_root / snapshot_id

    if archive_root.exists():
        shutil.rmtree(archive_root)
    shutil.copytree(latest_root, archive_root)
    return archive_root
