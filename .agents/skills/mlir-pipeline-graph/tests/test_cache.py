from __future__ import annotations

import json
from pathlib import Path

from cache import archive_latest_snapshot, initialize_latest_snapshot


def test_initialize_latest_snapshot_writes_manifest(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    manifest = initialize_latest_snapshot(
        out_root=out_root,
        base_commit="abc1234",
        base_tree="tree1234",
        branch="npuir-dev",
        remote="origin",
        analysis_mode="hybrid",
    )

    manifest_path = out_root / "cache" / "latest" / "manifest.json"
    stored = json.loads(manifest_path.read_text())

    assert manifest["base_commit"] == "abc1234"
    assert manifest["analysis_mode"] == "hybrid"
    assert stored["base_commit"] == "abc1234"
    assert stored["analysis_mode"] == "hybrid"
    assert (out_root / "cache" / "latest" / "index").is_dir()
    assert manifest_path.read_text() == json.dumps(manifest, indent=2, sort_keys=True)


def test_archive_latest_snapshot_copies_latest(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    latest = out_root / "cache" / "latest"
    (latest / "index").mkdir(parents=True)
    (latest / "manifest.json").write_text('{"snapshot_id":"20260418-120000-abc1234"}')

    archive_path = archive_latest_snapshot(out_root)

    assert archive_path.name == "20260418-120000-abc1234"
    assert (archive_path / "manifest.json").exists()
    assert (archive_path / "index").is_dir()
