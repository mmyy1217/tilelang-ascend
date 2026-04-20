from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import cache
from cache import archive_latest_snapshot, initialize_latest_snapshot


def test_create_snapshot_id_shape(monkeypatch):
    class _FakeDatetime:
        @staticmethod
        def now(tz):
            class _FakeNow:
                @staticmethod
                def strftime(fmt):
                    assert fmt == "%Y%m%d-%H%M%S"
                    return "20260418-120000"

            return _FakeNow()

    monkeypatch.setattr(cache, "datetime", _FakeDatetime)

    snapshot_id = cache.create_snapshot_id("abc1234")

    assert snapshot_id == "20260418-120000-abc1234"
    assert re.fullmatch(r"\d{8}-\d{6}-.{7}", snapshot_id)


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
    assert (out_root / "cache" / "latest" / "research").is_dir()
    assert (out_root / "cache" / "latest" / "diffs").is_dir()
    assert (out_root / "cache" / "latest" / "git").is_dir()
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


def test_archive_latest_snapshot_refuses_existing_history_snapshot(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    latest = out_root / "cache" / "latest"
    (latest / "index").mkdir(parents=True)
    (latest / "research").mkdir()
    (latest / "diffs").mkdir()
    (latest / "git").mkdir()
    (latest / "manifest.json").write_text('{"snapshot_id":"20260418-120000-abc1234"}')

    first_archive = archive_latest_snapshot(out_root)
    (first_archive / "marker.txt").write_text("keep")

    with pytest.raises(FileExistsError):
        archive_latest_snapshot(out_root)

    assert (first_archive / "marker.txt").read_text() == "keep"
