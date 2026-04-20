from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import cache
from cache import archive_latest_snapshot, initialize_latest_snapshot


def test_create_snapshot_id_shape(monkeypatch):
    class _FakeDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 4, 18, 12, 0, 0, 123456, tzinfo=timezone.utc)

    monkeypatch.setattr(cache, "datetime", _FakeDatetime)

    snapshot_id = cache.create_snapshot_id("abc1234")

    assert snapshot_id == "20260418-120000-123456-abc1234"
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}-.{7}", snapshot_id)


def test_create_snapshot_id_distinguishes_same_second_reruns(monkeypatch):
    moments = [
        datetime(2026, 4, 18, 12, 0, 0, 111111, tzinfo=timezone.utc),
        datetime(2026, 4, 18, 12, 0, 0, 222222, tzinfo=timezone.utc),
    ]

    class _FakeDatetime:
        calls = 0

        @staticmethod
        def now(tz):
            moment = moments[_FakeDatetime.calls]
            _FakeDatetime.calls += 1
            return moment

    monkeypatch.setattr(cache, "datetime", _FakeDatetime)

    first = cache.create_snapshot_id("abc1234")
    second = cache.create_snapshot_id("abc1234")

    assert first == "20260418-120000-111111-abc1234"
    assert second == "20260418-120000-222222-abc1234"
    assert first != second


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


def test_initialize_latest_snapshot_resets_latest_tree(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    latest = out_root / "cache" / "latest"
    (latest / "index" / "nested").mkdir(parents=True)
    (latest / "index" / "nested" / "stale.json").write_text("stale")
    (latest / "research").mkdir(parents=True)
    (latest / "research" / "stale.txt").write_text("stale")

    manifest = initialize_latest_snapshot(
        out_root=out_root,
        base_commit="abc1234",
        base_tree="tree1234",
        branch="npuir-dev",
        remote="origin",
        analysis_mode="hybrid",
    )

    assert not (latest / "index" / "nested" / "stale.json").exists()
    assert not (latest / "research" / "stale.txt").exists()
    assert (latest / "index").is_dir()
    assert (latest / "research").is_dir()
    assert (latest / "diffs").is_dir()
    assert (latest / "git").is_dir()
    assert json.loads((latest / "manifest.json").read_text()) == manifest


def test_same_second_reruns_archive_to_distinct_snapshots(monkeypatch, tmp_path: Path):
    moments = [
        datetime(2026, 4, 18, 12, 0, 0, 111111, tzinfo=timezone.utc),
        datetime(2026, 4, 18, 12, 0, 0, 222222, tzinfo=timezone.utc),
    ]

    class _FakeDatetime:
        calls = 0

        @staticmethod
        def now(tz):
            moment = moments[_FakeDatetime.calls]
            _FakeDatetime.calls += 1
            return moment

    monkeypatch.setattr(cache, "datetime", _FakeDatetime)

    out_root = tmp_path / ".agent_pipelines"

    first_manifest = initialize_latest_snapshot(
        out_root=out_root,
        base_commit="abc1234",
        base_tree="tree1234",
        branch="npuir-dev",
        remote="origin",
        analysis_mode="hybrid",
    )
    first_archive = archive_latest_snapshot(out_root)

    second_manifest = initialize_latest_snapshot(
        out_root=out_root,
        base_commit="abc1234",
        base_tree="tree1234",
        branch="npuir-dev",
        remote="origin",
        analysis_mode="hybrid",
    )
    second_archive = archive_latest_snapshot(out_root)

    assert first_manifest["snapshot_id"] == "20260418-120000-111111-abc1234"
    assert second_manifest["snapshot_id"] == "20260418-120000-222222-abc1234"
    assert first_archive.name == first_manifest["snapshot_id"]
    assert second_archive.name == second_manifest["snapshot_id"]
    assert first_archive != second_archive


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
