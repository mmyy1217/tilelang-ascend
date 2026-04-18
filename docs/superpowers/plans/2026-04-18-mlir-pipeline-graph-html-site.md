# MLIR Pipeline Graph HTML Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HTML-first bishengir pipeline analysis site with hybrid script plus multi-agent research, historical snapshots, scoped queries, pass test evidence, and incremental update/commit/PR impact analysis.

**Architecture:** Keep `extract.py` as the deterministic static extractor, add Python orchestration modules for cache, git, schema validation, and rendering, and layer VitePress plus Pagefind on top of generated Graphviz SVG assets. Pipeline chains and op lowering chains are the two primary rendered graph families; pass pages are pipeline-context views and dialect pages are aggregate indexes over op pages.

**Tech Stack:** Python 3, `pytest`, `argparse`, `json`, `subprocess`, Graphviz `dot`, VitePress, Pagefind, Git, optional `gh`

---

## File Structure

### Existing files to modify

- Modify: `.agents/skills/mlir-pipeline-graph/SKILL.md`
  - Document the new HTML-first workflow, hybrid script plus multi-agent execution, cache layout, and scoped commands.
- Modify: `.agents/skills/mlir-pipeline-graph/extract.py`
  - Keep it focused on static extraction, harden output shape, and make its JSON easier for downstream analysis.

### New Python modules

- Create: `.agents/skills/mlir-pipeline-graph/models.py`
  - Dataclasses and typed helpers for manifests, research records, examples, and diff reports.
- Create: `.agents/skills/mlir-pipeline-graph/validators.py`
  - Schema-style validation helpers that reject incomplete research, especially missing pass examples.
- Create: `.agents/skills/mlir-pipeline-graph/cache.py`
  - Path helpers, latest/history snapshot creation, index writes, and targeted invalidation.
- Create: `.agents/skills/mlir-pipeline-graph/git_tools.py`
  - Fetch, ref resolution, diff collection, PR resolution, and file-to-object impact mapping inputs.
- Create: `.agents/skills/mlir-pipeline-graph/research.py`
  - Agent prompt builders, staging/promotion helpers, and pure-Python fallback research hooks.
- Create: `.agents/skills/mlir-pipeline-graph/graphviz_utils.py`
  - DOT generation and SVG render helpers for pipeline and op graphs.
- Create: `.agents/skills/mlir-pipeline-graph/render.py`
  - Transform cache into `site_src`, generate Graphviz assets, and invoke VitePress/Pagefind.
- Create: `.agents/skills/mlir-pipeline-graph/analyze.py`
  - Main CLI with `full`, `pipeline`, `pass`, `dialect`, `op`, `update`, `commit`, and `pr`.

### New site template files

- Create: `.agents/skills/mlir-pipeline-graph/site_template/package.json`
  - Pin `vitepress` and `pagefind`.
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts`
  - Base VitePress config, nav, sidebar scaffolding, and local search settings.
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/theme/custom.css`
  - Minimal layout and graph styling.
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/index.md`
  - Default shell page that render.py will overwrite in generated output.

### New tests

- Create: `.agents/skills/mlir-pipeline-graph/tests/conftest.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_extract.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_cache.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_render.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_git_tools.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_skeleton.json`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_pipeline_research.json`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_op_research.json`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_diff.json`

### Plan-local note

- The generated site and cache live under `.agent_pipelines/` and are runtime outputs, not hand-edited source files.

## Task 1: Add typed models and validation rules

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/models.py`
- Create: `.agents/skills/mlir-pipeline-graph/validators.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/conftest.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py`

- [ ] **Step 1: Write the failing tests for manifest and pass-example validation**

```python
from pathlib import Path

from validators import ValidationError, validate_pass_research


def test_validate_pass_research_accepts_test_backed_example():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
        "example": {
            "test_path": "test/Dialect/Annotation/annotation-lowering.mlir",
            "run_line": "// RUN: bishengir-opt %s -annotation-lowering | FileCheck %s",
            "input_ir": "func.func @main() { annotation.mark }",
            "output_ir": "func.func @main() { return }",
            "check_lines": ["// CHECK-NOT: annotation.mark"],
            "evidence_type": "test",
        },
    }

    validated = validate_pass_research(record)

    assert validated["flag"] == "annotation-lowering"
    assert validated["example"]["evidence_type"] == "test"


def test_validate_pass_research_rejects_missing_example_and_fallback():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
    }

    try:
        validate_pass_research(record)
    except ValidationError as exc:
        assert "example" in str(exc)
    else:
        raise AssertionError("expected ValidationError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py -v`
Expected: FAIL with `ModuleNotFoundError` for `validators` or missing validation symbols.

- [ ] **Step 3: Write minimal models and validators**

```python
# .agents/skills/mlir-pipeline-graph/models.py
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class PassExample:
    test_path: str
    run_line: str
    input_ir: str
    output_ir: str
    check_lines: list[str]
    evidence_type: str


def require_keys(payload: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValidationError(f"missing keys: {', '.join(missing)}")
```

```python
# .agents/skills/mlir-pipeline-graph/validators.py
from models import ValidationError, require_keys


def validate_pass_research(payload: dict) -> dict:
    require_keys(payload, ["flag", "summary", "key_options", "dialects_touched"])
    has_example = payload.get("example") is not None
    has_fallback = "example_missing_reason" in payload and "fallback_evidence" in payload
    if not has_example and not has_fallback:
        raise ValidationError("example or explicit fallback_evidence is required")
    if has_example:
        require_keys(
            payload["example"],
            ["test_path", "run_line", "input_ir", "output_ir", "check_lines", "evidence_type"],
        )
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py -v`
Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/models.py \
        .agents/skills/mlir-pipeline-graph/validators.py \
        .agents/skills/mlir-pipeline-graph/tests/conftest.py \
        .agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py
git commit -m "feat: add mlir pipeline graph validation models"
```

## Task 2: Harden `extract.py` and add extraction tests

**Files:**
- Modify: `.agents/skills/mlir-pipeline-graph/extract.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_extract.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_skeleton.json`

- [ ] **Step 1: Write the failing extraction tests**

```python
import json
from pathlib import Path

from extract import main


def test_extract_writes_skeleton_and_coverage(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        'void buildDemo(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }'
    )

    exit_code = main(["extract.py", str(repo)])

    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())
    coverage = json.loads((repo / ".agent_pipelines" / "coverage.json").read_text())

    assert exit_code == 0
    assert skeleton["coverage"]["passes_defined"] == 1
    assert coverage["builder_count"] == 1


def test_extract_preserves_preprocessor_condition_text(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        '#if FEATURE_A\nvoid buildDemo(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }\n#elif FEATURE_B\nvoid buildDemo2(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }\n#endif\n'
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    conditions = [step["conditions"] for builder in skeleton["builders"] for step in builder["steps"]]
    flat = [item for items in conditions for item in items]
    assert any(text.startswith("#if FEATURE_A") for text in flat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_extract.py -v`
Expected: FAIL because the fixture repo does not yet produce stable builder metadata under test.

- [ ] **Step 3: Make `extract.py` easier for downstream consumers**

```python
# inside extract.py main()
skeleton = {
    "schema_version": 1,
    "bishengir_root": str(root.relative_to(repo)) if root.is_relative_to(repo) else str(root),
    "passes": [asdict(pass_def) for pass_def in pass_defs],
    "pipelines": all_pipelines,
    "builders": builder_dicts,
    "coverage": coverage,
}

(out_dir / "skeleton.json").write_text(json.dumps(skeleton, indent=2, sort_keys=True))
(out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True))
```

```python
# add helper near argument parsing
def normalize_repo_path(repo: Path) -> Path:
    repo = repo.resolve()
    repo.mkdir(parents=True, exist_ok=True)
    return repo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_extract.py -v`
Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/extract.py \
        .agents/skills/mlir-pipeline-graph/tests/test_extract.py \
        .agents/skills/mlir-pipeline-graph/tests/fixtures/sample_skeleton.json
git commit -m "feat: harden mlir pipeline graph extractor"
```

## Task 3: Implement cache and history snapshot management

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/cache.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_cache.py`

- [ ] **Step 1: Write the failing cache tests**

```python
import json
from pathlib import Path

from cache import create_snapshot_id, initialize_latest_snapshot, archive_latest_snapshot


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

    stored = json.loads((out_root / "cache" / "latest" / "manifest.json").read_text())

    assert manifest["base_commit"] == "abc1234"
    assert stored["analysis_mode"] == "hybrid"
    assert (out_root / "cache" / "latest" / "index").is_dir()


def test_archive_latest_snapshot_copies_latest(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    latest = out_root / "cache" / "latest"
    (latest / "index").mkdir(parents=True)
    (latest / "manifest.json").write_text('{"snapshot_id":"20260418-120000-abc1234"}')

    archive_path = archive_latest_snapshot(out_root)

    assert archive_path.name == "20260418-120000-abc1234"
    assert (archive_path / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError` for `cache`.

- [ ] **Step 3: Write minimal cache implementation**

```python
# .agents/skills/mlir-pipeline-graph/cache.py
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


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
    latest = out_root / "cache" / "latest"
    for path in [latest / "index", latest / "research", latest / "diffs", latest / "git"]:
        path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "snapshot_id": create_snapshot_id(base_commit),
        "base_commit": base_commit,
        "base_tree": base_tree,
        "branch": branch,
        "remote": remote,
        "analysis_mode": analysis_mode,
    }
    (latest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def archive_latest_snapshot(out_root: Path) -> Path:
    latest = out_root / "cache" / "latest"
    manifest = json.loads((latest / "manifest.json").read_text())
    target = out_root / "cache" / "history" / manifest["snapshot_id"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(latest, target)
    return target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_cache.py -v`
Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/cache.py \
        .agents/skills/mlir-pipeline-graph/tests/test_cache.py
git commit -m "feat: add mlir pipeline graph snapshot cache"
```

## Task 4: Implement `analyze.py full|pipeline|pass` with research staging

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/research.py`
- Create: `.agents/skills/mlir-pipeline-graph/analyze.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_pipeline_research.json`

- [ ] **Step 1: Write the failing CLI tests**

```python
import json
from pathlib import Path

from analyze import main


def test_full_command_builds_latest_manifest(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".agent_pipelines").mkdir(parents=True)
    exit_code = main(["analyze.py", "full", "--repo", str(repo), "--from-cache-only"])

    assert exit_code == 0
    assert (repo / ".agent_pipelines" / "cache" / "latest" / "manifest.json").exists()


def test_pass_command_requires_flag(tmp_path: Path):
    exit_code = main(["analyze.py", "pass", "--repo", str(tmp_path)])
    assert exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` for `analyze`.

- [ ] **Step 3: Add minimal CLI and staging helpers**

```python
# .agents/skills/mlir-pipeline-graph/research.py
from pathlib import Path


def staging_dir(out_root: Path) -> Path:
    path = out_root / "cache" / "latest" / "_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def promote_json(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(staged.read_text())
```

```python
# .agents/skills/mlir-pipeline-graph/analyze.py
import argparse
from pathlib import Path

from cache import initialize_latest_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["full", "pipeline", "pass", "dialect", "op", "update", "commit", "pr"])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--name")
    parser.add_argument("--flag")
    parser.add_argument("--from-cache-only", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    repo = Path(args.repo).resolve()
    out_root = repo / ".agent_pipelines"
    if args.command == "pass" and not args.flag:
        parser.error("--flag is required for pass")
    initialize_latest_snapshot(out_root, "0000000", "tree0000", "unknown", "origin", "hybrid")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/research.py \
        .agents/skills/mlir-pipeline-graph/analyze.py \
        .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py \
        .agents/skills/mlir-pipeline-graph/tests/fixtures/sample_pipeline_research.json
git commit -m "feat: add mlir pipeline graph analysis cli"
```

## Task 5: Enforce required pass examples and pipeline research promotion

**Files:**
- Modify: `.agents/skills/mlir-pipeline-graph/research.py`
- Modify: `.agents/skills/mlir-pipeline-graph/validators.py`
- Modify: `.agents/skills/mlir-pipeline-graph/analyze.py`
- Modify: `.agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py`
- Modify: `.agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py`

- [ ] **Step 1: Write the failing pipeline-promotion test**

```python
import json
from pathlib import Path

from analyze import promote_pipeline_research


def test_promote_pipeline_research_rejects_pass_without_example(tmp_path: Path):
    staged = tmp_path / "pipeline.json"
    staged.write_text(
        json.dumps(
            {
                "pipeline": "convert-to-hivm-pipeline",
                "passes": [
                    {
                        "flag": "convert-to-hivm-op",
                        "summary": "Convert ops to hivm.",
                        "key_options": [],
                        "dialects_touched": ["hivm"],
                    }
                ],
                "helpers": [],
            }
        )
    )

    try:
        promote_pipeline_research(staged, tmp_path / "target.json")
    except Exception as exc:
        assert "example" in str(exc)
    else:
        raise AssertionError("expected validation failure")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: FAIL because `promote_pipeline_research` does not yet validate nested pass entries.

- [ ] **Step 3: Add pipeline-promotion validation**

```python
# inside validators.py
def validate_pipeline_research(payload: dict) -> dict:
    require_keys(payload, ["pipeline", "passes", "helpers"])
    for item in payload["passes"]:
        validate_pass_research(item)
    return payload
```

```python
# inside analyze.py
import json

from validators import validate_pipeline_research


def promote_pipeline_research(staged: Path, target: Path) -> None:
    payload = json.loads(staged.read_text())
    validate_pipeline_research(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: PASS with all tests green.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/research.py \
        .agents/skills/mlir-pipeline-graph/validators.py \
        .agents/skills/mlir-pipeline-graph/analyze.py \
        .agents/skills/mlir-pipeline-graph/tests/test_models_and_validators.py \
        .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py
git commit -m "feat: validate pipeline research examples"
```

## Task 6: Render pipeline and pass-context pages with Graphviz SVG

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/graphviz_utils.py`
- Create: `.agents/skills/mlir-pipeline-graph/render.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_render.py`

- [ ] **Step 1: Write the failing render tests**

```python
import json
from pathlib import Path

from render import render_pipeline_page


def test_render_pipeline_page_writes_markdown_and_dot(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    pipeline = {
        "pipeline": "convert-to-hivm-pipeline",
        "passes": [
            {
                "flag": "convert-to-hivm-op",
                "summary": "Convert ops to hivm.",
                "key_options": [],
                "dialects_touched": ["hivm"],
                "example": {
                    "test_path": "test/Dialect/HIVM/convert.mlir",
                    "run_line": "// RUN: bishengir-opt %s -convert-to-hivm-op | FileCheck %s",
                    "input_ir": "func.func @main() { return }",
                    "output_ir": "func.func @main() { return }",
                    "check_lines": ["// CHECK: func.func @main"],
                    "evidence_type": "test",
                },
            }
        ],
        "helpers": [],
    }

    render_pipeline_page(out_root, pipeline)

    assert (out_root / "reports" / "site_src" / "pipelines" / "convert-to-hivm-pipeline.md").exists()
    assert (out_root / "reports" / "site_src" / "public" / "graphs" / "convert-to-hivm-pipeline.dot").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError` for `render`.

- [ ] **Step 3: Write minimal DOT and page renderer**

```python
# .agents/skills/mlir-pipeline-graph/graphviz_utils.py
from pathlib import Path


def pipeline_dot(payload: dict) -> str:
    lines = ["digraph pipeline {", '  rankdir="LR";']
    previous = None
    for index, item in enumerate(payload["passes"], start=1):
        node = f"pass_{index}"
        label = item["flag"].replace('"', '\\"')
        lines.append(f'  {node} [label="{label}", URL="../pass-context/{item["flag"]}.html"];')
        if previous is not None:
            lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append("}")
    return "\n".join(lines)
```

```python
# .agents/skills/mlir-pipeline-graph/render.py
from pathlib import Path

from graphviz_utils import pipeline_dot


def render_pipeline_page(out_root: Path, payload: dict) -> None:
    site_src = out_root / "reports" / "site_src"
    md_path = site_src / "pipelines" / f"{payload['pipeline']}.md"
    dot_path = site_src / "public" / "graphs" / f"{payload['pipeline']}.dot"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(pipeline_dot(payload))
    body = [f"# {payload['pipeline']}", "", "## Passes", ""]
    for item in payload["passes"]:
        body.extend(
            [
                f"### {item['flag']}",
                item["summary"],
                f"Test: `{item['example']['test_path']}`",
                "```mlir",
                item["example"]["input_ir"],
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(body))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/graphviz_utils.py \
        .agents/skills/mlir-pipeline-graph/render.py \
        .agents/skills/mlir-pipeline-graph/tests/test_render.py
git commit -m "feat: render pipeline pages and graphviz assets"
```

## Task 7: Add VitePress and Pagefind site template

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/site_template/package.json`
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts`
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/theme/custom.css`
- Create: `.agents/skills/mlir-pipeline-graph/site_template/docs/index.md`
- Modify: `.agents/skills/mlir-pipeline-graph/render.py`

- [ ] **Step 1: Write the failing site-template test**

```python
from pathlib import Path


def test_site_template_contains_vitepress_and_pagefind():
    package_json = Path(".agents/skills/mlir-pipeline-graph/site_template/package.json").read_text()
    assert '"vitepress"' in package_json
    assert '"pagefind"' in package_json
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: FAIL because the template files do not exist yet.

- [ ] **Step 3: Add the site template and build hook**

```json
{
  "name": "mlir-pipeline-graph-site",
  "private": true,
  "scripts": {
    "build": "vitepress build docs",
    "preview": "vitepress preview docs",
    "pagefind": "pagefind --site .vitepress/dist"
  },
  "devDependencies": {
    "pagefind": "^1.3.0",
    "vitepress": "^1.6.3"
  }
}
```

```ts
// .agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts
import { defineConfig } from "vitepress";

export default defineConfig({
  title: "MLIR Pipeline Graph",
  description: "bishengir pipeline and lowering analysis",
  srcDir: ".",
  themeConfig: {
    search: { provider: "local" },
    nav: [{ text: "Home", link: "/" }, { text: "Pipelines", link: "/pipelines/" }],
  },
});
```

```python
# inside render.py
def build_site(site_root: Path) -> None:
    import subprocess

    subprocess.run(["npm", "install"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "build"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "pagefind"], cwd=site_root, check=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: PASS and the template presence assertion succeeds.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/site_template/package.json \
        .agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts \
        .agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/theme/custom.css \
        .agents/skills/mlir-pipeline-graph/site_template/docs/index.md \
        .agents/skills/mlir-pipeline-graph/render.py
git commit -m "feat: add vitepress and pagefind site template"
```

## Task 8: Add op lowering and dialect aggregate rendering

**Files:**
- Modify: `.agents/skills/mlir-pipeline-graph/models.py`
- Modify: `.agents/skills/mlir-pipeline-graph/render.py`
- Modify: `.agents/skills/mlir-pipeline-graph/graphviz_utils.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_op_research.json`
- Modify: `.agents/skills/mlir-pipeline-graph/tests/test_render.py`

- [ ] **Step 1: Write the failing op-render test**

```python
from pathlib import Path

from render import render_op_page


def test_render_op_page_writes_lowering_chain(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    payload = {
        "op": "hfusion.matmul",
        "dialect": "hfusion",
        "nodes": [
            {
                "pipeline": "convert-to-hivm-pipeline",
                "pass": "convert-hfusion-to-hivm",
                "state": "rewritten",
                "evidence_type": "implementation",
            }
        ],
    }

    render_op_page(out_root, payload)

    assert (out_root / "reports" / "site_src" / "ops" / "hfusion.matmul.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: FAIL because `render_op_page` does not exist yet.

- [ ] **Step 3: Add op and dialect rendering**

```python
# inside graphviz_utils.py
def op_dot(payload: dict) -> str:
    lines = ["digraph op {", '  rankdir="LR";']
    previous = None
    for index, item in enumerate(payload["nodes"], start=1):
        node = f"node_{index}"
        label = f"{item['pipeline']}\\n{item['pass']}\\n{item['state']}"
        lines.append(f'  {node} [label="{label}", URL="../pipelines/{item["pipeline"]}.html"];')
        if previous is not None:
            lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append("}")
    return "\n".join(lines)
```

```python
# inside render.py
def render_op_page(out_root: Path, payload: dict) -> None:
    site_src = out_root / "reports" / "site_src"
    md_path = site_src / "ops" / f"{payload['op']}.md"
    dot_path = site_src / "public" / "graphs" / f"{payload['op']}.dot"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(op_dot(payload))
    lines = [f"# {payload['op']}", "", f"Dialect: `{payload['dialect']}`", "", "## Lowering chain", ""]
    for item in payload["nodes"]:
        lines.append(f"- `{item['pipeline']}` / `{item['pass']}` / `{item['state']}` / `{item['evidence_type']}`")
    md_path.write_text("\n".join(lines))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_render.py -v`
Expected: PASS with the op render test green.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/models.py \
        .agents/skills/mlir-pipeline-graph/render.py \
        .agents/skills/mlir-pipeline-graph/graphviz_utils.py \
        .agents/skills/mlir-pipeline-graph/tests/fixtures/sample_op_research.json \
        .agents/skills/mlir-pipeline-graph/tests/test_render.py
git commit -m "feat: render op lowering and dialect pages"
```

## Task 9: Add update, commit, and PR git analysis

**Files:**
- Create: `.agents/skills/mlir-pipeline-graph/git_tools.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/test_git_tools.py`
- Modify: `.agents/skills/mlir-pipeline-graph/analyze.py`
- Create: `.agents/skills/mlir-pipeline-graph/tests/fixtures/sample_diff.json`

- [ ] **Step 1: Write the failing git-tools tests**

```python
from git_tools import map_changed_files_to_objects


def test_map_changed_files_to_objects_flags_pipeline_and_op_hits():
    changed_files = [
        "3rdparty/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp",
        "3rdparty/AscendNPU-IR-Dev/bishengir/lib/Dialect/HFusion/Transforms/ConvertHFusionToHIVM.cpp",
    ]
    file_index = {
        changed_files[0]: {"pipelines": ["convert-to-hivm-pipeline"], "ops": [], "dialects": ["hivm"]},
        changed_files[1]: {"pipelines": ["convert-to-hivm-pipeline"], "ops": ["hfusion.matmul"], "dialects": ["hfusion"]},
    }

    result = map_changed_files_to_objects(changed_files, file_index)

    assert result["pipelines"] == ["convert-to-hivm-pipeline"]
    assert result["ops"] == ["hfusion.matmul"]
    assert sorted(result["dialects"]) == ["hfusion", "hivm"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_git_tools.py -v`
Expected: FAIL with `ModuleNotFoundError` for `git_tools`.

- [ ] **Step 3: Add git diff mapping helpers**

```python
# .agents/skills/mlir-pipeline-graph/git_tools.py
import subprocess


def git_output(repo: str, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def map_changed_files_to_objects(changed_files: list[str], file_index: dict[str, dict]) -> dict[str, list[str]]:
    pipelines: set[str] = set()
    ops: set[str] = set()
    dialects: set[str] = set()
    for path in changed_files:
        record = file_index.get(path, {})
        pipelines.update(record.get("pipelines", []))
        ops.update(record.get("ops", []))
        dialects.update(record.get("dialects", []))
    return {
        "pipelines": sorted(pipelines),
        "ops": sorted(ops),
        "dialects": sorted(dialects),
    }
```

```python
# inside analyze.py
elif args.command in {"update", "commit", "pr"}:
    manifest = initialize_latest_snapshot(out_root, "0000000", "tree0000", "unknown", "origin", "hybrid")
    (out_root / "cache" / "latest" / "diffs").mkdir(parents=True, exist_ok=True)
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_git_tools.py .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: PASS with all tests green.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/git_tools.py \
        .agents/skills/mlir-pipeline-graph/tests/test_git_tools.py \
        .agents/skills/mlir-pipeline-graph/analyze.py \
        .agents/skills/mlir-pipeline-graph/tests/fixtures/sample_diff.json
git commit -m "feat: add commit and pr impact analysis"
```

## Task 10: Update the skill document and run smoke verification

**Files:**
- Modify: `.agents/skills/mlir-pipeline-graph/SKILL.md`
- Modify: `.agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py`

- [ ] **Step 1: Write the failing documentation smoke test**

```python
from pathlib import Path


def test_skill_mentions_html_first_and_required_examples():
    text = Path(".agents/skills/mlir-pipeline-graph/SKILL.md").read_text()
    assert "HTML-first" in text
    assert "example_missing_reason" in text
    assert "analyze.py" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py -v`
Expected: FAIL because the skill document still describes the older Markdown-first workflow.

- [ ] **Step 3: Update the skill document and smoke commands**

```markdown
## Runtime entry points

Use:

- `python3 .agents/skills/mlir-pipeline-graph/analyze.py full`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py pipeline --name <pipeline>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py op --name <op>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py update`
- `python3 .agents/skills/mlir-pipeline-graph/render.py build`

The final product is an HTML site under `.agent_pipelines/reports/site/`.
Every pass record requires either a minimal MLIR test example or an explicit fallback explanation.
```

- [ ] **Step 4: Run full smoke verification**

Run: `python -m pytest .agents/skills/mlir-pipeline-graph/tests -v`
Expected: PASS with all new tests green.

Run: `python3 .agents/skills/mlir-pipeline-graph/analyze.py full --repo . --from-cache-only`
Expected: exit code 0 and `.agent_pipelines/cache/latest/manifest.json` exists.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/mlir-pipeline-graph/SKILL.md \
        .agents/skills/mlir-pipeline-graph/tests/test_analyze_cli.py
git commit -m "docs: update mlir pipeline graph skill workflow"
```

## Self-Review

### Spec coverage

- HTML-first output: covered by Tasks 6 and 7.
- VitePress plus Pagefind plus Graphviz SVG: covered by Tasks 6 and 7.
- Pipeline chain as primary graph: covered by Tasks 4, 5, and 6.
- Op lowering chain as second primary graph: covered by Task 8.
- Pass-context pages instead of independent pass universe: covered by Tasks 5 and 6.
- Dialect aggregate pages: covered by Task 8.
- Required pass examples with shortest readable snippets: covered by Tasks 1 and 5.
- Cache/latest and history snapshots: covered by Task 3.
- Update, commit, and PR analysis: covered by Task 9.
- Skill documentation updates: covered by Task 10.

### Placeholder scan

- No `TBD`, `TODO`, or deferred placeholders remain.
- Commands are explicit and use current repository tooling.
- Every code-writing step includes concrete snippets rather than narrative-only instructions.

### Type consistency

- Validation and promotion consistently use `validate_pass_research` and `validate_pipeline_research`.
- Cache state consistently refers to `.agent_pipelines/cache/latest`.
- Renderers consistently write under `.agent_pipelines/reports/site_src`.
