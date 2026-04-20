import json
from dataclasses import asdict
from pathlib import Path

from models import OpLoweringNode, OpLoweringPage
from render import build_site, render_dialect_page, render_op_page, render_pipeline_page


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
            },
            {
                "flag": "legalize-hivm",
                "summary": "Legalize hivm ops.",
                "key_options": [],
                "dialects_touched": ["hivm"],
                "example": {
                    "test_path": "test/Dialect/HIVM/legalize.mlir",
                    "run_line": "// RUN: bishengir-opt %s -legalize-hivm | FileCheck %s",
                    "input_ir": "func.func @main() { return }",
                    "output_ir": "func.func @main() { return }",
                    "check_lines": ["// CHECK: func.func @main"],
                    "evidence_type": "test",
                },
            },
        ],
        "helpers": [],
    }

    render_pipeline_page(out_root, pipeline)

    md_path = (
        out_root / "reports" / "site_src" / "pipelines" / "convert-to-hivm-pipeline.md"
    )
    dot_path = (
        out_root / "reports" / "site_src" / "public" / "graphs" / "convert-to-hivm-pipeline.dot"
    )

    assert md_path.exists()
    assert dot_path.exists()

    md = md_path.read_text()
    dot = dot_path.read_text()

    assert "# convert-to-hivm-pipeline" in md
    assert "- `convert-to-hivm-op`: Convert ops to hivm." in md
    assert "## convert-to-hivm-op" in md
    assert "Test path: `test/Dialect/HIVM/convert.mlir`" in md
    assert "```mlir" in md
    assert "func.func @main() { return }" in md
    assert "convert-to-hivm-op" in dot
    assert "legalize-hivm" in dot
    assert "pass_1 -> pass_2" in dot
    assert 'URL="../pass-context/convert-to-hivm-op.html"' in dot


def test_render_pipeline_page_renders_explicit_fallback_payload(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    pipeline = {
        "pipeline": "convert-to-hivm-pipeline",
        "passes": [
            {
                "flag": "annotation-lowering",
                "summary": "Erase annotation.mark ops.",
                "key_options": [],
                "dialects_touched": ["annotation"],
                "example": None,
                "example_missing_reason": "No standalone regression test is available yet.",
                "fallback_evidence": {
                    "notes": ["Observed in downstream pipeline trace"],
                },
            }
        ],
        "helpers": [],
    }

    render_pipeline_page(out_root, pipeline)

    md_path = (
        out_root / "reports" / "site_src" / "pipelines" / "convert-to-hivm-pipeline.md"
    )

    md = md_path.read_text()

    assert "Example missing reason: No standalone regression test is available yet." in md
    assert "Fallback evidence:" in md
    assert "Observed in downstream pipeline trace" in md
    assert "Test path:" not in md
    assert "```mlir" not in md


def test_site_template_contains_vitepress_and_pagefind():
    package_json = Path(
        ".agents/skills/mlir-pipeline-graph/site_template/package.json"
    ).read_text()
    config_mts = Path(
        ".agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts"
    ).read_text()

    assert '"vitepress"' in package_json
    assert '"pagefind"' in package_json
    assert 'provider: "local"' in config_mts


def test_site_template_builds_generated_site_src_root():
    package_json = Path(
        ".agents/skills/mlir-pipeline-graph/site_template/package.json"
    ).read_text()
    config_mts = Path(
        ".agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/config.mts"
    ).read_text()

    assert '"build": "vitepress build ."' in package_json
    assert '"pagefind": "pagefind --site .vitepress/dist"' in package_json
    assert 'srcDir: "."' in config_mts


def test_site_template_theme_imports_custom_css():
    theme_index = Path(
        ".agents/skills/mlir-pipeline-graph/site_template/docs/.vitepress/theme/index.ts"
    ).read_text()

    assert 'import "./custom.css"' in theme_index


def test_build_site_invokes_npm_install_build_and_pagefind(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(cmd, cwd, check):
        calls.append((cmd, Path(cwd), check))

    monkeypatch.setattr("subprocess.run", fake_run)

    build_site(tmp_path)

    assert calls == [
        (["npm", "install"], tmp_path, True),
        (["npm", "run", "build"], tmp_path, True),
        (["npm", "run", "pagefind"], tmp_path, True),
    ]


def test_build_site_materializes_vitepress_template(monkeypatch, tmp_path: Path):
    def fake_run(cmd, cwd, check):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)

    build_site(tmp_path)

    assert (tmp_path / "package.json").exists()
    assert (tmp_path / ".vitepress" / "config.mts").exists()
    assert (tmp_path / ".vitepress" / "theme" / "custom.css").exists()
    assert (tmp_path / ".vitepress" / "theme" / "index.ts").exists()
    assert (tmp_path / "index.md").exists()


def test_build_site_preserves_existing_generated_index(monkeypatch, tmp_path: Path):
    def fake_run(cmd, cwd, check):
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    index_path = tmp_path / "index.md"
    index_path.write_text("# Generated Home\n")

    build_site(tmp_path)

    assert index_path.read_text() == "# Generated Home\n"


def test_render_op_page_writes_lowering_chain(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    payload = json.loads(
        Path(
            ".agents/skills/mlir-pipeline-graph/tests/fixtures/sample_op_research.json"
        ).read_text()
    )

    render_op_page(out_root, payload)

    md_path = out_root / "reports" / "site_src" / "ops" / "hfusion.matmul.md"
    dot_path = out_root / "reports" / "site_src" / "public" / "graphs" / "hfusion.matmul.dot"

    assert md_path.exists()
    assert dot_path.exists()

    md = md_path.read_text()
    dot = dot_path.read_text()

    assert "# hfusion.matmul" in md
    assert "Dialect: `hfusion`" in md
    assert "- `convert-to-hivm-pipeline` / `convert-hfusion-to-hivm` / `rewritten` / `implementation`" in md
    assert "- `legalize-hivm-pipeline` / `legalize-hivm` / `lowered` / `test`" in md
    assert "convert-to-hivm-pipeline" in dot
    assert "legalize-hivm-pipeline" in dot
    assert "node_1 -> node_2" in dot
    assert 'URL="../pipelines/convert-to-hivm-pipeline.html"' in dot


def test_render_op_page_accepts_dataclass_serialized_payload(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    payload = asdict(
        OpLoweringPage(
            op="hfusion.matmul",
            dialect="hfusion",
            nodes=[
                OpLoweringNode(
                    pipeline="convert-to-hivm-pipeline",
                    pass_name="convert-hfusion-to-hivm",
                    state="rewritten",
                    evidence_type="implementation",
                )
            ],
        )
    )

    render_op_page(out_root, payload)

    md_path = out_root / "reports" / "site_src" / "ops" / "hfusion.matmul.md"
    md = md_path.read_text()

    assert "- `convert-to-hivm-pipeline` / `convert-hfusion-to-hivm` / `rewritten` / `implementation`" in md


def test_render_dialect_page_writes_aggregate_index(tmp_path: Path):
    out_root = tmp_path / ".agent_pipelines"
    payload = {
        "dialect": "hfusion",
        "ops": [
            {
                "op": "hfusion.matmul",
                "summary": "Lowered through hivm legalization.",
            },
            {
                "op": "hfusion.reduce",
                "summary": "Lowered through reduction decomposition.",
            },
        ],
        "pipelines": ["convert-to-hivm-pipeline", "legalize-hivm-pipeline"],
    }

    render_dialect_page(out_root, payload)

    md_path = out_root / "reports" / "site_src" / "dialects" / "hfusion.md"
    md = md_path.read_text()

    assert "# hfusion" in md
    assert "- `hfusion.matmul`: Lowered through hivm legalization." in md
    assert "- `hfusion.reduce`: Lowered through reduction decomposition." in md
    assert "- `convert-to-hivm-pipeline`" in md
    assert "- `legalize-hivm-pipeline`" in md
