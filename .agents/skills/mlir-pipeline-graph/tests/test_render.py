from pathlib import Path

from render import build_site, render_pipeline_page


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
