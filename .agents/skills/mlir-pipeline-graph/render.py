from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from graphviz_utils import op_dot, pipeline_dot


def render_index_page(out_root: Path) -> None:
    site_src = out_root / "reports" / "site_src"
    lines = [
        "# MLIR Pipeline Graph",
        "",
        "## Pipelines",
        "",
    ]

    pipeline_dir = site_src / "pipelines"
    for path in sorted(pipeline_dir.glob("*.md")):
        lines.append(f"- [{path.stem}](./pipelines/{path.name})")

    lines.extend(["", "## Dialects", "",])
    dialect_dir = site_src / "dialects"
    for path in sorted(dialect_dir.glob("*.md")):
        lines.append(f"- [{path.stem}](./dialects/{path.name})")

    lines.extend(["", "## Ops", "",])
    op_dir = site_src / "ops"
    for path in sorted(op_dir.glob("*.md")):
        lines.append(f"- [{path.stem}](./ops/{path.name})")

    (site_src / "index.md").write_text("\n".join(lines).rstrip() + "\n")


def render_pipeline_page(out_root: Path, payload: dict) -> None:
    site_src = out_root / "reports" / "site_src"
    md_path = site_src / "pipelines" / f"{payload['pipeline']}.md"
    dot_path = site_src / "public" / "graphs" / f"{payload['pipeline']}.dot"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.parent.mkdir(parents=True, exist_ok=True)

    dot_path.write_text(pipeline_dot(payload))

    lines = [
        f"# {payload['pipeline']}",
        "",
        "## Passes",
        "",
    ]

    for item in payload.get("passes", []):
        flag = item["flag"]
        example = item.get("example")
        lines.extend(
            [
                f"- `{flag}`: {item['summary']}",
                "",
                f"## {flag}",
                "",
                item["summary"],
                "",
            ]
        )

        if example is None:
            lines.extend(
                [
                    f"Example missing reason: {item.get('example_missing_reason', '')}",
                    "",
                    "Fallback evidence:",
                    "",
                    "```json",
                    json.dumps(item.get("fallback_evidence", {}), indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
            continue

        lines.extend(
            [
                f"Test path: `{example.get('test_path', '')}`",
                "",
                "```mlir",
                example.get("input_ir", ""),
                "```",
                "",
            ]
        )

    md_path.write_text("\n".join(lines).rstrip() + "\n")


def render_op_page(out_root: Path, payload: dict) -> None:
    site_src = out_root / "reports" / "site_src"
    md_path = site_src / "ops" / f"{payload['op']}.md"
    dot_path = site_src / "public" / "graphs" / f"{payload['op']}.dot"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.parent.mkdir(parents=True, exist_ok=True)

    dot_path.write_text(op_dot(payload))

    lines = [
        f"# {payload['op']}",
        "",
        f"Dialect: `{payload['dialect']}`",
        "",
        "## Nodes",
        "",
    ]

    for item in payload.get("nodes", []):
        pass_name = item.get("pass", item.get("pass_name", ""))
        lines.append(
            f"- `{item['pipeline']}` / `{pass_name}` / `{item['state']}` / `{item['evidence_type']}`"
        )

    md_path.write_text("\n".join(lines).rstrip() + "\n")


def render_dialect_page(out_root: Path, payload: dict) -> None:
    site_src = out_root / "reports" / "site_src"
    md_path = site_src / "dialects" / f"{payload['dialect']}.md"

    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {payload['dialect']}",
        "",
        "## Ops",
        "",
    ]

    for item in payload.get("ops", []):
        lines.append(f"- `{item['op']}`: {item['summary']}")

    lines.extend(["", "## Pipelines", "",])
    for pipeline in payload.get("pipelines", []):
        lines.append(f"- `{pipeline}`")

    md_path.write_text("\n".join(lines).rstrip() + "\n")


def build_site(site_root: Path) -> None:
    template_root = Path(__file__).resolve().parent / "site_template"
    docs_root = template_root / "docs"
    generated_root = site_root.parent / "site_src"

    site_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_root / "package.json", site_root / "package.json")

    def ignore_index_mdoc(src: str, names: list[str]) -> set[str]:
        if Path(src) == docs_root and (site_root / "index.md").exists():
            return {"index.md"}
        return set()

    shutil.copytree(docs_root, site_root, dirs_exist_ok=True, ignore=ignore_index_mdoc)
    if generated_root.exists():
        shutil.copytree(generated_root, site_root, dirs_exist_ok=True)

    subprocess.run(["npm", "install"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "build"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "pagefind"], cwd=site_root, check=True)


def build_site_tree(out_root: Path) -> Path:
    research_root = out_root / "cache" / "latest" / "research"
    site_src = out_root / "reports" / "site_src"
    site_root = out_root / "reports" / "site"

    if site_src.exists():
        shutil.rmtree(site_src)
    if site_root.exists():
        shutil.rmtree(site_root)

    for payload_path in sorted((research_root / "pipelines").glob("*.json")):
        render_pipeline_page(out_root, json.loads(payload_path.read_text()))
    for payload_path in sorted((research_root / "ops").glob("*.json")):
        render_op_page(out_root, json.loads(payload_path.read_text()))
    for payload_path in sorted((research_root / "dialects").glob("*.json")):
        render_dialect_page(out_root, json.loads(payload_path.read_text()))
    render_index_page(out_root)

    build_site(site_root)
    return site_root


def seed_demo_research(out_root: Path) -> None:
    research_root = out_root / "cache" / "latest" / "research"
    pipeline_dir = research_root / "pipelines"
    op_dir = research_root / "ops"
    dialect_dir = research_root / "dialects"

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    op_dir.mkdir(parents=True, exist_ok=True)
    dialect_dir.mkdir(parents=True, exist_ok=True)

    (pipeline_dir / "convert-to-hivm-pipeline.json").write_text(
        json.dumps(
            {
                "pipeline": "convert-to-hivm-pipeline",
                "passes": [
                    {
                        "flag": "convert-to-hivm-op",
                        "summary": "Convert HFusion-style ops into hivm-compatible ops.",
                        "key_options": [],
                        "dialects_touched": ["hfusion", "hivm"],
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
                        "summary": "Legalize hivm ops for downstream execution.",
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
            },
            indent=2,
            sort_keys=True,
        )
    )

    (op_dir / "hfusion.matmul.json").write_text(
        json.dumps(
            {
                "op": "hfusion.matmul",
                "dialect": "hfusion",
                "nodes": [
                    {
                        "pipeline": "convert-to-hivm-pipeline",
                        "pass": "convert-hfusion-to-hivm",
                        "state": "rewritten",
                        "evidence_type": "implementation",
                    },
                    {
                        "pipeline": "convert-to-hivm-pipeline",
                        "pass": "legalize-hivm",
                        "state": "lowered",
                        "evidence_type": "test",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    (dialect_dir / "hfusion.json").write_text(
        json.dumps(
            {
                "dialect": "hfusion",
                "ops": [
                    {
                        "op": "hfusion.matmul",
                        "summary": "Lowered through hivm conversion and legalization.",
                    }
                ],
                "pipelines": ["convert-to-hivm-pipeline"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="render.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--repo", default=".")

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--repo", default=".")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args_list = sys.argv[1:] if argv is None else argv[1:]

    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return int(exc.code)

    repo = Path(args.repo).resolve()
    out_root = repo / ".agent_pipelines"

    if args.command == "build":
        build_site_tree(out_root)
        return 0
    if args.command == "demo":
        seed_demo_research(out_root)
        build_site_tree(out_root)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
