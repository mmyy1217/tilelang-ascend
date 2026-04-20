from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from graphviz_utils import pipeline_dot


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


def build_site(site_root: Path) -> None:
    template_root = Path(__file__).resolve().parent / "site_template"
    docs_root = template_root / "docs"

    site_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_root / "package.json", site_root / "package.json")

    def ignore_index_mdoc(src: str, names: list[str]) -> set[str]:
        if Path(src) == docs_root and (site_root / "index.md").exists():
            return {"index.md"}
        return set()

    shutil.copytree(docs_root, site_root, dirs_exist_ok=True, ignore=ignore_index_mdoc)

    subprocess.run(["npm", "install"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "build"], cwd=site_root, check=True)
    subprocess.run(["npm", "run", "pagefind"], cwd=site_root, check=True)
