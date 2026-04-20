from __future__ import annotations

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
        example = item.get("example") or {}
        lines.extend(
            [
                f"- `{flag}`: {item['summary']}",
                "",
                f"## {flag}",
                "",
                item["summary"],
                "",
                f"Test path: `{example.get('test_path', '')}`",
                "",
                "```mlir",
                example.get("input_ir", ""),
                "```",
                "",
            ]
        )

    md_path.write_text("\n".join(lines).rstrip() + "\n")
