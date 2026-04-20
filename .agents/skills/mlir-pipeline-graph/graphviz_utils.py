from __future__ import annotations


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def pipeline_dot(payload: dict) -> str:
    lines = ["digraph pipeline {", '  rankdir="LR";']
    previous_node = None

    for index, item in enumerate(payload.get("passes", []), start=1):
        node_name = f"pass_{index}"
        flag = str(item["flag"])
        label = _dot_escape(flag)
        url = _dot_escape(f"../pass-context/{flag}.html")
        lines.append(
            f'  {node_name} [label="{label}", URL="{url}", target="_top"];'
        )
        if previous_node is not None:
            lines.append(f"  {previous_node} -> {node_name};")
        previous_node = node_name

    lines.append("}")
    return "\n".join(lines)
