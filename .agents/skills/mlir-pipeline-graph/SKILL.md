---
name: mlir-pipeline-graph
description: Use when the user asks for bishengir pipeline graphs, pass ordering, op lowering chains, dialect aggregation, pass-example evidence, or update/commit/PR impact analysis.
---

# MLIR Pipeline Graph

This skill is now **HTML-first**. The source of truth is the structured cache under `.agent_pipelines/`, and rendered site content is generated from that cache into VitePress-ready page sources.

## Runtime Entry Points

Use the current repo-local entry points:

- `python3 .agents/skills/mlir-pipeline-graph/analyze.py full --repo <repo> --from-cache-only`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py pipeline --repo <repo> --name <pipeline>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py dialect --repo <repo> --name <dialect>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py op --repo <repo> --name <op>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py pass --repo <repo> --flag <pass-flag>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py update --repo <repo>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py commit --repo <repo> --ref <git-ref>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py pr --repo <repo> --id <pr-number>`
- `python3 .agents/skills/mlir-pipeline-graph/analyze.py pr --repo <repo> --url <pr-url>`

`render.py` is currently a library module, not a standalone CLI. The build helper is `build_site(site_root)`, and the page helpers are `render_pipeline_page(...)`, `render_op_page(...)`, and `render_dialect_page(...)`.

## Output Layout

The generated state lives under `.agent_pipelines/`:

```text
.agent_pipelines/
├── cache/
│   └── latest/
│       ├── manifest.json
│       ├── index/files.json
│       ├── research/
│       └── diffs/
└── reports/
    └── site_src/
        ├── pipelines/
        ├── ops/
        ├── dialects/
        └── public/graphs/
```

`analyze.py full` creates the latest snapshot manifest. `update`, `commit`, and `pr` write impact reports under `.agent_pipelines/cache/latest/diffs/`.

## Research Contract

Pipeline research is cache-backed JSON, not free-form prose.

Every pass record must include one of:

- a minimal MLIR test-backed `example`
- `example: null` plus `example_missing_reason` and fallback evidence

Do not silently omit missing examples. The explicit `example_missing_reason` field is required when no attributable test snippet exists.

## Rendered Views

Primary rendered views:

- pipeline chain pages under `.agent_pipelines/reports/site_src/pipelines/`
- op lowering chain pages under `.agent_pipelines/reports/site_src/ops/`
- dialect aggregate pages under `.agent_pipelines/reports/site_src/dialects/`

Graph assets are DOT files under `.agent_pipelines/reports/site_src/public/graphs/`.

## Practical Workflow

1. Run `extract.py` only for static skeleton extraction.
2. Use `analyze.py` for cache-oriented commands and scoped queries.
3. Promote only validated research into `cache/latest/research/`.
4. Render pages from structured payloads, not by hand-writing Markdown.
5. Treat `update`, `commit`, and `pr` as impact-analysis entry points that map changed files to pipelines, ops, and dialects.

## Smoke Verification

Before claiming the skill works, run:

```bash
python -m pytest .agents/skills/mlir-pipeline-graph/tests -v
python3 .agents/skills/mlir-pipeline-graph/analyze.py full --repo . --from-cache-only
```

The final product is an HTML-oriented documentation site assembled from `.agent_pipelines/reports/site_src/`, with VitePress/Pagefind template assets materialized at build time.
