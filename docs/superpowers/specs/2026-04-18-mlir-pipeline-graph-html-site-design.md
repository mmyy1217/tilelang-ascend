---
title: MLIR Pipeline Graph HTML Site and Incremental Analysis Design
date: 2026-04-18
status: proposed
branch: npuir-dev
---

# MLIR Pipeline Graph HTML Site and Incremental Analysis Design

## Overview

Replace the current Markdown-first, partially manual MLIR pipeline graph workflow with an HTML-first analysis and browsing system for `bishengir`. The new system keeps deterministic structure extraction in Python, preserves multi-agent semantic research for hard compiler analysis, and publishes a locally browsable static HTML site with clickable graphs, full-text search, historical snapshots, and incremental update analysis.

The system has two primary graph views:

- `pipeline chain`: one integrated chain that includes pipeline order, pass nodes, helper expansion, cross-calls, and conditional branches
- `op lowering chain`: one per op, showing where the op is matched, rewritten, replaced, or disappears across pipeline execution

`pass` pages are context pages inside pipeline chains, not a separate full-universe graph. `dialect` pages are aggregate index pages over ops and pipelines, not a forced single chain.

## Goals

- Produce an HTML-first browsing experience similar in convenience to Doxygen, but modeled around compiler pipelines and transformations instead of API comments.
- Keep `pipeline + pass + conditions` as one integrated object for analysis and rendering.
- Generate a standalone lowering chain page for each op when requested.
- Support scoped analysis for:
  - one pipeline
  - one pass in pipeline context
  - one dialect as an aggregate view
  - one op lowering chain
- Support incremental workflows after a full analysis:
  - fetch latest changes, report impact, and update the latest snapshot
  - analyze an arbitrary commit, ref, or GitHub PR and map its impact into pipeline/pass/op/dialect structure
- Preserve multi-agent research for expensive semantic understanding such as pass behavior, test evidence, op lowering, and complicated branches.
- Require test-backed evidence for pass semantics whenever possible.
- Make pages and graphs mutually linked and directly clickable.

## Non-Goals

- This system is not an API reference generator for the whole repository.
- This system does not attempt fully precise compiler semantics for every pass without source/test evidence.
- `dialect` pages do not force a single monolithic chain when the underlying semantics are graph-shaped.
- The system does not require long-term artifact publishing outside the local repository, though historical snapshots are retained internally.

## Core Design Decisions

### HTML-first output

The final user-facing output is a static HTML site, not Markdown as the primary product. Markdown may still exist as intermediate content for VitePress page generation, but users browse HTML pages.

### Tooling stack

- Structure extraction: Python
- Semantic analysis and cache orchestration: Python plus multi-agent research
- Graph rendering: Graphviz to SVG
- Site shell: VitePress
- Search: Pagefind
- Optional future global explorer: Cytoscape.js

Graphviz is preferred over Mermaid for primary navigation because SVG links, tooltips, and cluster-level structure are first-class and reliable for a locally served static site.

### Mixed architecture: scripts plus multi-agent research

The system must not collapse into pure software analysis. Deterministic, repeatable work belongs in scripts. Expensive semantic understanding belongs in distributed agents.

Scripts handle:

- static extraction
- cache management
- history snapshots
- git fetch and ref resolution
- commit and PR diff collection
- file-to-object impact mapping
- schema validation
- rendering
- site build and search indexing

Agents handle:

- pass semantics from implementation details
- test evidence selection
- pipeline-specific helper and branch interpretation
- op lowering path tracing
- dialect aggregation where op-level evidence is missing or stale

The orchestration rule is:

`agents produce structured semantic JSON; scripts produce final site artifacts`

## User Workflows

### Full analysis

User intent: build or refresh a full browsable site for the current repository state.

Behavior:

1. Run static extraction.
2. Expand pipeline structure.
3. Dispatch pipeline research agents.
4. Build or refresh cache/latest.
5. Archive the result under history.
6. Render site source.
7. Build the static HTML site.
8. Build Pagefind search index.

### Scoped analysis

User intent: inspect one pipeline, pass, dialect, or op without redoing full research.

Behavior:

- Read `cache/latest` first.
- If the requested object is missing or stale, rerun only the minimum required analysis.
- Re-render only affected pages and shared indexes.

### Incremental update

User intent: fetch the latest state, identify what changed, update the current research, and produce an impact report.

Behavior:

1. Fetch remote state.
2. Compare the latest analyzed commit against the new target commit.
3. Map changed files into affected pipelines, passes, ops, and dialects.
4. Invalidate only impacted research objects.
5. Re-dispatch only the needed agents.
6. Produce an update report and refresh `cache/latest`.
7. Archive the refreshed snapshot under history.

### Commit or PR impact analysis

User intent: inspect what one commit or PR changed and whether it affects the pass pipeline system.

Behavior:

- Resolve the target commit/ref/PR head.
- Compute the diff against the appropriate base.
- Map changed files to structured analysis objects.
- If no relevant object is hit, report explicitly that the change does not affect pass or pipeline analysis.
- If relevant objects are hit, generate an impact report with links into the affected pages.

## Information Model

### Primary object 1: Pipeline chain

A pipeline chain is the integrated graph of:

- registered pipeline entry point
- ordered pass steps
- nested passes
- helper expansion
- cross-pipeline calls
- `runPipeline` transitions
- conditional branches

This is the primary structural view of the compiler.

### Primary object 2: Op lowering chain

An op lowering chain is a per-op graph showing:

- where the op first appears
- which passes match or rewrite it
- whether it is replaced by another op or dialect
- where it disappears
- which steps are confirmed by implementation
- which steps are supported by tests
- which steps remain inference only

This is the primary semantic view for op-centric questions.

### Secondary object: Pass context

A pass context page answers:

- where the pass appears in pipelines
- what its neighbors are
- which conditions gate it
- what implementation symbol defines it
- which minimal MLIR test example explains its behavior

It is not rendered as an independent full graph family.

### Secondary object: Dialect aggregate

A dialect page aggregates:

- related ops
- links to op lowering pages
- related pipelines
- related passes
- related files and tests

It acts as an index and summary page.

## Filesystem Layout

The persistent analysis lives under `.agent_pipelines/`.

```text
.agent_pipelines/
├── cache/
│   ├── latest/
│   │   ├── manifest.json
│   │   ├── skeleton.json
│   │   ├── coverage.json
│   │   ├── expanded.json
│   │   ├── index/
│   │   │   ├── pipelines.json
│   │   │   ├── passes.json
│   │   │   ├── ops.json
│   │   │   ├── dialects.json
│   │   │   ├── files.json
│   │   │   └── conditions.json
│   │   ├── research/
│   │   │   ├── pipelines/<pipeline>.json
│   │   │   ├── passes/<flag>.json
│   │   │   ├── ops/<op>.json
│   │   │   └── dialects/<dialect>.json
│   │   ├── diffs/<run-id>.json
│   │   └── git/
│   │       ├── refs.json
│   │       └── last_fetch.json
│   └── history/
│       └── <snapshot-id>/
│           └── ...
└── reports/
    ├── site_src/
    │   ├── index.md
    │   ├── pipelines/
    │   ├── pass-context/
    │   ├── ops/
    │   ├── dialects/
    │   ├── diffs/
    │   ├── public/
    │   │   ├── graphs/
    │   │   └── data/
    │   └── .vitepress/
    └── site/
        └── ...
```

`latest/` is mutable and represents the current working snapshot. `history/<snapshot-id>/` is immutable. A snapshot id uses `YYYYMMDD-HHMMSS-<shortsha>`.

## Python Components

### `extract.py`

Responsibility:

- parse `Passes.td`
- locate pipeline builders and registrations
- extract pass steps, helper calls, cross-calls, and conditions
- emit `skeleton.json` and `coverage.json`

This script remains the low-level static extractor and does not own semantic research, git analysis, or site generation.

### `analyze.py`

Responsibility:

- orchestrate extraction, cache, history, invalidation, and agent dispatch
- build expanded structural indexes
- stage and validate semantic research
- map commit/PR diffs to analysis objects

Proposed subcommands:

- `full`
- `pipeline`
- `pass`
- `dialect`
- `op`
- `update`
- `commit`
- `pr`

### `render.py`

Responsibility:

- transform cached analysis into VitePress-ready page sources
- generate Graphviz dot files and SVG assets
- build inter-page links
- generate site summaries and index pages

### Site build step

Responsibility:

- run VitePress build
- run Pagefind index generation
- produce the final static site in `reports/site/`

## Multi-Agent Research Model

### Dispatch strategy

`full` mode:

- script builds the static skeleton and expanded pipeline structure
- one research agent per pipeline
- optional op agents only when requested or when eager mode is enabled

Scoped modes:

- `pipeline` and `pass` dispatch only the owning pipeline agent if cache is stale
- `op` dispatches a dedicated op-lowering agent
- `dialect` dispatches only missing op agents under that dialect

Diff modes:

- script identifies impacted objects first
- only impacted pipeline/op/dialect agents are dispatched

### Output contract

Agents write structured JSON into staging locations. The main process validates schema and promotes only valid results into `cache/latest/research/`.

Agents never write final HTML, Graphviz, or VitePress files directly.

## Required Research Schemas

### Pipeline research schema

Each `research/pipelines/<pipeline>.json` must contain:

- pipeline identity and registration metadata
- integrated ordered chain
- per-step conditions
- helper and cross-call notes
- pass semantics
- evidence blocks

### Pass evidence schema

Each pass entry must contain a test-backed example unless no such example can be found. This is a hard requirement.

Required fields:

```json
{
  "flag": "annotation-lowering",
  "summary": "Removes annotation.mark operations and lowers the annotation dialect.",
  "key_options": [],
  "dialects_touched": ["annotation"],
  "example": {
    "test_path": "test/Dialect/Annotation/annotation-lowering.mlir",
    "run_line": "// RUN: bishengir-opt %s -annotation-lowering | FileCheck %s",
    "input_ir": "%0 = ...",
    "output_ir": "...",
    "check_lines": ["// CHECK-NOT: annotation.mark"],
    "evidence_type": "test"
  }
}
```

Fallback is allowed only with explicit explanation:

```json
{
  "example": null,
  "example_missing_reason": "No dedicated or minimally attributable test found.",
  "fallback_evidence": {
    "type": "implementation",
    "source_file": ".../SomePass.cpp",
    "source_symbol": "runOnOperation"
  }
}
```

### Example selection policy

Default example policy favors the shortest readable slice:

- target length: 10 to 30 lines
- prefer `before_ir + check_lines` over full-file reproduction
- always include a link to the source test file
- allow at most one `secondary_example` when a single slice is insufficient

### Op research schema

Each `research/ops/<op>.json` must contain:

- op identity
- owning dialect
- chain nodes
- each node's associated pipeline/pass
- evidence source for each node: `implementation`, `test`, or `inference`
- terminal state: `survives`, `rewritten`, `replaced`, or `disappears`

### Dialect research schema

Each `research/dialects/<dialect>.json` must contain:

- dialect identity
- related ops
- links to op pages
- related pipelines
- related passes
- impacted files and tests

## Incremental Invalidation Rules

### Structural invalidation

Changes to any of the following require rerunning extraction and invalidating affected pipeline chains:

- `Passes.td`
- pipeline builder sources
- `runPipeline` sites
- helper expansion logic
- conditional branch logic in pipeline code

Affected op and dialect pages depending on those pipelines are then invalidated.

### Semantic invalidation

Changes to pass implementation files invalidate:

- the pipelines that reference those passes
- op pages whose lowering chains traverse those passes
- dialect aggregates built from those ops

### Evidence invalidation

Changes to tests invalidate:

- example snippets
- evidence blocks
- pass context examples

They do not invalidate structural graphs by themselves.

### Render-only invalidation

Changes to render templates, Graphviz styles, VitePress layout, or Pagefind wiring invalidate only rendered site artifacts, not semantic research.

## Git, Commit, and PR Analysis

### Update mode

`update` performs:

1. remote fetch
2. target ref resolution
3. comparison against `cache/latest/manifest.json`
4. file-to-object impact mapping
5. targeted invalidation and refresh
6. report generation

### Commit mode

`commit --ref <sha|ref>` performs:

- diff collection for the target commit
- impacted object mapping
- a report that states either:
  - affected pipelines/passes/ops/dialects, or
  - explicitly that the commit does not affect pass or pipeline structure

### PR mode

`pr --id` or `pr --url` performs:

- PR ref resolution
- remote fetch
- diff collection against the PR base
- impacted object mapping
- report generation

Preferred mechanisms:

- use `gh` when available for PR resolution and metadata
- fall back to git ref fetch and local comparison when `gh` is unavailable

## HTML Site Architecture

### Homepage

Contents:

- current snapshot summary
- recent update/commit/PR reports
- pipeline index
- dialect index
- op entry point and search
- global navigation links

### Pipeline page

This is the primary chain page.

Contents:

- clickable Graphviz SVG
- ordered chain with conditions
- helper expansion and cross-calls
- pass summaries
- per-pass minimal test example
- links to pass-context pages

### Pass-context page

Contents:

- occurrences across pipelines
- neighboring steps
- conditions
- implementation symbol
- minimal MLIR test example

### Op page

This is the second primary chain page.

Contents:

- clickable lowering chain SVG
- step-by-step transformation path
- evidence labels per step
- implementation and test evidence
- inferred steps clearly marked

### Dialect page

Contents:

- related ops
- links to op pages
- related pipelines and passes
- file and test index

### Diff pages

Contents:

- changed files
- affected structured objects
- explicit non-impact statement when appropriate
- links to affected pages

## Linking and Navigation Rules

Every page must contain:

- breadcrumbs
- a stable canonical URL path
- standard HTML links in the body
- graph links where applicable
- back-links to related objects

The system must not rely on graph clickability alone. Every graph-linked object must also have a normal text link on the page.

## Search

Pagefind indexes the built site after VitePress output is produced.

Search must support at least:

- pipeline names
- pass flags
- dialect names
- op names
- commit or PR report titles
- file paths appearing in evidence

## CLI Contract

Primary entry points:

```bash
python3 .agents/skills/mlir-pipeline-graph/analyze.py full
python3 .agents/skills/mlir-pipeline-graph/analyze.py pipeline --name <pipeline>
python3 .agents/skills/mlir-pipeline-graph/analyze.py pass --flag <pass-flag>
python3 .agents/skills/mlir-pipeline-graph/analyze.py dialect --name <dialect>
python3 .agents/skills/mlir-pipeline-graph/analyze.py op --name <op>
python3 .agents/skills/mlir-pipeline-graph/analyze.py update
python3 .agents/skills/mlir-pipeline-graph/analyze.py commit --ref <sha-or-ref>
python3 .agents/skills/mlir-pipeline-graph/analyze.py pr --id <pr>
python3 .agents/skills/mlir-pipeline-graph/analyze.py pr --url <url>
python3 .agents/skills/mlir-pipeline-graph/render.py build
```

Common options:

- `--snapshot <id>`
- `--refresh`
- `--from-cache-only`
- `--no-render`
- `--mode auto|script-only|hybrid|agents-heavy`
- `--eager-op-pages`
- `--format html|json`

Default mode is `hybrid`.

## Verification Strategy

Implementation must follow TDD:

- unit-test extraction enhancements
- unit-test cache invalidation rules
- unit-test file-to-object mapping
- integration-test Graphviz/VitePress/Pagefind build pipeline
- schema-test agent JSON outputs
- smoke-test scoped rebuilds

Verification for semantic research must include:

- schema validation
- explicit rejection of missing pass example evidence
- re-dispatch on malformed or incomplete agent output

## Rollout Plan

### Phase 1

- keep `extract.py`
- add expanded structure generation
- add cache/latest and history layout

### Phase 2

- add `analyze.py full|pipeline|pass`
- add pipeline research schema with required pass examples
- add Graphviz pipeline rendering

### Phase 3

- add op lowering research and rendering
- add dialect aggregate pages

### Phase 4

- add `update|commit|pr`
- add targeted invalidation and historical snapshots

### Phase 5

- add VitePress shell
- add Pagefind indexing
- add optional global explorer hook points

## Risks and Mitigations

### Risk: agent output drift

Mitigation:

- strict JSON schemas
- promotion only after validation
- fallback redispatch instead of manual patch-up

### Risk: op lowering over-inference

Mitigation:

- label every chain edge with evidence type
- separate implementation-backed, test-backed, and inferred transitions

### Risk: large cost for full op generation

Mitigation:

- default op pages are on-demand
- optional eager generation remains explicit

### Risk: git/PR resolution variability

Mitigation:

- prefer `gh`
- provide pure-git fallback
- persist resolved refs in cache/git metadata

## Decision Summary

- HTML-first output
- VitePress as the site shell
- Graphviz SVG as the primary graph renderer
- Pagefind for static-site search
- Python plus multi-agent hybrid architecture
- pipeline chain and op lowering chain as the two primary graph families
- pass examples with minimal MLIR test snippets as a hard requirement
- historical snapshots retained internally under `.agent_pipelines/cache/history/`
