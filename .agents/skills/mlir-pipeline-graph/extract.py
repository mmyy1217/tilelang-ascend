#!/usr/bin/env python3
"""Static skeleton extractor for bishengir MLIR pass pipelines.

Walks bishengir/include/**/Passes.td for the pass catalog, then walks
bishengir/lib/**/*.cpp for pipeline builders, capturing every addPass,
nested addPass, helper call, cross-pipeline call, and conditional stack.

By default this emits `.agent_pipelines/skeleton.json` and
`.agent_pipelines/coverage.json` under the repo root. Callers can override
the output directory with `--out-dir`.

Pure stdlib. No clang. Brace tracking is intentionally lightweight.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


CANDIDATE_ROOTS = [
    "3rdparty/AscendNPU-IR-Dev/bishengir",
    "3rdparty/AscendNPU-IR/bishengir",
]


def find_bishengir_root(repo: Path) -> Path:
    for candidate in CANDIDATE_ROOTS:
        root = repo / candidate
        if (root / "include").exists() and (root / "lib").exists():
            return root
    raise SystemExit(
        f"Could not find a bishengir tree under {repo}. Tried: {CANDIDATE_ROOTS}"
    )


def resolve_bishengir_root(repo: Path, explicit_root: Optional[Path]) -> Path:
    if explicit_root is None:
        return find_bishengir_root(repo)
    root = explicit_root if explicit_root.is_absolute() else repo / explicit_root
    if (root / "include").exists() and (root / "lib").exists():
        return root
    raise SystemExit(
        f"Explicit bishengir root {root} does not contain include/ and lib/."
    )


def resolve_out_dir(repo: Path, explicit_out_dir: Optional[Path]) -> Path:
    if explicit_out_dir is None:
        return repo / ".agent_pipelines"
    return explicit_out_dir if explicit_out_dir.is_absolute() else repo / explicit_out_dir


def normalize_repo_path(repo: Path) -> Path:
    repo = repo.resolve()
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def strip_comments(text: str) -> str:
    """Strip // line and /* ... */ block comments while preserving newlines."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue
        if c == '"':
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def offset_of_line(text: str, line: int) -> int:
    if line <= 1:
        return 0
    offset = 0
    for _ in range(line - 1):
        nl = text.find("\n", offset)
        if nl < 0:
            return len(text)
        offset = nl + 1
    return offset


PASS_DEF_RE = re.compile(
    r"\bdef\s+(\w+)\s*:\s*Pass<\s*\"([^\"]+)\"\s*(?:,\s*\"([^\"]*)\")?\s*>",
)
LET_STR_RE = re.compile(r"let\s+(\w+)\s*=\s*\"([^\"]*)\"\s*;")
LET_DESC_RE = re.compile(r"let\s+description\s*=\s*\[\{(.*?)\}\]\s*;", re.DOTALL)
LET_LIST_RE = re.compile(r"let\s+(\w+)\s*=\s*\[(.*?)\]\s*;", re.DOTALL)
OPTION_RE = re.compile(
    r"\bOption<\s*\"(\w+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"",
)
LIST_OPTION_RE = re.compile(
    r"\bListOption<\s*\"(\w+)\"\s*,\s*\"([^\"]+)\"\s*,\s*\"([^\"]+)\"",
)
CONSTRUCTOR_FN_RE = re.compile(r"(?:::)?(?:\w+::)*?(create\w+)\s*\(")


@dataclass
class PassDef:
    cls: str
    flag: str
    op: str
    summary: str = ""
    description: str = ""
    constructor_fn: Optional[str] = None
    constructor_full: Optional[str] = None
    constructor_namespace: Optional[str] = None
    options: list[dict] = field(default_factory=list)
    dependent_dialects: list[str] = field(default_factory=list)
    td_file: str = ""
    td_line: int = 0


def parse_passes_td(td_path: Path) -> list[PassDef]:
    raw = td_path.read_text(errors="replace")
    text = strip_comments(raw)
    out: list[PassDef] = []
    for match in PASS_DEF_RE.finditer(text):
        cls, flag, op = match.group(1), match.group(2), (match.group(3) or "")
        body_start = text.find("{", match.end())
        if body_start < 0:
            continue
        depth = 0
        i = body_start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[body_start + 1 : i]

        pass_def = PassDef(
            cls=cls,
            flag=flag,
            op=op,
            td_file=str(td_path),
            td_line=line_of(text, match.start()),
        )

        for summary_match in LET_STR_RE.finditer(body):
            if summary_match.group(1) == "summary":
                pass_def.summary = summary_match.group(2).strip()
            elif summary_match.group(1) == "constructor":
                constructor_match = CONSTRUCTOR_FN_RE.search(summary_match.group(2))
                if constructor_match:
                    pass_def.constructor_fn = constructor_match.group(1)
                    full = summary_match.group(2).strip()
                    full_no_parens = full.split("(")[0].lstrip(":")
                    pass_def.constructor_full = full_no_parens
                    parts = full_no_parens.split("::")
                    if len(parts) >= 2:
                        pass_def.constructor_namespace = parts[-2]

        description_match = LET_DESC_RE.search(body)
        if description_match:
            pass_def.description = description_match.group(1).strip()

        for list_match in LET_LIST_RE.finditer(body):
            name, contents = list_match.group(1), list_match.group(2)
            if name == "dependentDialects":
                pass_def.dependent_dialects = [
                    item.strip().strip('"')
                    for item in contents.split(",")
                    if item.strip().strip('"')
                ]
            elif name == "options":
                for option_match in OPTION_RE.finditer(contents):
                    pass_def.options.append(
                        {
                            "name": option_match.group(1),
                            "flag": option_match.group(2),
                            "type": option_match.group(3),
                        }
                    )
                for option_match in LIST_OPTION_RE.finditer(contents):
                    pass_def.options.append(
                        {
                            "name": option_match.group(1),
                            "flag": option_match.group(2),
                            "type": option_match.group(3),
                            "list": True,
                        }
                    )

        out.append(pass_def)
    return out


FUNC_DEF_RE = re.compile(
    r"""(?mx)
    ^
    (?:static\s+)?
    (?:inline\s+)?
    (?:llvm::)?(?:void|LogicalResult)\s+
    (?:[\w:]+::)?
    (?P<name>\w+)
    \s*\(
    (?P<sig>[^){]*)
    \)
    \s*\{
    """
)

PASS_CALL_RE = re.compile(
    r"""(?x)
    \b(?P<pm>pm|passManager)\b
    (?:\.nest<\s*(?P<nest>[^>]+?)\s*>\(\))?
    \.addPass\(\s*
    (?:::)?(?:[\w:]+::)*?(?P<ctor>create\w+)
    \s*\(
    """
)

HELPER_CALL_RE = re.compile(
    r"""(?x)
    \b(?:(?P<ns>[\w:]+)::)?
    (?P<fn>\w+)
    \s*\(\s*
    (?P<arg0>[\w&]+)
    \s*[,)]
    """
)

RUN_PIPELINE_RE = re.compile(
    r"""(?x)
    \brunPipeline\s*\(\s*
    \w+\s*,\s*
    (?P<builder>\w+)\s*,
    """
)

PIPELINE_REG_PREFIX_RE = re.compile(
    r"""(?x)
    \b(?:mlir::)?PassPipelineRegistration\s*<\s*(?P<opts>[^>]*)\s*>\s*(?=\()
    """
)


@dataclass
class Step:
    kind: str
    conditions: list[str] = field(default_factory=list)
    line: int = 0
    constructor_fn: Optional[str] = None
    nested_op: Optional[str] = None
    flag: Optional[str] = None
    target: Optional[str] = None
    target_namespace: Optional[str] = None
    label: Optional[str] = None


@dataclass
class Builder:
    name: str
    file: str
    line: int
    steps: list[Step] = field(default_factory=list)
    is_pipeline: bool = False
    pipeline_name: Optional[str] = None
    pipeline_desc: Optional[str] = None
    options_class: Optional[str] = None


def find_function_bodies(text: str) -> list[tuple[str, int, int, int]]:
    out = []
    for match in FUNC_DEF_RE.finditer(text):
        if "PassManager" not in match.group("sig"):
            continue
        body_open = text.rfind("{", match.start(), match.end())
        if body_open < 0:
            continue
        depth = 1
        i = body_open + 1
        in_str = False
        while i < len(text) and depth > 0:
            c = text[i]
            if in_str:
                if c == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
            i += 1
        if depth != 0:
            continue
        out.append((match.group("name"), match.start(), body_open, i - 1))
    return out


def preprocessor_conditions_at(text: str, limit: int) -> list[str]:
    """Return active top-level preprocessor directive lines at byte offset limit."""
    conditions: list[str] = []
    branch_stack: list[str] = []
    brace_depth = 0
    in_str = False
    i = 0

    while i < limit:
        c = text[i]
        if in_str:
            if c == "\\" and i + 1 < limit:
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue

        if c == '"':
            in_str = True
        elif c == "{":
            brace_depth += 1
        elif c == "}":
            brace_depth = max(0, brace_depth - 1)
        elif (
            c == "#"
            and brace_depth == 0
            and (i == 0 or text[i - 1] == "\n")
        ):
            line_end = text.find("\n", i, limit)
            if line_end < 0:
                line_end = limit
            line = text[i:line_end].strip()
            if line.startswith("#if"):
                branch_stack.append(line)
            elif line.startswith("#elif") and branch_stack:
                branch_stack[-1] = line
            elif line.startswith("#else") and branch_stack:
                branch_stack[-1] = line
            elif line.startswith("#endif") and branch_stack:
                branch_stack.pop()
            i = line_end
        i += 1

    return list(branch_stack)


def _join_adjacent_strings(text: str, start: int, stop: int) -> tuple[str, int]:
    i = start
    while i < stop and text[i] in " \t\r\n":
        i += 1
    if i >= stop or text[i] != '"':
        return "", start
    parts = []
    while i < stop and text[i] == '"':
        j = i + 1
        while j < stop and text[j] != '"':
            if text[j] == "\\" and j + 1 < stop:
                j += 2
                continue
            j += 1
        if j >= stop:
            break
        parts.append(text[i + 1 : j])
        i = j + 1
        while i < stop and text[i] in " \t\r\n":
            i += 1
    return "".join(parts), i


def find_pipeline_registrations(text: str) -> list[dict]:
    out = []
    for match in PIPELINE_REG_PREFIX_RE.finditer(text):
        call_open = text.find("(", match.end())
        if call_open < 0:
            continue
        call_close = BodyWalker._balanced_paren(text, call_open)
        if call_close < 0:
            continue
        name, pos = _join_adjacent_strings(text, call_open + 1, call_close)
        if not name:
            continue
        while pos < call_close and text[pos] in " \t\r\n":
            pos += 1
        if pos < call_close and text[pos] == ",":
            pos += 1
        desc, pos = _join_adjacent_strings(text, pos, call_close)
        while pos < call_close and text[pos] in " \t\r\n":
            pos += 1
        if pos < call_close and text[pos] == ",":
            pos += 1

        i = pos
        paren_depth = 1
        body_open = -1
        while i < len(text) and paren_depth > 0:
            c = text[i]
            if c == "(":
                paren_depth += 1
            elif c == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    break
            elif c == "{" and body_open < 0:
                body_open = i
                depth = 1
                j = i + 1
                in_str = False
                while j < len(text) and depth > 0:
                    cc = text[j]
                    if in_str:
                        if cc == "\\" and j + 1 < len(text):
                            j += 2
                            continue
                        if cc == '"':
                            in_str = False
                    else:
                        if cc == '"':
                            in_str = True
                        elif cc == "{":
                            depth += 1
                        elif cc == "}":
                            depth -= 1
                    j += 1
                i = j
                continue
            i += 1

        if body_open < 0:
            entry_fn = None
            tail = text[pos:call_close].strip().rstrip(",").strip()
            if tail:
                entry_fn = re.sub(r".*::", "", tail).strip()
                fn_match = re.match(r"\w+", entry_fn)
                entry_fn = fn_match.group(0) if fn_match else None
            out.append(
                {
                    "name": name,
                    "desc": desc,
                    "options_class": match.group("opts").strip(),
                    "body_open": -1,
                    "body_close": -1,
                    "entry_builder": entry_fn,
                    "line": line_of(text, match.start()),
                }
            )
            continue

        out.append(
            {
                "name": name,
                "desc": desc,
                "options_class": match.group("opts").strip(),
                "body_open": body_open,
                "body_close": i - 1,
                "line": line_of(text, match.start()),
            }
        )
    return out


class BodyWalker:
    def __init__(self, text: str, start: int, end: int, helper_names: set[str]):
        self.text = text
        self.start = start
        self.end = end
        self.helper_names = helper_names
        self.steps: list[Step] = []

    @staticmethod
    def _balanced_paren(text: str, open_idx: int) -> int:
        depth = 1
        i = open_idx + 1
        in_str = False
        while i < len(text) and depth > 0:
            c = text[i]
            if in_str:
                if c == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1

    @staticmethod
    def _matched_brace(text: str, open_idx: int) -> int:
        depth = 1
        i = open_idx + 1
        in_str = False
        while i < len(text) and depth > 0:
            c = text[i]
            if in_str:
                if c == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1

    def walk(self, conditions: Optional[list[str]] = None) -> None:
        if conditions is None:
            conditions = []
        i = self.start
        while i < self.end:
            c = self.text[i]
            if c.isspace() or c in "{};":
                i += 1
                continue
            if c == "#":
                j = self.text.find("\n", i)
                if j < 0:
                    j = self.end
                line = self.text[i:j]
                while line.rstrip().endswith("\\") and j < self.end:
                    nxt = self.text.find("\n", j + 1)
                    if nxt < 0:
                        nxt = self.end
                    line += self.text[j:nxt]
                    j = nxt
                line = line.strip()
                if line.startswith("#if"):
                    body_start = j + 1
                    body_end = self._find_pp_close(self.text, body_start, self.end)
                    branches = self._split_pp_branches(
                        self.text, body_start, body_end, line
                    )
                    for branch_start, branch_end, branch_directive in branches:
                        sub = BodyWalker(self.text, branch_start, branch_end, self.helper_names)
                        sub.walk(conditions + [branch_directive])
                        self.steps.extend(sub.steps)
                    i = body_end + 1
                    nl = self.text.find("\n", i)
                    i = nl + 1 if nl >= 0 else self.end
                    continue
                i = j + 1
                continue
            if self.text.startswith("if", i) and (i + 2 < self.end) and self.text[i + 2] in " \t\n(":
                i = self._walk_if_chain(i, conditions)
                continue

            pass_match = PASS_CALL_RE.match(self.text, i)
            if pass_match:
                self.steps.append(
                    Step(
                        kind="nested_pass" if pass_match.group("nest") else "pass",
                        conditions=list(conditions),
                        line=line_of(self.text, pass_match.start()),
                        constructor_fn=pass_match.group("ctor"),
                        nested_op=pass_match.group("nest").strip() if pass_match.group("nest") else None,
                    )
                )
                op = self.text.find("(", pass_match.end() - 1)
                cl = self._balanced_paren(self.text, op) if op >= 0 else -1
                i = cl + 1 if cl > 0 else pass_match.end()
                continue

            run_match = RUN_PIPELINE_RE.match(self.text, i)
            if run_match:
                op = self.text.find("(", i)
                cl = self._balanced_paren(self.text, op) if op >= 0 else -1
                label = None
                if cl > 0:
                    args = self.text[op + 1 : cl]
                    parts = [part.strip() for part in args.split(",")]
                    if len(parts) >= 4 and parts[3].startswith('"'):
                        label = parts[3].strip('"')
                self.steps.append(
                    Step(
                        kind="run_pipeline",
                        conditions=list(conditions),
                        line=line_of(self.text, run_match.start()),
                        target=run_match.group("builder"),
                        label=label,
                    )
                )
                i = cl + 1 if cl > 0 else run_match.end()
                continue

            helper_match = HELPER_CALL_RE.match(self.text, i)
            if helper_match:
                fn = helper_match.group("fn")
                ns = helper_match.group("ns")
                arg0 = helper_match.group("arg0")
                if (
                    (arg0 in ("pm", "passManager") or fn in self.helper_names)
                    and fn
                    not in (
                        "if",
                        "while",
                        "for",
                        "switch",
                        "return",
                        "sizeof",
                        "static_cast",
                        "reinterpret_cast",
                        "dyn_cast",
                        "succeeded",
                        "failed",
                        "isa",
                    )
                ):
                    kind = "helper_call" if fn in self.helper_names and not ns else "cross_call"
                    self.steps.append(
                        Step(
                            kind=kind,
                            conditions=list(conditions),
                            line=line_of(self.text, helper_match.start()),
                            target=fn,
                            target_namespace=ns,
                        )
                    )
                    op = self.text.find("(", i)
                    cl = self._balanced_paren(self.text, op) if op >= 0 else -1
                    i = cl + 1 if cl > 0 else helper_match.end()
                    continue
            i += 1

    @staticmethod
    def _pp_expr(line: str) -> str:
        s = line.strip()
        for prefix in ("#ifdef", "#ifndef", "#if", "#elif"):
            if s.startswith(prefix):
                rest = s[len(prefix):].strip()
                rest = re.sub(r"defined\(\s*([\w]+)\s*\)", r"\1", rest)
                rest = re.sub(r"\s+", " ", rest)
                if prefix == "#ifndef":
                    rest = f"!({rest})"
                return rest
        return s

    def _walk_if_chain(self, start: int, conditions: list[str]) -> int:
        pop = self.text.find("(", start)
        if pop < 0:
            return start + 2
        pcl = self._balanced_paren(self.text, pop)
        if pcl < 0:
            return pop + 1
        cond = re.sub(r"\s+", " ", self.text[pop + 1 : pcl].strip())
        k = pcl + 1
        while k < self.end and self.text[k].isspace():
            k += 1
        if k < self.end and self.text[k] == "{":
            body_close = self._matched_brace(self.text, k)
            sub = BodyWalker(self.text, k + 1, body_close, self.helper_names)
            sub.walk(conditions + [cond])
            self.steps.extend(sub.steps)
            next_i = body_close + 1
        else:
            stmt_end = self._stmt_end(self.text, k, self.end)
            sub = BodyWalker(self.text, k, stmt_end, self.helper_names)
            sub.walk(conditions + [cond])
            self.steps.extend(sub.steps)
            next_i = stmt_end + 1

        k = next_i
        while k < self.end and self.text[k].isspace():
            k += 1
        if self.text.startswith("else", k):
            k2 = k + 4
            while k2 < self.end and self.text[k2].isspace():
                k2 += 1
            if k2 < self.end and self.text[k2] == "{":
                body_close = self._matched_brace(self.text, k2)
                sub = BodyWalker(self.text, k2 + 1, body_close, self.helper_names)
                sub.walk(conditions + [f"!({cond})"])
                self.steps.extend(sub.steps)
                return body_close + 1
            if self.text.startswith("if", k2):
                return self._walk_if_chain(k2, conditions + [f"!({cond})"])
            stmt_end = self._stmt_end(self.text, k2, self.end)
            sub = BodyWalker(self.text, k2, stmt_end, self.helper_names)
            sub.walk(conditions + [f"!({cond})"])
            self.steps.extend(sub.steps)
            return stmt_end + 1
        return next_i

    @staticmethod
    def _find_pp_close(text: str, start: int, end: int) -> int:
        depth = 1
        i = start
        while i < end:
            if text[i] != "#":
                i += 1
                continue
            line_start = text.rfind("\n", 0, i) + 1
            if text[line_start:i].strip():
                i += 1
                continue
            j = text.find("\n", i)
            if j < 0:
                j = end
            line = text[i:j].strip()
            if line.startswith("#if"):
                depth += 1
            elif line.startswith("#endif"):
                depth -= 1
                if depth == 0:
                    return j
            i = j + 1
        return end

    @staticmethod
    def _split_pp_branches(
        text: str, start: int, end: int, first_directive: str
    ) -> list[tuple[int, int, str]]:
        branches = []
        cur = start
        depth = 0
        current_directive = first_directive
        i = start
        while i < end:
            if text[i] != "#":
                i += 1
                continue
            line_start = text.rfind("\n", 0, i) + 1
            if text[line_start:i].strip():
                i += 1
                continue
            j = text.find("\n", i)
            if j < 0:
                j = end
            line = text[i:j].strip()
            if line.startswith("#if"):
                depth += 1
            elif line.startswith("#endif"):
                if depth == 0:
                    branches.append((cur, i, current_directive))
                    return branches
                depth -= 1
            elif depth == 0 and line.startswith("#elif"):
                branches.append((cur, i, current_directive))
                current_directive = line
                cur = j + 1
            elif depth == 0 and line.startswith("#else"):
                branches.append((cur, i, current_directive))
                current_directive = line
                cur = j + 1
            i = j + 1
        branches.append((cur, end, current_directive))
        return branches

    @staticmethod
    def _stmt_end(text: str, start: int, end: int) -> int:
        depth_p = 0
        depth_b = 0
        i = start
        in_str = False
        while i < end:
            c = text[i]
            if in_str:
                if c == "\\" and i + 1 < end:
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "(":
                    depth_p += 1
                elif c == ")":
                    depth_p -= 1
                elif c == "{":
                    depth_b += 1
                elif c == "}":
                    if depth_b == 0:
                        return i - 1
                    depth_b -= 1
                elif c == ";" and depth_p == 0 and depth_b == 0:
                    return i
            i += 1
        return end


def find_pipeline_cpp_files(root: Path) -> list[Path]:
    out = []
    for cpp in (root / "lib").rglob("*.cpp"):
        text = cpp.read_text(errors="replace")
        if (
            "PassPipelineRegistration" in text
            or ("PassManager" in text and ".addPass(" in text)
            or "runBiShengIRPipeline" in text
            or "runPipeline(" in text
        ):
            out.append(cpp)
    return out


def parse_cpp_file(path: Path) -> tuple[list[Builder], list[dict]]:
    text = strip_comments(path.read_text(errors="replace"))
    fn_bodies = find_function_bodies(text)
    helper_names = {name for name, _, _, _ in fn_bodies}

    builders: list[Builder] = []
    for name, sig_start, body_open, body_close in fn_bodies:
        builder = Builder(name=name, file=str(path), line=line_of(text, sig_start))
        walker = BodyWalker(text, body_open + 1, body_close, helper_names - {name})
        walker.walk(conditions=preprocessor_conditions_at(text, sig_start))
        builder.steps = walker.steps
        builders.append(builder)

    regs = find_pipeline_registrations(text)
    pipelines: list[dict] = []
    for reg in regs:
        if reg["body_open"] < 0:
            steps_dicts = []
            if reg.get("entry_builder"):
                steps_dicts.append(
                    {
                        "kind": "helper_call",
                        "conditions": [],
                        "line": reg["line"],
                        "constructor_fn": None,
                        "nested_op": None,
                        "flag": None,
                        "target": reg["entry_builder"],
                        "target_namespace": None,
                        "label": None,
                    }
                )
            pipelines.append(
                {
                    "name": reg["name"],
                    "desc": reg["desc"],
                    "options_class": reg["options_class"],
                    "file": str(path),
                    "line": reg["line"],
                    "steps": steps_dicts,
                }
            )
            if reg.get("entry_builder"):
                for builder in builders:
                    if builder.name == reg["entry_builder"]:
                        builder.is_pipeline = True
                        builder.pipeline_name = reg["name"]
                        builder.pipeline_desc = reg["desc"]
                        builder.options_class = reg["options_class"]
            continue

        walker = BodyWalker(text, reg["body_open"] + 1, reg["body_close"], helper_names)
        walker.walk(
            conditions=preprocessor_conditions_at(text, offset_of_line(text, reg["line"]))
        )
        pipelines.append(
            {
                "name": reg["name"],
                "desc": reg["desc"],
                "options_class": reg["options_class"],
                "file": str(path),
                "line": reg["line"],
                "steps": [asdict(step) for step in walker.steps],
            }
        )
        if walker.steps:
            for step in walker.steps:
                if step.kind == "helper_call" and step.target:
                    for builder in builders:
                        if builder.name == step.target:
                            builder.is_pipeline = True
                            builder.pipeline_name = reg["name"]
                            builder.pipeline_desc = reg["desc"]
                            builder.options_class = reg["options_class"]
    return builders, pipelines


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a bishengir pipeline skeleton from a repository.",
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="Repository root that contains the bishengir checkout.",
    )
    parser.add_argument(
        "--bishengir-root",
        help="Explicit bishengir root, relative to repo or absolute.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory for skeleton.json and coverage.json, relative to repo or absolute.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo = normalize_repo_path(Path(args.repo))
    out_dir = resolve_out_dir(repo, Path(args.out_dir) if args.out_dir else None)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_bishengir_root(
        repo,
        Path(args.bishengir_root) if args.bishengir_root else None,
    )
    print(f"[extract] bishengir root: {root}", file=sys.stderr)

    pass_defs: list[PassDef] = []
    for td in (root / "include").rglob("Passes.td"):
        pass_defs.extend(parse_passes_td(td))
    pass_defs.sort(key=lambda item: (item.td_file, item.td_line, item.flag, item.cls))
    print(f"[extract] passes defined: {len(pass_defs)}", file=sys.stderr)

    ctor_candidates: dict[str, list[tuple[Optional[str], str]]] = {}
    for pass_def in pass_defs:
        if pass_def.constructor_fn:
            ctor_candidates.setdefault(pass_def.constructor_fn, []).append(
                (pass_def.constructor_namespace, pass_def.flag)
            )

    def resolve_flag(constructor: str, file_path: str) -> Optional[str]:
        candidates = ctor_candidates.get(constructor, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][1]
        file_path_lower = file_path.lower()
        for namespace, flag in candidates:
            if namespace and namespace.lower() in file_path_lower:
                return flag
        return candidates[0][1]

    all_builders: list[Builder] = []
    all_pipelines: list[dict] = []
    for cpp in find_pipeline_cpp_files(root):
        try:
            builders, pipelines = parse_cpp_file(cpp)
        except Exception as exc:
            print(f"[extract] WARN parse failed for {cpp}: {exc}", file=sys.stderr)
            continue
        all_builders.extend(builders)
        all_pipelines.extend(pipelines)
    all_builders.sort(key=lambda item: (item.file, item.line, item.name))
    all_pipelines.sort(key=lambda item: (item["file"], item["line"], item["name"]))
    print(f"[extract] builders found: {len(all_builders)}", file=sys.stderr)
    print(f"[extract] pipeline registrations: {len(all_pipelines)}", file=sys.stderr)

    builder_dicts = []
    for builder in all_builders:
        builder_dict = asdict(builder)
        for step in builder_dict["steps"]:
            if step.get("constructor_fn"):
                step["flag"] = resolve_flag(step["constructor_fn"], builder.file)
        builder_dicts.append(builder_dict)
    for pipeline in all_pipelines:
        for step in pipeline["steps"]:
            if step.get("constructor_fn"):
                step["flag"] = resolve_flag(step["constructor_fn"], pipeline["file"])

    referenced_ctors = set()
    referenced_flags = set()
    for builder in all_builders:
        for step in builder.steps:
            if step.constructor_fn:
                referenced_ctors.add(step.constructor_fn)
                flag = resolve_flag(step.constructor_fn, builder.file)
                if flag:
                    referenced_flags.add(flag)

    defined_ctors = {pass_def.constructor_fn for pass_def in pass_defs if pass_def.constructor_fn}
    unreferenced = sorted(
        pass_def.flag
        for pass_def in pass_defs
        if pass_def.constructor_fn and pass_def.flag not in referenced_flags
    )

    coverage = {
        "passes_defined": len(pass_defs),
        "passes_referenced": len(referenced_ctors & defined_ctors),
        "unreferenced_pass_flags": unreferenced,
        "referenced_unknown_constructors": sorted(referenced_ctors - defined_ctors),
        "pipeline_count": len(all_pipelines),
        "builder_count": len(all_builders),
    }

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
    print(f"[extract] wrote {out_dir / 'skeleton.json'}", file=sys.stderr)
    print(
        f"[extract] coverage: {coverage['passes_referenced']}/{coverage['passes_defined']} "
        f"passes referenced ({len(unreferenced)} unreferenced)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
