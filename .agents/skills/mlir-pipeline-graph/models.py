from __future__ import annotations

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
