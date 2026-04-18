from __future__ import annotations

from models import ValidationError, require_keys


def validate_pass_research(payload: dict) -> dict:
    require_keys(payload, ["flag", "summary", "key_options", "dialects_touched"])

    has_example = payload.get("example") is not None
    has_fallback = "example_missing_reason" in payload and "fallback_evidence" in payload
    if not has_example and not has_fallback:
        raise ValidationError("example or explicit fallback_evidence is required")

    if has_example:
        require_keys(
            payload["example"],
            ["test_path", "run_line", "input_ir", "output_ir", "check_lines", "evidence_type"],
        )

    return payload
