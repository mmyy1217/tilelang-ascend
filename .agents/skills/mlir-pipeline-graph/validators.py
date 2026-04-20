from __future__ import annotations

from collections.abc import Mapping

from models import ValidationError, require_keys

_PLACEHOLDER_TEXT = {
    "n/a",
    "na",
    "none",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}


def _is_meaningful_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return bool(normalized) and normalized not in _PLACEHOLDER_TEXT


def _is_meaningful_value(value: object) -> bool:
    if isinstance(value, str):
        return _is_meaningful_text(value)
    if isinstance(value, Mapping):
        return bool(value) and all(_is_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(_is_meaningful_value(item) for item in value)
    return bool(value)


def validate_pass_research(payload: dict) -> dict:
    require_keys(payload, ["flag", "summary", "key_options", "dialects_touched"])

    has_example = payload.get("example") is not None
    has_fallback = "example_missing_reason" in payload and "fallback_evidence" in payload
    if not has_example and not has_fallback:
        raise ValidationError("example or explicit fallback_evidence is required")

    if has_example:
        example = payload["example"]
        if not isinstance(example, Mapping):
            raise ValidationError("example must be a mapping")
        require_keys(example, ["test_path", "run_line", "input_ir", "output_ir", "check_lines", "evidence_type"])
    else:
        example_missing_reason = payload.get("example_missing_reason")
        fallback_evidence = payload.get("fallback_evidence")
        if not _is_meaningful_text(example_missing_reason):
            raise ValidationError("example_missing_reason must be meaningful")
        if not _is_meaningful_value(fallback_evidence):
            raise ValidationError("fallback_evidence must be meaningful")

    return payload


def validate_pipeline_research(payload: dict) -> dict:
    require_keys(payload, ["pipeline", "passes", "helpers"])

    passes = payload["passes"]
    if not isinstance(passes, list):
        raise ValidationError("passes must be a list")
    if not isinstance(payload["helpers"], list):
        raise ValidationError("helpers must be a list")

    for index, pass_payload in enumerate(passes):
        if not isinstance(pass_payload, Mapping):
            raise ValidationError(f"pass entry {index} must be a mapping")
        validate_pass_research(pass_payload)

    return payload
