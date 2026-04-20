import pytest

from validators import ValidationError, validate_pass_research, validate_pipeline_research


def test_validate_pass_research_accepts_test_backed_example():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
        "example": {
            "test_path": "test/Dialect/Annotation/annotation-lowering.mlir",
            "run_line": "// RUN: bishengir-opt %s -annotation-lowering | FileCheck %s",
            "input_ir": "func.func @main() { annotation.mark }",
            "output_ir": "func.func @main() { return }",
            "check_lines": ["// CHECK-NOT: annotation.mark"],
            "evidence_type": "test",
        },
    }

    validated = validate_pass_research(record)

    assert validated["flag"] == "annotation-lowering"
    assert validated["example"]["evidence_type"] == "test"


def test_validate_pass_research_rejects_missing_example_and_fallback():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
    }

    try:
        validate_pass_research(record)
    except ValidationError as exc:
        assert "example" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_validate_pass_research_rejects_non_object_example():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
        "example": "not-an-example-object",
    }

    try:
        validate_pass_research(record)
    except ValidationError as exc:
        assert "example" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_validate_pass_research_rejects_placeholder_fallback():
    record = {
        "flag": "annotation-lowering",
        "summary": "Erase annotation.mark ops.",
        "key_options": [],
        "dialects_touched": ["annotation"],
        "example": None,
        "example_missing_reason": "TBD",
        "fallback_evidence": {},
    }

    try:
        validate_pass_research(record)
    except ValidationError as exc:
        assert "meaningful" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_validate_pipeline_research_rejects_missing_helpers_key():
    record = {
        "pipeline": "convert-to-hivm-pipeline",
        "passes": [],
    }

    try:
        validate_pipeline_research(record)
    except ValidationError as exc:
        assert "helpers" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_validate_pipeline_research_rejects_non_mapping_payload():
    with pytest.raises(ValidationError) as excinfo:
        validate_pipeline_research(None)

    assert "mapping" in str(excinfo.value)
