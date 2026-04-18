from validators import ValidationError, validate_pass_research


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
