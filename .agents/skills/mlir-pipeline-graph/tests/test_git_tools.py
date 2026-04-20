from pathlib import Path
from types import SimpleNamespace

import git_tools
from git_tools import git_output, map_changed_files_to_objects


def test_git_output_invokes_git_in_repo(monkeypatch):
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append((cmd, check, capture_output, text))
        return SimpleNamespace(stdout="first line\nsecond line\n")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    result = git_output(Path("/tmp/repo"), "status", "--short")

    assert result == "first line\nsecond line"
    assert calls == [(["git", "-C", "/tmp/repo", "status", "--short"], True, True, True)]


def test_map_changed_files_to_objects_flags_pipeline_and_op_hits():
    changed_files = [
        "3rdparty/AscendNPU-IR-Dev/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp",
        "3rdparty/AscendNPU-IR-Dev/bishengir/lib/Dialect/HFusion/Transforms/ConvertHFusionToHIVM.cpp",
    ]
    file_index = {
        changed_files[0]: {
            "pipelines": ["convert-to-hivm-pipeline"],
            "ops": [],
            "dialects": ["hivm"],
        },
        changed_files[1]: {
            "pipelines": ["convert-to-hivm-pipeline"],
            "ops": ["hfusion.matmul"],
            "dialects": ["hfusion"],
        },
    }

    result = map_changed_files_to_objects(changed_files, file_index)

    assert result["pipelines"] == ["convert-to-hivm-pipeline"]
    assert result["ops"] == ["hfusion.matmul"]
    assert sorted(result["dialects"]) == ["hfusion", "hivm"]
