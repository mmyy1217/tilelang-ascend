import json
from pathlib import Path

import extract
from extract import main


def _normalize_repo_paths(value, repo_root: Path):
    if isinstance(value, dict):
        return {key: _normalize_repo_paths(val, repo_root) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_repo_paths(item, repo_root) for item in value]
    if isinstance(value, str):
        repo_root_str = str(repo_root)
        if value.startswith(repo_root_str):
            return value.replace(repo_root_str, "/repo", 1)
    return value


def test_extract_writes_skeleton_and_coverage(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        'void buildDemo(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }'
    )

    exit_code = main(["extract.py", str(repo)])

    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())
    coverage = json.loads((repo / ".agent_pipelines" / "coverage.json").read_text())
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "sample_skeleton.json").read_text()
    )

    assert exit_code == 0
    assert skeleton["schema_version"] == 1
    assert skeleton["coverage"]["passes_defined"] == 1
    assert skeleton["coverage"] == coverage
    assert coverage["builder_count"] == 1
    assert _normalize_repo_paths(skeleton, repo.resolve()) == fixture


def test_extract_preserves_preprocessor_condition_text(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "#if FEATURE_A\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "#elif FEATURE_B\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "#else\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "#endif\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    conditions = {
        tuple(step["conditions"])
        for builder in skeleton["builders"]
        for step in builder["steps"]
    }
    assert conditions == {
        ("#if FEATURE_A",),
        ("#elif FEATURE_B",),
        ("#else",),
    }


def test_extract_sorts_top_level_arrays_deterministically(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    for path in [
        root / "include" / "a",
        root / "include" / "b",
        root / "lib" / "a",
        root / "lib" / "b",
    ]:
        path.mkdir(parents=True)

    (root / "include" / "a" / "Passes.td").write_text(
        'def APass : Pass<"a-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createAPass()"; }'
    )
    (root / "include" / "b" / "Passes.td").write_text(
        'def BPass : Pass<"b-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createBPass()"; }'
    )
    (root / "lib" / "a" / "A.cpp").write_text(
        'void buildA(mlir::OpPassManager &pm) { pm.addPass(::mlir::createAPass()); }\n'
        'static mlir::PassPipelineRegistration<>("pipeline-a", "desc-a", buildA);\n'
    )
    (root / "lib" / "b" / "B.cpp").write_text(
        'void buildB(mlir::OpPassManager &pm) { pm.addPass(::mlir::createBPass()); }\n'
        'static mlir::PassPipelineRegistration<>("pipeline-b", "desc-b", buildB);\n'
    )

    original_rglob = Path.rglob

    def fake_rglob(self: Path, pattern: str):
        results = list(original_rglob(self, pattern))
        if self == root / "include" and pattern == "Passes.td":
            return list(reversed(results))
        return results

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    monkeypatch.setattr(
        extract,
        "find_pipeline_cpp_files",
        lambda _root: [
            root / "lib" / "b" / "B.cpp",
            root / "lib" / "a" / "A.cpp",
        ],
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert [item["flag"] for item in skeleton["passes"]] == ["a-pass", "b-pass"]
    assert [item["name"] for item in skeleton["builders"]] == ["buildA", "buildB"]
    assert [item["name"] for item in skeleton["pipelines"]] == [
        "pipeline-a",
        "pipeline-b",
    ]


def test_extract_preserves_pipeline_registration_preprocessor_conditions(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "#if FEATURE_PIPE\n"
        'static mlir::PassPipelineRegistration<>(\n'
        '    "demo-pipeline",\n'
        '    "demo desc",\n'
        '    [](mlir::OpPassManager &pm) {\n'
        '      pm.addPass(::mlir::createDemoPass());\n'
        "    });\n"
        "#endif\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert skeleton["pipelines"][0]["steps"][0]["conditions"] == ["#if FEATURE_PIPE"]


def test_extract_does_not_leak_else_if_condition_to_trailing_steps(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  if (useFirst()) {\n"
        "    pm.addPass(::mlir::createDemoPass());\n"
        "  } else if (useSecond()) {\n"
        "    pm.addPass(::mlir::createDemoPass());\n"
        "  }\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert skeleton["builders"][0]["steps"][-1]["conditions"] == []


def test_extract_preserves_nested_preprocessor_conditions(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "namespace demo {\n"
        "#if FEATURE_OUTER\n"
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "}\n"
        'static mlir::PassPipelineRegistration<>("demo-pipeline", "demo desc", [](mlir::OpPassManager &pm) {\n'
        "  pm.addPass(::mlir::createDemoPass());\n"
        "});\n"
        "#endif\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert skeleton["builders"][0]["steps"][0]["conditions"] == ["#if FEATURE_OUTER"]
    assert skeleton["pipelines"][0]["steps"][0]["conditions"] == ["#if FEATURE_OUTER"]


def test_extract_preserves_function_pointer_registration_conditions(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "#if FEATURE_PIPE\n"
        "void buildFoo(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "}\n"
        'static mlir::PassPipelineRegistration<>("demo-pipeline", "demo desc", buildFoo);\n'
        "#endif\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert skeleton["pipelines"][0]["steps"][0]["conditions"] == ["#if FEATURE_PIPE"]
    assert skeleton["pipelines"][0]["steps"][0]["target"] == "buildFoo"


def test_extract_counts_pipeline_constructor_references_in_coverage(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        'static mlir::PassPipelineRegistration<>("demo-pipeline", "demo desc", [](mlir::OpPassManager &pm) {\n'
        "  pm.addPass(::mlir::createDemoPass());\n"
        "});\n"
    )

    main(["extract.py", str(repo)])
    coverage = json.loads((repo / ".agent_pipelines" / "coverage.json").read_text())

    assert coverage["passes_referenced"] == 1
    assert coverage["unreferenced_pass_flags"] == []


def test_extract_parses_add_nested_pass_calls(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "func::FuncOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  pm.addNestedPass<func::FuncOp>(::mlir::createDemoPass());\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    step = skeleton["builders"][0]["steps"][0]
    assert step["kind"] == "nested_pass"
    assert step["nested_op"] == "func::FuncOp"
    assert step["flag"] == "demo-pass"


def test_extract_counts_referenced_pass_flags_with_duplicate_constructors(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Dialect" / "HFusion").mkdir(parents=True)
    (root / "include" / "bishengir" / "Dialect" / "HIVM").mkdir(parents=True)
    (root / "lib" / "Dialect" / "HFusion").mkdir(parents=True)
    (root / "lib" / "Dialect" / "HIVM").mkdir(parents=True)
    (root / "include" / "bishengir" / "Dialect" / "HFusion" / "Passes.td").write_text(
        'def HFusionFlattenPass : Pass<"hfusion-flatten", "mlir::ModuleOp"> { '
        'let constructor = "::mlir::hfusion::createFlattenOpsPass()"; }'
    )
    (root / "include" / "bishengir" / "Dialect" / "HIVM" / "Passes.td").write_text(
        'def HIVMFlattenPass : Pass<"hivm-flatten", "mlir::ModuleOp"> { '
        'let constructor = "::mlir::hivm::createFlattenOpsPass()"; }'
    )
    (root / "lib" / "Dialect" / "HFusion" / "HFusion.cpp").write_text(
        "void buildHFusion(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::hfusion::createFlattenOpsPass());\n"
        "}\n"
    )
    (root / "lib" / "Dialect" / "HIVM" / "HIVM.cpp").write_text(
        "void buildHIVM(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::hivm::createFlattenOpsPass());\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    coverage = json.loads((repo / ".agent_pipelines" / "coverage.json").read_text())

    assert coverage["passes_defined"] == 2
    assert coverage["passes_referenced"] == 2
    assert coverage["unreferenced_pass_flags"] == []


def test_extract_links_qualified_local_builder_calls_from_pipeline_registration(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "namespace bishengir {\n"
        "void createDemoPipeline(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::createDemoPass());\n"
        "}\n"
        'static mlir::PassPipelineRegistration<>("demo-pipeline", "demo desc", [](mlir::OpPassManager &pm) {\n'
        "  bishengir::createDemoPipeline(pm);\n"
        "});\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    pipeline_step = skeleton["pipelines"][0]["steps"][0]
    builder = skeleton["builders"][0]

    assert pipeline_step["kind"] == "helper_call"
    assert pipeline_step["target"] == "createDemoPipeline"
    assert builder["name"] == "createDemoPipeline"
    assert builder["is_pipeline"] is True
    assert builder["pipeline_name"] == "demo-pipeline"
