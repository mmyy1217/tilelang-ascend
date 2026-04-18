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


def test_extract_recovers_constructor_from_wrapped_add_pass_local_variable(
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
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  auto demoPass = ::mlir::createDemoPass();\n"
        "  pm.addPass(std::unique_ptr<mlir::Pass>(dyn_cast<mlir::Pass>(demoPass.release())));\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    steps = skeleton["builders"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["kind"] == "pass"
    assert steps[0]["constructor_fn"] == "createDemoPass"
    assert steps[0]["flag"] == "demo-pass"


def test_extract_recovers_constructor_from_inline_wrapped_add_pass(
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
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(std::unique_ptr<mlir::Pass>(wrap(::mlir::createDemoPass())));\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    steps = skeleton["builders"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["kind"] == "pass"
    assert steps[0]["constructor_fn"] == "createDemoPass"
    assert steps[0]["flag"] == "demo-pass"


def test_extract_does_not_recover_unrelated_wrapped_local_constructor(
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
        "void buildDemo(mlir::OpPassManager &pm) {\n"
        "  auto unrelated = createNotAPass();\n"
        "  pm.addPass(wrap(unrelated));\n"
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    steps = skeleton["builders"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["kind"] == "pass"
    assert steps[0]["constructor_fn"] is None
    assert steps[0]["flag"] is None


def test_extract_matches_indented_namespaced_builder_definition(
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
        "  void buildDemo(mlir::OpPassManager &pm) {\n"
        "    pm.addPass(::mlir::createDemoPass());\n"
        "  }\n"
        '  static mlir::PassPipelineRegistration<>("demo-pipeline", "demo desc", buildDemo);\n'
        "}\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert [builder["name"] for builder in skeleton["builders"]] == ["buildDemo"]
    assert skeleton["builders"][0]["is_pipeline"] is True
    assert skeleton["builders"][0]["pipeline_name"] == "demo-pipeline"
    assert skeleton["pipelines"][0]["steps"][0]["target"] == "buildDemo"
    assert skeleton["pipelines"][0]["steps"][0]["target_namespace"] == "demo"


def test_extract_uses_namespace_when_linking_same_named_pipeline_builders(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def AlphaPass : Pass<"alpha-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createAlphaPass()"; }\n'
        'def BetaPass : Pass<"beta-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createBetaPass()"; }\n'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        "namespace alpha {\n"
        "void buildShared(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::createAlphaPass());\n"
        "}\n"
        "}\n"
        "namespace beta {\n"
        "void buildShared(mlir::OpPassManager &pm) {\n"
        "  pm.addPass(::mlir::createBetaPass());\n"
        "}\n"
        "}\n"
        'static mlir::PassPipelineRegistration<>("alpha-pipeline", "alpha desc", alpha::buildShared);\n'
        'static mlir::PassPipelineRegistration<>("beta-pipeline", "beta desc", [](mlir::OpPassManager &pm) {\n'
        "  beta::buildShared(pm);\n"
        "});\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    pipeline_builders = [builder for builder in skeleton["builders"] if builder["is_pipeline"]]
    assert len(pipeline_builders) == 2
    assert sorted(builder["pipeline_name"] for builder in pipeline_builders) == [
        "alpha-pipeline",
        "beta-pipeline",
    ]
    by_flag = {builder["steps"][0]["flag"]: builder for builder in pipeline_builders}
    assert by_flag["alpha-pass"]["pipeline_name"] == "alpha-pipeline"
    assert by_flag["beta-pass"]["pipeline_name"] == "beta-pipeline"
    beta_pipeline = next(
        pipeline for pipeline in skeleton["pipelines"] if pipeline["name"] == "beta-pipeline"
    )
    assert beta_pipeline["steps"][0]["target"] == "buildShared"
    assert beta_pipeline["steps"][0]["target_namespace"] == "beta"


def test_extract_preserves_outer_preprocessor_conditions_with_leading_whitespace(
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
        "  #if FEATURE_PIPE\n"
        "  void buildDemo(mlir::OpPassManager &pm) {\n"
        "    pm.addPass(::mlir::createDemoPass());\n"
        "  }\n"
        "  #endif\n"
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    assert skeleton["builders"][0]["steps"][0]["conditions"] == ["#if FEATURE_PIPE"]


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
