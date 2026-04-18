import json
from pathlib import Path

from extract import main


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

    assert exit_code == 0
    assert skeleton["coverage"]["passes_defined"] == 1
    assert coverage["builder_count"] == 1


def test_extract_preserves_preprocessor_condition_text(tmp_path: Path):
    repo = tmp_path / "repo"
    root = repo / "3rdparty" / "AscendNPU-IR-Dev" / "bishengir"
    (root / "include" / "bishengir" / "Transforms").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Demo").mkdir(parents=True)
    (root / "include" / "bishengir" / "Transforms" / "Passes.td").write_text(
        'def DemoPass : Pass<"demo-pass", "mlir::ModuleOp"> { let constructor = "::mlir::createDemoPass()"; }'
    )
    (root / "lib" / "Dialect" / "Demo" / "Demo.cpp").write_text(
        '#if FEATURE_A\nvoid buildDemo(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }\n#elif FEATURE_B\nvoid buildDemo2(mlir::OpPassManager &pm) { pm.addPass(::mlir::createDemoPass()); }\n#endif\n'
    )

    main(["extract.py", str(repo)])
    skeleton = json.loads((repo / ".agent_pipelines" / "skeleton.json").read_text())

    conditions = [step["conditions"] for builder in skeleton["builders"] for step in builder["steps"]]
    flat = [item for items in conditions for item in items]
    assert any(text.startswith("#if FEATURE_A") for text in flat)
