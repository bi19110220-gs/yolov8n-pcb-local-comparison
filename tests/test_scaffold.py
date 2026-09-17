from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist():
    required = {
        ".github/workflows",
        ".vscode",
        "configs/original",
        "configs/trial035_parent",
        "configs/trial044_gpu_adaptation",
        "docs",
        "manifests/dataset",
        "manifests/hashes",
        "models/original",
        "models/trial035_parent",
        "models/trial044_gpu_adaptation",
        "results/original",
        "results/trial044_gpu_adaptation",
        "results/comparison",
        "scripts/package",
        "scripts/verify",
        "tools",
        "models/official",
        "tests",
    }
    assert not {path for path in required if not (ROOT / path).is_dir()}


def test_readme_states_reproduction_boundaries():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "Windows",
        "VS Code",
        "Ultralytics 8.4.84",
        "RTX 3080 adaptation",
        "different recorded training authorities",
        "not a controlled same-data ablation",
        "Held-out test evaluation was not run",
    ):
        assert phrase in text


def test_private_use_and_dataset_authorization_are_explicit():
    private_notice = (ROOT / "PRIVATE_USE_NOTICE.md").read_text(encoding="utf-8")
    provenance = (ROOT / "DATASET_PROVENANCE.md").read_text(encoding="utf-8")

    assert "does not grant an open-source license" in private_notice
    assert "authorized private redistribution" in provenance
    assert "no globally held-out image set" in provenance
    assert "test evaluation and test manifests remain excluded" in provenance


def test_vscode_launches_canonical_tools_package():
    import json
    tasks = json.loads((ROOT / ".vscode/tasks.json").read_text(encoding="utf-8"))
    commands = json.dumps(tasks)
    assert "src" not in commands
    assert "tools" in commands


def test_windows_entrypoints_bind_repository_virtual_environment():
    for name in ("tools/run_local_vscode_comparison.ps1", "tools/verify_package.ps1", "tools/prepare_dataset.ps1"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert '.venv\\Scripts\\python.exe' in text
        assert 'Test-Path -LiteralPath $pythonExe' in text
        assert 'Get-Command python' not in text


def test_comparison_reports_actual_training_time_from_csv(tmp_path):
    import csv
    from tools.run_local_vscode_comparison import write_comparison
    validation = dict(precision=0.5, recall=0.5, f1=0.5, map50=0.5, map50_95=0.5, inference_latency_ms=1, parameters=1, model_size_mb=1, per_class=[{"ap50_95": 0.5}] * 6)
    record = dict(model="model", training_authority="authority", device="cuda:0", validation=validation, started_at_unix=0, ended_at_unix=99, training={"elapsed_seconds": 42}, best_checkpoint="best.pt")
    write_comparison(tmp_path, record, record)
    with (tmp_path / "comparison.csv").open(newline="") as stream:
        assert float(next(csv.DictReader(stream))["training_seconds"]) == 42
