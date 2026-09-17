import json
from pathlib import Path

import pytest
import yaml

from tools import run_local_vscode_comparison as runner


@pytest.fixture
def completed_original(tmp_path, monkeypatch):
    root = tmp_path / "portable repo"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    dataset = root / "dataset/pcb_yolo_dataset"
    monkeypatch.setattr(runner, "DATASET_ROOT", dataset)
    artifacts = []
    manifest_names = {
        "ORIGINAL_TRAIN_MANIFEST": "original_grouped_v1_train.txt",
        "ORIGINAL_VAL_MANIFEST": "original_grouped_v1_val.txt",
        "ENHANCED_TRAIN_MANIFEST": "enhanced_trial044_ohem_train.txt",
        "ENHANCED_VAL_MANIFEST": "enhanced_standard_val.txt",
    }
    expected_keys = ("original_train", "original_val", "enhanced_train", None)
    for (attribute, name), key in zip(manifest_names.items(), expected_keys):
        manifest = root / "manifests/dataset/authority" / name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("pcb_yolo_dataset/images/pool/a.jpg\n", encoding="utf-8")
        monkeypatch.setattr(runner, attribute, manifest)
        artifacts.append({"destination": manifest.relative_to(root).as_posix(), "source_sha256": runner.EXPECTED_HASHES.get(key, "0" * 64), "published_sha256": runner.sha256(manifest)})
    for attribute, relative in (("TRIAL035_BEST", "models/trial035_parent/best.pt"), ("OFFICIAL_YOLOV8N", "models/official/yolov8n.pt"), ("TRIAL044_CONFIG", "configs/trial044.json")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        monkeypatch.setattr(runner, attribute, path)
        artifacts.append({"destination": relative, "source_sha256": runner.EXPECTED_HASHES["trial035_best"] if attribute == "TRIAL035_BEST" else runner.sha256(path), "published_sha256": runner.sha256(path)})
    runner.atomic_json(root / "manifests/hashes/artifact_provenance.json", {"artifacts": artifacts})
    for relative in ("images/pool/a.jpg", "labels/pool/a.txt"):
        file = dataset / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"data")
    output = root / "output"
    authorities = runner.prepare_data_authorities(output)
    run = output / "runs/original_grouped_v1_yolov8n"
    (run / "weights").mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (run / "weights" / name).write_bytes(name.encode())
    (run / "results.csv").write_text("epoch,time\n" + "".join(f"{i},{i}\n" for i in range(1, 101)))
    best, last = run / "weights/best.pt", run / "weights/last.pt"
    original = {
        "model": "original_grouped_v1_yolov8n", "training_authority": "grouped-v1", "test_split_used": False,
        "best_checkpoint": str(best.resolve()), "last_checkpoint": str(last.resolve()),
        "validation": {"checkpoint": str(best.resolve()), "checkpoint_sha256": runner.sha256(best), "data_yaml": authorities["original"]["data_yaml"], "split": "val", "test_split_used": False, "precision": 0.5, "recall": 0.5, "f1": 0.5, "map50": 0.5, "map50_95": 0.5},
        "training": runner._results_csv_summary(run),
    }
    runner.atomic_json(output / "metrics/original_clean_validation.json", original)
    runner.atomic_json(output / "run_state.json", {"schema": "local_vscode_yolov8n_comparison/v1", "status": "FAILED", "current_stage": "enhanced_training", "original_completed": True, "test_split_used": False})
    return output, authorities, original


def test_resume_accepts_real_completed_original_and_exact_runtime_authorities(completed_original):
    output, authorities, original = completed_original
    _, observed, metrics = runner.require_enhanced_resume_context(output)
    assert observed == authorities
    assert metrics == original


@pytest.mark.parametrize("tamper", ["yaml_train", "yaml_val", "yaml_root", "yaml_test", "manifest", "record_hash", "record_manifest", "packaged_manifest", "checkpoint", "metric_checkpoint", "metric_yaml", "metric_test", "missing_metrics", "incomplete_training", "running_state"])
def test_resume_rejects_tampering(completed_original, tamper):
    output, authorities, original = completed_original
    enhanced = authorities["enhanced"]
    if tamper.startswith("yaml_"):
        file = Path(enhanced["data_yaml"])
        data = yaml.safe_load(file.read_text())
        key = {"yaml_train": "train", "yaml_val": "val", "yaml_root": "path", "yaml_test": "test"}[tamper]
        data[key] = str(output / "unapproved" / "test" / "images")
        file.write_text(yaml.safe_dump(data))
    elif tamper == "manifest":
        Path(enhanced["train_manifest"]).write_text(str(output / "unapproved.jpg") + "\n")
    elif tamper in ("record_hash", "record_manifest"):
        enhanced["runtime_train_sha256" if tamper == "record_hash" else "train_manifest"] = "wrong"
        runner.atomic_json(output / "authority/authority_record.json", authorities)
    elif tamper == "packaged_manifest":
        runner.ENHANCED_TRAIN_MANIFEST.write_text("pcb_yolo_dataset/images/pool/tampered.jpg\n")
    elif tamper == "checkpoint":
        Path(original["best_checkpoint"]).write_bytes(b"changed")
    elif tamper == "metric_checkpoint":
        original["validation"]["checkpoint"] = str(output / "other.pt")
    elif tamper == "metric_yaml":
        original["validation"]["data_yaml"] = enhanced["data_yaml"]
    elif tamper == "metric_test":
        original["validation"]["test_split_used"] = True
    elif tamper == "missing_metrics":
        del original["validation"]["map50"]
    elif tamper == "incomplete_training":
        Path(original["training"]["results_csv"]).write_text("epoch,time\n1,1\n")
    elif tamper == "running_state":
        state_path = output / "run_state.json"
        state = json.loads(state_path.read_text())
        state["status"] = "RUNNING"
        runner.atomic_json(state_path, state)
    runner.atomic_json(output / "metrics/original_clean_validation.json", original)
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        runner.require_enhanced_resume_context(output)
