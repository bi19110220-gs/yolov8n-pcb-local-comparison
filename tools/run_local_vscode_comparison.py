"""Run the approved original-versus-Trial044 local validation comparison.

This runner is intentionally separate from the stateful top5 campaign. It writes
only to a new versioned output directory, never evaluates the test split, and
labels Trial044 as a CUDA adaptation because its recorded run used CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

DATASET_ROOT = ROOT / "dataset" / "pcb_yolo_dataset"
PUBLISHED_AUTHORITY = ROOT / "manifests" / "dataset" / "authority"
ORIGINAL_TRAIN_MANIFEST = PUBLISHED_AUTHORITY / "original_grouped_v1_train.txt"
ORIGINAL_VAL_MANIFEST = PUBLISHED_AUTHORITY / "original_grouped_v1_val.txt"
ENHANCED_TRAIN_MANIFEST = PUBLISHED_AUTHORITY / "enhanced_trial044_ohem_train.txt"
ENHANCED_VAL_MANIFEST = PUBLISHED_AUTHORITY / "enhanced_standard_val.txt"
TRIAL044_CONFIG = ROOT / "configs" / "trial044_gpu_adaptation" / "historical_cpu_trial044.json"
TRIAL035_BEST = ROOT / "models" / "trial035_parent" / "best.pt"
OFFICIAL_YOLOV8N = ROOT / "models" / "official" / "yolov8n.pt"

EXPECTED_HASHES = {
    "original_train": "b453321adecbf13015bfc4022aaf787dc6c61135af62f662e393390c26642392",
    "original_val": "bdafa2aee661aac03ab8ca7fad1ae551bbd754006454f3ab7a97289e776412a0",
    "enhanced_train": "8a9a0712524b457e18c0553bb8b409737d599bcd2c13be66abd75c12489ba6e6",
    "trial035_best": "073692d72d506d42bb5f4a995b1b36d55b5e8002e748e44b85b3fefdece8e5bd",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed.lower() != expected.lower():
        raise PermissionError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return observed


def require_published_artifact(path: Path, source_hash: str | None = None) -> dict[str, Any]:
    """Bind sanitized publication bytes to their recorded research-source bytes."""
    payload = json.loads((ROOT / "manifests" / "hashes" / "artifact_provenance.json").read_text(encoding="utf-8"))
    relative = path.relative_to(ROOT).as_posix()
    matches = [item for item in payload["artifacts"] if item["destination"] == relative]
    if len(matches) != 1:
        raise PermissionError(f"missing or ambiguous provenance: {relative}")
    record = matches[0]
    if source_hash is not None and record["source_sha256"] != source_hash:
        raise PermissionError(f"historical source SHA-256 mismatch: {relative}")
    require_hash(path, record["published_sha256"], relative)
    return record


def verify_packaged_inputs() -> dict[str, Any]:
    paths = (
        (ORIGINAL_TRAIN_MANIFEST, EXPECTED_HASHES["original_train"]),
        (ORIGINAL_VAL_MANIFEST, EXPECTED_HASHES["original_val"]),
        (ENHANCED_TRAIN_MANIFEST, EXPECTED_HASHES["enhanced_train"]),
        (ENHANCED_VAL_MANIFEST, None),
        (TRIAL035_BEST, EXPECTED_HASHES["trial035_best"]),
        (TRIAL044_CONFIG, None), (OFFICIAL_YOLOV8N, None),
    )
    return {"artifacts": [require_published_artifact(path, expected) for path, expected in paths], "test_split_used": False}


def manifest_image_paths(source: Path, *, dataset_root: Path) -> list[str]:
    """Resolve only attested train/val filenames, retaining order and duplicates."""
    lines = []
    dataset_root = Path(dataset_root)
    for entry in source.read_text(encoding="utf-8-sig").splitlines():
        if not entry.strip():
            continue
        parts = entry.strip().replace("\\", "/").split("/")
        if len(parts) != 4 or parts[0] != "pcb_yolo_dataset":
            raise PermissionError("invalid dataset manifest path")
        suffix = parts[parts.index("pcb_yolo_dataset") + 1:]
        if len(suffix) != 3 or suffix[0] != "images" or suffix[1] != "pool":
            raise PermissionError("manifest must contain canonical pool images only")
        _, split, name = suffix
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:jpg|jpeg|png|bmp|webp|tif|tiff)", name, re.IGNORECASE):
            raise PermissionError("invalid image filename in manifest")
        image = dataset_root / "images" / split / name
        label = dataset_root / "labels" / split / (Path(name).stem + ".txt")
        for path in (image, label):
            if not path.is_file():
                raise FileNotFoundError(f"required train/val dataset file is missing: {path}; extract the Release archive to dataset first")
            if any(parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()) for parent in (path, *path.parents) if parent == dataset_root or dataset_root in parent.parents):
                raise PermissionError("dataset filesystem aliases are forbidden")
            if not path.resolve().is_relative_to((dataset_root / path.parent.parent.name / split).resolve()):
                raise PermissionError("dataset path escapes its declared split")
        lines.append(str(image.resolve()))
    if not lines:
        raise ValueError("dataset manifest is empty")
    return lines


def materialize_manifest(source: Path, destination: Path, *, dataset_root: Path) -> None:
    lines = manifest_image_paths(source, dataset_root=dataset_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_new_output_directory(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def require_enhanced_resume_context(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    output_dir = Path(output_dir).resolve()
    verify_packaged_inputs()
    state_path = output_dir / "run_state.json"
    authority_path = output_dir / "authority" / "authority_record.json"
    original_metrics_path = output_dir / "metrics" / "original_clean_validation.json"
    for label, path in (
        ("run state", state_path),
        ("authority record", authority_path),
        ("original clean validation", original_metrics_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"resume {label} is missing: {path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not (
        state.get("schema") == "local_vscode_yolov8n_comparison/v1"
        and state.get("status") == "FAILED"
        and state.get("current_stage") == "enhanced_training"
        and state.get("original_completed") is True
        and state.get("test_split_used") is False
    ):
        raise PermissionError(
            "resume requires the exact FAILED enhanced-training boundary with "
            "a completed original run and test_split_used=false"
        )

    authorities = json.loads(authority_path.read_text(encoding="utf-8"))
    for model, stem, train, val in (
        ("original", "original_grouped_v1", ORIGINAL_TRAIN_MANIFEST, ORIGINAL_VAL_MANIFEST),
        ("enhanced", "enhanced_trial044", ENHANCED_TRAIN_MANIFEST, ENHANCED_VAL_MANIFEST),
    ):
        record = authorities.get(model, {})
        expected_yaml = output_dir / "authority" / f"{stem}_validation_only.yaml"
        if record.get("test_split_used") is not False or record.get("data_yaml") != str(expected_yaml):
            raise PermissionError(f"{model} runtime authority identity changed")
        expected_payload = {"path": str(DATASET_ROOT.resolve()), "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
        for role, published in (("train", train), ("val", val)):
            runtime = output_dir / "authority" / published.name
            publication = require_published_artifact(published)
            if record.get(f"{role}_manifest") != str(runtime) or record.get(f"{role}_sha256") != publication["source_sha256"]:
                raise PermissionError(f"{model} {role} authority identity changed")
            require_hash(runtime, str(record.get(f"runtime_{role}_sha256", "")), f"{model} runtime {role}")
            expected_lines = manifest_image_paths(published, dataset_root=DATASET_ROOT)
            if runtime.read_text(encoding="utf-8-sig").splitlines() != expected_lines or record.get(f"{role}_entries") != len(expected_lines):
                raise PermissionError(f"{model} {role} manifest differs from packaged authority")
            expected_payload[role] = str(runtime)
        if yaml.safe_load(expected_yaml.read_text(encoding="utf-8")) != expected_payload:
            raise PermissionError(f"{model} YAML must exactly match validation-only runtime authority")

    original = json.loads(original_metrics_path.read_text(encoding="utf-8"))
    if (
        original.get("model") != "original_grouped_v1_yolov8n"
        or original.get("test_split_used") is not False
    ):
        raise PermissionError("completed original metrics are not reusable")
    original_run = output_dir / "runs" / "original_grouped_v1_yolov8n"
    best = original_run / "weights" / "best.pt"
    last = original_run / "weights" / "last.pt"
    validation = original.get("validation", {})
    if (original.get("training_authority") != "grouped-v1"
        or original.get("best_checkpoint") != str(best)
        or original.get("last_checkpoint") != str(last)
        or not last.is_file()
        or validation.get("checkpoint") != str(best)
        or validation.get("data_yaml") != authorities["original"]["data_yaml"]
        or validation.get("split") != "val"
        or validation.get("test_split_used") is not False):
        raise PermissionError("completed original checkpoint/validation identity changed")
    require_hash(best, str(validation.get("checkpoint_sha256", "")), "completed original best checkpoint")
    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        value = validation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise PermissionError(f"completed original validation metric is missing or invalid: {key}")
    training = _results_csv_summary(original_run)
    if original.get("training") != training or training["epochs_logged"] != 100 or training["last_epoch"] != 100:
        raise PermissionError("original training completion evidence changed")

    enhanced_run_dir = output_dir / "runs" / "enhanced_trial044_gpu_adaptation"
    if enhanced_run_dir.exists() and any(enhanced_run_dir.iterdir()):
        raise FileExistsError(f"enhanced run directory is not empty: {enhanced_run_dir}")
    return state, authorities, original


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_validation_only_yaml(path: Path, *, train: Path, val: Path) -> None:
    payload = {
        "path": str(DATASET_ROOT.resolve()),
        "train": str(train.resolve()),
        "val": str(val.resolve()),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "test" in reloaded:
        raise PermissionError("validation-only authority unexpectedly contains test")


def prepare_data_authorities(output_dir: Path) -> dict[str, dict[str, Any]]:
    authority_dir = Path(output_dir) / "authority"
    authority_dir.mkdir(parents=True, exist_ok=True)

    original_train_hash = require_published_artifact(
        ORIGINAL_TRAIN_MANIFEST, EXPECTED_HASHES["original_train"]
    )["source_sha256"]
    original_val_hash = require_published_artifact(
        ORIGINAL_VAL_MANIFEST, EXPECTED_HASHES["original_val"]
    )["source_sha256"]
    enhanced_train_hash = require_published_artifact(
        ENHANCED_TRAIN_MANIFEST, EXPECTED_HASHES["enhanced_train"]
    )["source_sha256"]
    enhanced_val_hash = require_published_artifact(ENHANCED_VAL_MANIFEST)["source_sha256"]

    original_train_copy = authority_dir / "original_grouped_v1_train.txt"
    original_val_copy = authority_dir / "original_grouped_v1_val.txt"
    enhanced_train_copy = authority_dir / "enhanced_trial044_ohem_train.txt"
    enhanced_val_copy = authority_dir / "enhanced_standard_val.txt"
    for source, destination in ((ORIGINAL_TRAIN_MANIFEST, original_train_copy), (ORIGINAL_VAL_MANIFEST, original_val_copy), (ENHANCED_TRAIN_MANIFEST, enhanced_train_copy), (ENHANCED_VAL_MANIFEST, enhanced_val_copy)):
        materialize_manifest(source, destination, dataset_root=DATASET_ROOT)

    original_yaml = authority_dir / "original_grouped_v1_validation_only.yaml"
    enhanced_yaml = authority_dir / "enhanced_trial044_validation_only.yaml"
    _write_validation_only_yaml(
        original_yaml,
        train=original_train_copy,
        val=original_val_copy,
    )
    _write_validation_only_yaml(
        enhanced_yaml,
        train=enhanced_train_copy,
        val=enhanced_val_copy,
    )

    records = {
        "original": {
            "authority": "grouped-v1",
            "data_yaml": str(original_yaml.resolve()),
            "train_manifest": str(original_train_copy.resolve()),
            "val_manifest": str(original_val_copy.resolve()),
            "train_sha256": original_train_hash,
            "val_sha256": original_val_hash,
            "runtime_train_sha256": sha256(original_train_copy),
            "runtime_val_sha256": sha256(original_val_copy),
            "train_entries": _line_count(original_train_copy),
            "val_entries": _line_count(original_val_copy),
            "test_split_used": False,
        },
        "enhanced": {
            "authority": "Trial044 20% OHEM train view plus standard validation",
            "data_yaml": str(enhanced_yaml.resolve()),
            "train_manifest": str(enhanced_train_copy.resolve()),
            "val_manifest": str(enhanced_val_copy.resolve()),
            "val_sha256": enhanced_val_hash,
            "runtime_val_sha256": sha256(enhanced_val_copy),
            "val_entries": _line_count(enhanced_val_copy),
            "train_sha256": enhanced_train_hash,
            "runtime_train_sha256": sha256(enhanced_train_copy),
            "train_entries": _line_count(enhanced_train_copy),
            "test_split_used": False,
        },
    }
    atomic_json(authority_dir / "authority_record.json", records)
    return records


def original_train_args(data_yaml: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "project": str(Path(output_dir) / "runs"),
        "name": "original_grouped_v1_yolov8n",
        "exist_ok": False,
        "epochs": 100,
        "patience": 100,
        "batch": -1,
        "imgsz": 640,
        "device": 0,
        "workers": 0,
        "pretrained": True,
        "optimizer": "auto",
        "seed": 42,
        "deterministic": True,
        "cos_lr": False,
        "close_mosaic": 10,
        "amp": True,
        "fraction": 1.0,
        "val": True,
        "plots": True,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.4,
    }


def enhanced_train_args(
    config: dict[str, Any], data_yaml: Path, output_dir: Path
) -> dict[str, Any]:
    keys = (
        "imgsz",
        "batch",
        "epochs",
        "patience",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "box",
        "cls",
        "dfl",
        "mosaic",
        "close_mosaic",
        "mixup",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "fliplr",
        "flipud",
        "erasing",
        "workers",
        "seed",
        "amp",
        "resume",
        "val",
        "plots",
    )
    args = {key: config[key] for key in keys if key in config}
    args.update(
        {
            "data": str(data_yaml),
            "project": str(Path(output_dir) / "runs"),
            "name": "enhanced_trial044_gpu_adaptation",
            "exist_ok": False,
            "device": 0,
        }
    )
    return args


def runtime_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in {sys.executable}; use the CUDA-enabled system Python"
        )
    device_name = torch.cuda.get_device_name(0)
    return {
        "recorded_at_utc": utc_now(),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": device_name,
        "gpu_count": torch.cuda.device_count(),
        "ultralytics": ultralytics.__version__,
        "test_split_used": False,
    }


def _metric_list(value: Any) -> list[float]:
    return [float(item) for item in value]


def clean_validation(
    checkpoint: Path,
    data_yaml: Path,
    *,
    output_dir: Path,
    name: str,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=0,
        conf=0.001,
        iou=0.7,
        max_det=300,
        augment=False,
        plots=False,
        workers=0,
        project=str(Path(output_dir) / "validation"),
        name=name,
        exist_ok=False,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    per_class_precision = _metric_list(metrics.box.p)
    per_class_recall = _metric_list(metrics.box.r)
    per_class_ap50 = _metric_list(metrics.box.ap50)
    per_class_ap50_95 = _metric_list(metrics.box.maps)
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "precision": per_class_precision[index],
                "recall": per_class_recall[index],
                "ap50": per_class_ap50[index],
                "ap50_95": per_class_ap50_95[index],
            }
        )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "data_yaml": str(data_yaml.resolve()),
        "split": "val",
        "test_split_used": False,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "inference_latency_ms": float((metrics.speed or {}).get("inference", 0.0)),
        "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "model_size_mb": checkpoint.stat().st_size / (1024 * 1024),
    }


def _results_csv_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training results are empty: {path}")
    return {
        "results_csv": str(path.resolve()),
        "epochs_logged": len(rows),
        "last_epoch": int(float(rows[-1]["epoch"])),
        "elapsed_seconds": float(rows[-1]["time"]),
    }


def run_original(output_dir: Path, data_yaml: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    require_published_artifact(OFFICIAL_YOLOV8N)
    args = original_train_args(data_yaml, output_dir)
    atomic_json(output_dir / "configs" / "original_train_args.json", args)
    started = time.time()
    model = YOLO(str(OFFICIAL_YOLOV8N))
    model.train(**args)
    run_dir = output_dir / "runs" / args["name"]
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("original training did not publish best.pt and last.pt")
    validation = clean_validation(
        best,
        data_yaml,
        output_dir=output_dir,
        name="original_clean_val",
        imgsz=640,
        batch=8,
    )
    record = {
        "model": "original_grouped_v1_yolov8n",
        "training_authority": "grouped-v1",
        "device": "cuda:0",
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "best_checkpoint": str(best.resolve()),
        "last_checkpoint": str(last.resolve()),
        "validation": validation,
        "training": _results_csv_summary(run_dir),
        "test_split_used": False,
    }
    atomic_json(output_dir / "metrics" / "original_clean_validation.json", record)
    return record


def run_enhanced(output_dir: Path, data_yaml: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    from tools.top5_classification_head_trainer import Top5ClassificationHeadTrainer
    from tools.top5_mpdiou_trainer import assert_mpdiou_checkpoint_is_clean

    parent_hash = require_hash(
        TRIAL035_BEST, EXPECTED_HASHES["trial035_best"], "Trial035 parent checkpoint"
    )
    config = json.loads(TRIAL044_CONFIG.read_text(encoding="utf-8"))
    args = enhanced_train_args(config, data_yaml, output_dir)
    adaptation = {
        "source_candidate_id": config["candidate_id"],
        "source_device": config["device"],
        "adapted_device": 0,
        "adaptation": "RTX 3080 CUDA run; all listed Trial044 training hyperparameters preserved",
        "parent_checkpoint": str(TRIAL035_BEST.resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "test_split_used": False,
        "train_args": args,
    }
    atomic_json(output_dir / "configs" / "enhanced_trial044_gpu_adaptation.json", adaptation)

    started = time.time()
    model = YOLO(str(TRIAL035_BEST))
    model.train(trainer=Top5ClassificationHeadTrainer, **args)
    scope_audit = getattr(model.trainer, "classification_scope_audit", None)
    if not isinstance(scope_audit, dict) or scope_audit.get("optimizer_parameter_ids_bound") is not True:
        raise PermissionError("classification-head-only optimizer scope was not attested")
    if scope_audit.get("trainable_tensors") != 24 or scope_audit.get("trainable_parameters") != 370_578:
        raise PermissionError("classification-head-only scope changed")

    run_dir = output_dir / "runs" / args["name"]
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("enhanced training did not publish best.pt and last.pt")
    assert_mpdiou_checkpoint_is_clean(best, expected_parameters=int(config["expected_parameters"]))
    assert_mpdiou_checkpoint_is_clean(last, expected_parameters=int(config["expected_parameters"]))
    validation = clean_validation(
        best,
        data_yaml,
        output_dir=output_dir,
        name="enhanced_clean_val",
        imgsz=1024,
        batch=3,
    )
    record = {
        "model": "enhanced_trial044_gpu_adaptation",
        "source_candidate_id": config["candidate_id"],
        "training_authority": "Trial044 OHEM train view plus standard validation",
        "device": "cuda:0",
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "best_checkpoint": str(best.resolve()),
        "last_checkpoint": str(last.resolve()),
        "scope_audit": scope_audit,
        "validation": validation,
        "training": _results_csv_summary(run_dir),
        "test_split_used": False,
    }
    atomic_json(output_dir / "metrics" / "enhanced_clean_validation.json", record)
    return record


def write_comparison(output_dir: Path, original: dict[str, Any], enhanced: dict[str, Any]) -> None:
    rows = []
    for record in (original, enhanced):
        validation = record["validation"]
        rows.append(
            {
                "model": record["model"],
                "training_authority": record["training_authority"],
                "device": record["device"],
                "precision": validation["precision"],
                "recall": validation["recall"],
                "f1": validation["f1"],
                "map50": validation["map50"],
                "map50_95": validation["map50_95"],
                "short_ap50_95": validation["per_class"][3]["ap50_95"],
                "inference_latency_ms": validation["inference_latency_ms"],
                "parameters": validation["parameters"],
                "model_size_mb": validation["model_size_mb"],
                "training_seconds": record["training"]["elapsed_seconds"],
                "best_checkpoint": record["best_checkpoint"],
                "test_split_used": False,
            }
        )
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Local VS Code YOLOv8n comparison",
        "",
        "> Method warning: the original and enhanced runs intentionally use different recorded training authorities. This is a reproduction comparison, not a same-data controlled ablation.",
        "",
        "| Model | Training authority | Device | Precision | Recall | mAP50 | mAP50-95 | Short AP50-95 | Training hours |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['training_authority']} | {row['device']} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['map50']:.6f} | "
            f"{row['map50_95']:.6f} | {row['short_ap50_95']:.6f} | "
            f"{row['training_seconds'] / 3600:.3f} |"
        )
    lines.extend(
        [
            "",
            "Held-out test evaluation was not run.",
            "",
            f"Generated: {utc_now()}",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-enhanced-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check local inputs and materialize authorities without loading CUDA or training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    state_path = output_dir / "run_state.json"
    if args.preflight_only and args.resume_enhanced_only:
        raise ValueError("preflight-only and resume-enhanced-only are mutually exclusive")
    verify_packaged_inputs()
    if args.preflight_only:
        require_new_output_directory(output_dir)
        authorities = prepare_data_authorities(output_dir)
        print(json.dumps({"status": "PREFLIGHT_COMPLETED", "authorities": authorities, "test_split_used": False}, indent=2))
        return 0
    if args.resume_enhanced_only:
        state, authorities, original = require_enhanced_resume_context(output_dir)
        prior_failure = {
            "ended_at_utc": state.get("ended_at_utc"),
            "error": state.get("error"),
            "traceback": state.get("traceback"),
        }
        state.pop("ended_at_utc", None)
        state.pop("error", None)
        state.pop("traceback", None)
        state.update(
            status="RUNNING",
            current_stage="enhanced_training",
            pid=os.getpid(),
            enhanced_resume_started_at_utc=utc_now(),
            prior_enhanced_failure=prior_failure,
        )
        atomic_json(state_path, state)
        try:
            environment = runtime_environment()
            if environment["ultralytics"] != "8.4.84":
                raise RuntimeError(
                    f"Ultralytics version drift: expected 8.4.84, got {environment['ultralytics']}"
                )
            enhanced = run_enhanced(
                output_dir, Path(authorities["enhanced"]["data_yaml"])
            )
            state.update(current_stage="comparison", enhanced_completed=True)
            atomic_json(state_path, state)
            write_comparison(output_dir, original, enhanced)
            state.update(
                status="COMPLETED",
                current_stage="completed",
                ended_at_utc=utc_now(),
                comparison_csv=str((output_dir / "comparison.csv").resolve()),
                comparison_markdown=str((output_dir / "comparison.md").resolve()),
            )
            atomic_json(state_path, state)
            print(json.dumps(state, indent=2))
            return 0
        except BaseException as error:
            state.update(
                status="FAILED",
                current_stage=state.get("current_stage"),
                ended_at_utc=utc_now(),
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            atomic_json(state_path, state)
            raise

    require_new_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema": "local_vscode_yolov8n_comparison/v1",
        "status": "PREFLIGHT",
        "current_stage": "preflight",
        "output_dir": str(output_dir),
        "started_at_utc": utc_now(),
        "pid": os.getpid(),
        "test_split_used": False,
    }
    atomic_json(state_path, state)
    try:
        environment = runtime_environment()
        if environment["ultralytics"] != "8.4.84":
            raise RuntimeError(
                f"Ultralytics version drift: expected 8.4.84, got {environment['ultralytics']}"
            )
        atomic_json(output_dir / "environment.json", environment)
        authorities = prepare_data_authorities(output_dir)

        state.update(status="RUNNING", current_stage="original_training")
        atomic_json(state_path, state)
        original = run_original(output_dir, Path(authorities["original"]["data_yaml"]))
        state.update(current_stage="enhanced_training", original_completed=True)
        atomic_json(state_path, state)

        enhanced = run_enhanced(output_dir, Path(authorities["enhanced"]["data_yaml"]))
        state.update(current_stage="comparison", enhanced_completed=True)
        atomic_json(state_path, state)
        write_comparison(output_dir, original, enhanced)

        state.update(
            status="COMPLETED",
            current_stage="completed",
            ended_at_utc=utc_now(),
            comparison_csv=str((output_dir / "comparison.csv").resolve()),
            comparison_markdown=str((output_dir / "comparison.md").resolve()),
        )
        atomic_json(state_path, state)
        print(json.dumps(state, indent=2))
        return 0
    except BaseException as error:
        state.update(
            status="FAILED",
            current_stage=state.get("current_stage"),
            ended_at_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
