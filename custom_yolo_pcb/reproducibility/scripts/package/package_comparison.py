"""Package the approved comparison artifacts into a portable repository tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SOURCE_ITEMS = (
    ("yolov8n.pt", "models/official/yolov8n.pt"),
    (
        "results/top5_20_percent/configs/top5_20_044_parent035_classification_head_cls140_cpu_1024.json",
        "configs/trial044_gpu_adaptation/historical_cpu_trial044.json",
    ),
    (
        "results/top5_20_percent/runs/top5_20_035_parent029_classification_head_cls120_cpu_1024/weights/best.pt",
        "models/trial035_parent/best.pt",
    ),
    (
        "results/top5_20_percent/configs/top5_20_035_parent029_classification_head_cls120_cpu_1024.json",
        "configs/trial035_parent/historical_cpu_trial035.json",
    ),
)

# The portable runtime is maintained in this repository, not overwritten from
# the research workspace's machine-specific launcher on subsequent exports.
# Include the package-level targets used by the copied compatibility wrappers.
# The full-repository verification wrapper stays in the checkout: an evidence
# export does not contain its Git metadata, presentation, or full test suite.
RUNTIME_ITEMS = tuple((name, name) for name in (
    ".streamlit/config.toml", "PCB_Quality_Inspector.ipynb", "app.py", "README.md", "RESULTS_REPORT.md",
    "train_local.py", "prepare_dataset.ps1", "requirements.txt",
    "model_code/inspector_core.py", "model_code/app_template.py",
    "model_code/generate_app.py", "model_code/notebook_workflow.py",
    "tools/__init__.py", "tools/run_local_vscode_comparison.py",
    "tools/run_local_vscode_comparison.ps1", "tools/top5_classification_head_trainer.py",
    "tools/top5_mpdiou_trainer.py", "pcb_mpdiou_loss.py",
    "tools/prepare_dataset.ps1", "scripts/package/extract_dataset.py",
    "reproducibility/scripts/build_notebook.py", "reproducibility/README.md",
    "reproducibility/pyproject.toml",
    "manifests/dataset/pcb_yolo_train_val_v1.0.0.sha256",
))

RUN_ITEMS = (
    ("authority", "manifests/dataset/authority"),
    ("configs", "configs/runtime"),
    ("metrics", "results/metrics"),
    ("runs/original_grouped_v1_yolov8n", "results/original/training"),
    (
        "runs/enhanced_trial044_gpu_adaptation",
        "results/trial044_gpu_adaptation/training",
    ),
    ("comparison.csv", "results/comparison/comparison.csv"),
    ("comparison.md", "results/comparison/comparison.md"),
    ("environment.json", "configs/environment.json"),
    ("run_state.json", "results/comparison/run_state.json"),
    ("configs/original_train_args.json", "configs/original/recorded_train_args.json"),
    ("runs/original_grouped_v1_yolov8n/weights/best.pt", "models/original/best.pt"),
    ("runs/original_grouped_v1_yolov8n/weights/last.pt", "models/original/last.pt"),
    ("runs/enhanced_trial044_gpu_adaptation/weights/best.pt", "models/trial044_gpu_adaptation/best.pt"),
    ("runs/enhanced_trial044_gpu_adaptation/weights/last.pt", "models/trial044_gpu_adaptation/last.pt"),
)


def portable_destination(name: str) -> str:
    weights = {
        'models/official/yolov8n.pt': 'weights/yolov8n.pt',
        'models/trial035_parent/best.pt': 'weights/trial035_parent_best.pt',
        'models/original/best.pt': 'weights/original_best.pt',
        'models/original/last.pt': 'reproducibility/checkpoints/original_last.pt',
        'models/trial044_gpu_adaptation/best.pt': 'weights/trial044_best.pt',
        'models/trial044_gpu_adaptation/last.pt': 'reproducibility/checkpoints/trial044_last.pt',
    }
    if name in weights:
        return weights[name]
    if name.startswith(('configs/', 'manifests/', 'scripts/')):
        return 'reproducibility/' + name
    if name.startswith('tools/') or name == 'pcb_mpdiou_loss.py':
        return 'model_code/' + name
    return name


SOURCE_ITEMS = tuple((source, portable_destination(destination)) for source, destination in SOURCE_ITEMS)
RUN_ITEMS = tuple((source, portable_destination(destination)) for source, destination in RUN_ITEMS)
RUNTIME_ITEMS = tuple((portable_destination(source), portable_destination(destination)) for source, destination in RUNTIME_ITEMS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="",
    )
    os.replace(temporary, path)


def normalize_text(path: Path, source_root: Path) -> tuple[str, dict[str, Any]]:
    raw = Path(path).read_bytes()
    original_text = raw.decode("utf-8-sig")
    text = re.sub(r"\r+\n?", "\n", original_text)
    def sanitize(value: str) -> str:
        roots = {str(source_root), str(source_root).replace("\\", "/"), str(source_root).replace("/", "\\")}
        for root in sorted(roots, key=len, reverse=True):
            value = re.sub(re.escape(root), lambda _: "__SOURCE_ROOT__", value, flags=re.IGNORECASE)
        value = re.sub(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s\"']+", "__USER_HOME__", value)
        value = re.sub(r"(?i)\b[A-Z]:[\\/][^\r\n\"']*", lambda m: "__ABSOLUTE_ROOT__/" + m[0][3:].replace("\\", "/"), value)
        return re.sub(r"\\\\[^\\\s]+\\[^\r\n\"']*", lambda m: "__NETWORK_ROOT__/" + m[0].lstrip("\\").replace("\\", "/"), value)

    def visit(value: Any) -> Any:
        if isinstance(value, str):
            return sanitize(value)
        if isinstance(value, dict):
            return {sanitize(key): visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    if Path(path).suffix.lower() == ".json":
        normalized = json.dumps(visit(json.loads(text)), indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        normalized = sanitize(text)
    return normalized, {
        "source": str(Path(path)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized": normalized != original_text,
    }


def require_completed_run(state_path: Path) -> dict[str, Any]:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    required = {
        "status": "COMPLETED",
        "current_stage": "completed",
        "original_completed": True,
        "enhanced_completed": True,
        "test_split_used": False,
    }
    if any(state.get(key) != value for key, value in required.items()):
        raise PermissionError(
            "packaging requires COMPLETED with both stages complete and "
            "test_split_used=false"
        )
    return state


def _is_held_out_test_content(path: Path) -> bool:
    parts = {part.lower() for part in Path(path).parts}
    return "test" in parts and bool(parts.intersection({"images", "labels"}))


def _require_within(path: Path, root: Path) -> None:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise PermissionError(f"path is outside destination root: {resolved_path}")


def copy_artifact(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    destination_root: Path,
) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink():
        raise PermissionError(f"symbolic links are not publishable: {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size >= 100 * 1024 * 1024:
        raise PermissionError(f"file reaches the 100 MiB Git limit: {source.name}")
    if _is_held_out_test_content(source):
        raise PermissionError(f"held-out test content is forbidden: {source}")
    _require_within(destination, destination_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(source)
    normalized = False
    if source.suffix.lower() in TEXT_SUFFIXES:
        text, normalization = normalize_text(source, source_root)
        destination.write_text(text, encoding="utf-8", newline="")
        normalized = bool(normalization["normalized"])
    else:
        shutil.copy2(source, destination)

    try:
        source_label = source.resolve().relative_to(Path(source_root).resolve()).as_posix()
    except ValueError:
        source_label = source.name
    return {
        "source": source_label,
        "destination": destination.resolve()
        .relative_to(Path(destination_root).resolve())
        .as_posix(),
        "source_sha256": source_hash,
        "published_sha256": sha256_file(destination),
        "normalized": normalized,
        "size_bytes": destination.stat().st_size,
    }


def _copy_item(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    destination_root: Path,
) -> list[dict[str, Any]]:
    if source.is_dir():
        records = []
        for child in sorted(path for path in source.rglob("*") if path.is_file()):
            if "weights" in child.relative_to(source).parts:
                continue  # Explicit canonical models/ copies below.
            records.append(
                copy_artifact(
                    child,
                    destination / child.relative_to(source),
                    source_root=source_root,
                    destination_root=destination_root,
                )
            )
        return records
    return [
        copy_artifact(
            source,
            destination,
            source_root=source_root,
            destination_root=destination_root,
        )
    ]


def _copy_items(
    root: Path,
    destination_root: Path,
    items: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    records = []
    for source_relative, destination_relative in items:
        records.extend(
            _copy_item(
                root / source_relative,
                destination_root / destination_relative,
                source_root=root,
                destination_root=destination_root,
            )
        )
    return records


def export_portable_authorities(source_root: Path, run_root: Path, destination_root: Path) -> list[dict[str, Any]]:
    """Preserve authority order while decoupling identity from physical split dirs."""
    destination = destination_root / "reproducibility/manifests/dataset/authority"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("original_grouped_v1_train.txt", "original_grouped_v1_val.txt", "enhanced_trial044_ohem_train.txt", "enhanced_standard_val.txt"):
        source = run_root / "authority" / name
        if name == "enhanced_standard_val.txt":
            # Ultralytics directory discovery sorts paths before validation.
            val_root = source_root / "pcb_yolo_dataset/images/val"
            paths = sorted(path for path in val_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
            if not paths or any(path.parent != val_root for path in paths):
                raise PermissionError("standard validation must be a nonempty flat image directory")
            names = [path.name for path in paths]
            raw = ("\n".join(f"pcb_yolo_dataset/images/val/{name}" for name in names) + "\n").encode("utf-8")
            source_label = "pcb_yolo_dataset/images/val (sorted filename inventory)"
        else:
            raw = source.read_bytes()
            entries = [line.strip().replace("\\", "/") for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            names = [entry.rsplit("/", 1)[-1] for entry in entries]
            source_label = f"authority/{name}"
        if not names or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:jpg|jpeg|png|bmp|webp|tif|tiff)", item, re.IGNORECASE) for item in names):
            raise PermissionError("invalid image identity in authority")
        target = destination / name
        target.write_text("\n".join(f"pcb_yolo_dataset/images/pool/{item}" for item in names) + "\n", encoding="utf-8", newline="")
        records.append({"source": source_label, "destination": target.relative_to(destination_root).as_posix(), "source_sha256": hashlib.sha256(raw).hexdigest(), "published_sha256": sha256_file(target), "normalized": True, "size_bytes": target.stat().st_size})
    return records


def package(source_root: Path, run_root: Path, destination_root: Path) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    run_root = Path(run_root).resolve()
    destination_root = Path(destination_root).resolve()
    require_completed_run(run_root / "run_state.json")
    records = _copy_items(source_root, destination_root, SOURCE_ITEMS)
    records.extend(_copy_items(run_root, destination_root, RUN_ITEMS))
    authorities = export_portable_authorities(source_root, run_root, destination_root)
    authority_paths = {record["destination"] for record in authorities}
    records = [record for record in records if record["destination"] not in authority_paths] + authorities
    runtime_root = Path(__file__).resolve().parents[3]
    if runtime_root != destination_root:
        records.extend(_copy_items(runtime_root, destination_root, RUNTIME_ITEMS))
    # Regenerate presentation from recorded metrics: source stage timestamps
    # include clean validation and must not be labeled training duration.
    if str(runtime_root / 'model_code') not in sys.path:
        sys.path.insert(0, str(runtime_root / 'model_code'))
    from tools.run_local_vscode_comparison import write_comparison
    original = json.loads((destination_root / "results/metrics/original_clean_validation.json").read_text(encoding="utf-8"))
    enhanced = json.loads((destination_root / "results/metrics/enhanced_clean_validation.json").read_text(encoding="utf-8"))
    write_comparison(destination_root / "results/comparison", original, enhanced)
    for record in records:
        if record["destination"] in {"results/comparison/comparison.csv", "results/comparison/comparison.md"}:
            published = destination_root / record["destination"]
            record.update(published_sha256=sha256_file(published), size_bytes=published.stat().st_size, normalized=True,
                          transformation="regenerated from recorded clean-validation metrics; training_seconds is final cumulative results.csv time")
    payload = {
        "schema": "yolov8n_pcb_comparison_provenance/v1",
        "test_split_used": False,
        "artifacts": records,
    }
    atomic_json(
        destination_root / "reproducibility" / "manifests" / "hashes" / "artifact_provenance.json",
        payload,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = package(args.source_root, args.run_root, args.destination_root)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
