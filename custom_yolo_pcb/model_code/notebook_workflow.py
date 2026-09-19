"""Notebook orchestration around the existing guarded local comparison runner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from inspector_core import sha256_file, verify_trial044_checkpoint


DATASET_RELEASE_URL = (
    "https://github.com/bi19110220-gs/yolov8n-pcb-local-comparison/"
    "releases/tag/v1.0.0"
)
STREAMLIT_URL = "http://localhost:8501"
_STREAMLIT_PROCESS: subprocess.Popen | None = None


class ProjectKernelError(EnvironmentError):
    """Raised when VS Code is not using the package-local Python kernel."""


class DatasetUnavailable(FileNotFoundError):
    """Raised when neither a prepared dataset nor verified Release ZIP exists."""


def _normalised(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def verify_project_kernel(package: Path, executable: Path | None = None) -> None:
    package = Path(package).resolve()
    expected = package / ".venv" / "Scripts" / "python.exe"
    actual = Path(executable or sys.executable).resolve()
    if _normalised(actual) != _normalised(expected):
        raise ProjectKernelError(
            "Select Python kernel -> custom_yolo_pcb\\.venv\\Scripts\\python.exe "
            "in VS Code, then click Run All again."
        )


def environment_summary() -> dict[str, Any]:
    import torch
    import ultralytics

    try:
        import streamlit

        streamlit_version = streamlit.__version__
    except ImportError:
        streamlit_version = "not installed"
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "streamlit": streamlit_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback",
    }


def _dataset_is_ready(package: Path) -> bool:
    dataset = package / "dataset" / "pcb_yolo_dataset"
    images = dataset / "images" / "pool"
    labels = dataset / "labels" / "pool"
    return (
        images.is_dir()
        and labels.is_dir()
        and next(images.iterdir(), None) is not None
        and next(labels.iterdir(), None) is not None
    )


def ensure_dataset_ready(package: Path) -> dict[str, str]:
    package = Path(package).resolve()
    if _dataset_is_ready(package):
        return {"status": "prepared", "path": str(package / "dataset" / "pcb_yolo_dataset")}
    archive = package.parent / "release-assets" / "pcb_yolo_train_val_v1.0.0.zip"
    checksum = (
        package
        / "reproducibility"
        / "manifests"
        / "dataset"
        / "pcb_yolo_train_val_v1.0.0.sha256"
    )
    if archive.is_file():
        scripts = package / "reproducibility" / "scripts"
        sys.path.insert(0, str(scripts))
        from package.extract_dataset import extract_dataset

        extract_dataset(archive, checksum, package / "dataset")
        if not _dataset_is_ready(package):
            raise DatasetUnavailable("Dataset extraction completed without a usable image/label pool.")
        return {"status": "extracted", "path": str(package / "dataset" / "pcb_yolo_dataset")}
    raise DatasetUnavailable(
        "Dataset is not prepared. Download pcb_yolo_train_val_v1.0.0.zip from "
        f"the private v1.0.0 Release, place it in release-assets, and Run All again: {DATASET_RELEASE_URL}"
    )


def verify_publication_inputs(package: Path) -> dict[str, str]:
    package = Path(package).resolve()
    model_code = package / "model_code"
    sys.path.insert(0, str(model_code))
    from tools.run_local_vscode_comparison import verify_packaged_inputs

    verify_packaged_inputs()
    trial044 = verify_trial044_checkpoint(package)
    original = package / "weights" / "original_best.pt"
    return {
        "original_best.pt": sha256_file(original),
        "trial044_best.pt": sha256_file(trial044),
        "test_split_used": "false",
    }


def display_recorded_results(package: Path) -> dict[str, Any]:
    package = Path(package).resolve()
    comparison = package / "results" / "comparison" / "comparison.csv"
    with comparison.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    charts = [
        package / "results" / "charts" / "original_training.png",
        package / "results" / "charts" / "trial044_training.png",
        package / "results" / "charts" / "final_comparison.png",
        package / "results" / "charts" / "per_class.png",
    ]
    missing = [str(path) for path in charts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Recorded chart is missing: {missing[0]}")
    return {"metrics": rows, "charts": charts, "test_split_used": False}


def new_notebook_run_dir(package: Path, now: datetime | None = None) -> Path:
    package = Path(package).resolve()
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output = package / "results" / "notebook_runs" / stamp
    if output.exists():
        raise FileExistsError(f"Notebook run directory already exists: {output}")
    return output


def run_complete_comparison(package: Path, now: datetime | None = None) -> Path:
    package = Path(package).resolve()
    output = new_notebook_run_dir(package, now)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"{output.name}.training.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        subprocess.run(
            [sys.executable, str(package / "train_local.py"), "--output-dir", str(output)],
            cwd=package,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return output


def run_enhanced_recovery(package: Path, recovery_output_dir: Path | None) -> Path:
    package = Path(package).resolve()
    if recovery_output_dir is None:
        raise ValueError("Set RECOVERY_OUTPUT_DIR to the exact failed notebook run before enabling recovery.")
    output = Path(recovery_output_dir).resolve()
    if not output.is_dir() or not output.is_relative_to(
        (package / "results" / "notebook_runs").resolve()
    ):
        raise ValueError("RECOVERY_OUTPUT_DIR must be an existing results/notebook_runs directory.")
    log_path = output.parent / f"{output.name}.recovery.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        subprocess.run(
            [
                sys.executable,
                str(package / "train_local.py"),
                "--output-dir",
                str(output),
                "--resume-enhanced-only",
            ],
            cwd=package,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return output


def _url_is_live(url: str = STREAMLIT_URL) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def start_streamlit(package: Path) -> dict[str, Any]:
    global _STREAMLIT_PROCESS
    package = Path(package).resolve()
    if _STREAMLIT_PROCESS is not None and _STREAMLIT_PROCESS.poll() is None:
        return {"status": "already_running", "url": STREAMLIT_URL, "pid": _STREAMLIT_PROCESS.pid}
    if _url_is_live():
        return {"status": "external_server_detected", "url": STREAMLIT_URL, "pid": None}
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _STREAMLIT_PROCESS = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(package / "app.py"),
            "--server.headless",
            "true",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=package,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    return {"status": "started", "url": STREAMLIT_URL, "pid": _STREAMLIT_PROCESS.pid}


def stop_streamlit() -> dict[str, Any]:
    global _STREAMLIT_PROCESS
    if _STREAMLIT_PROCESS is None:
        return {"status": "not_started_by_notebook"}
    if _STREAMLIT_PROCESS.poll() is None:
        _STREAMLIT_PROCESS.terminate()
        try:
            _STREAMLIT_PROCESS.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _STREAMLIT_PROCESS.kill()
            _STREAMLIT_PROCESS.wait(timeout=5)
    return_code = _STREAMLIT_PROCESS.returncode
    _STREAMLIT_PROCESS = None
    return {"status": "stopped", "return_code": return_code}
