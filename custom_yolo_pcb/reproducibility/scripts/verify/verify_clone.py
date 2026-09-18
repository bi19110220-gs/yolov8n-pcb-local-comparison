"""Check the files a fresh Git clone actually receives, without running training."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_FILES = (
    "tools/__init__.py", "tools/run_local_vscode_comparison.py",
    "tools/run_local_vscode_comparison.ps1", "tools/top5_mpdiou_trainer.py",
    "tools/prepare_dataset.ps1", "tools/verify_package.ps1",
    "scripts/package/extract_dataset.py",
    "tools/top5_classification_head_trainer.py", "pcb_mpdiou_loss.py",
    "models/official/yolov8n.pt", "models/trial035_parent/best.pt",
    "models/original/best.pt", "models/original/last.pt",
    "models/trial044_gpu_adaptation/best.pt", "models/trial044_gpu_adaptation/last.pt",
    "configs/original/recorded_train_args.json",
    "configs/trial035_parent/historical_cpu_trial035.json",
    "configs/trial044_gpu_adaptation/historical_cpu_trial044.json",
    "manifests/hashes/artifact_provenance.json",
    "manifests/dataset/authority/original_grouped_v1_train.txt",
    "manifests/dataset/authority/original_grouped_v1_val.txt",
    "manifests/dataset/authority/enhanced_trial044_ohem_train.txt",
    "manifests/dataset/authority/enhanced_standard_val.txt",
)


# Required files are expressed relative to the repository, not the working directory.
_WEIGHTS = {
    'models/official/yolov8n.pt': 'weights/yolov8n.pt',
    'models/trial035_parent/best.pt': 'weights/trial035_parent_best.pt',
    'models/original/best.pt': 'weights/original_best.pt',
    'models/original/last.pt': 'reproducibility/checkpoints/original_last.pt',
    'models/trial044_gpu_adaptation/best.pt': 'weights/trial044_best.pt',
    'models/trial044_gpu_adaptation/last.pt': 'reproducibility/checkpoints/trial044_last.pt',
}
REQUIRED_FILES = tuple('custom_yolo_pcb/' + (
    _WEIGHTS[name] if name in _WEIGHTS else
    'model_code/' + name if name.startswith('tools/') or name == 'pcb_mpdiou_loss.py' else
    'reproducibility/' + name
) for name in REQUIRED_FILES) + tuple('custom_yolo_pcb/' + name for name in ('train_local.py', 'prepare_dataset.ps1', 'README.md', 'RESULTS_REPORT.md', 'requirements.txt'))


def copy_git_candidates(root: Path, destination: Path) -> None:
    """Include tracked and nonignored untracked files; never copy empty folders."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("clone simulation destination must be empty")
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=True,
    )
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        source, target = root / name, destination / name
        if not source.is_file():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise PermissionError(f"Git candidate escapes repository: {name}")
        if not target.resolve().is_relative_to(destination):
            raise PermissionError(f"Git candidate escapes clone: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify_layout(root: Path) -> list[str]:
    return [f"missing or empty required file: {name}" for name in REQUIRED_FILES
            if not (Path(root) / name).is_file() or (Path(root) / name).stat().st_size == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pcb fresh clone ") as temporary:
        clone = Path(temporary)
        copy_git_candidates(args.root, clone)
        failures = verify_layout(clone)
        if failures:
            print("\n".join(failures))
            return 1
        for command in (
            [sys.executable, "custom_yolo_pcb/train_local.py", "--help"],
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'custom_yolo_pcb/model_code'); from tools.run_local_vscode_comparison import verify_packaged_inputs; verify_packaged_inputs()"],
            [sys.executable, "custom_yolo_pcb/reproducibility/scripts/verify/verify_package.py", "--root", "."],
        ):
            subprocess.run(command, cwd=clone, check=True)
    print("Fresh Git-file clone verification passed; no training or test evaluation was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
