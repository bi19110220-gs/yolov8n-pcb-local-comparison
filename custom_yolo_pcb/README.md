# PCB defect detection: local YOLOv8n comparison

Start with [the results report](RESULTS_REPORT.md). It contains both models' saved training curves, final validation results, and matching regenerated validation plots. You do not need to train again to examine the results.

This Windows / VS Code package compares the original YOLOv8n with the Trial 044 RTX 3080 adaptation. They use different recorded training authorities, starting checkpoints, schedules, and image sizes; this is not a controlled same-data ablation. Held-out test evaluation was not run. There is no globally held-out image set in the combined dataset release because the two authority schemes assign different roles to some images.

## 1. Install locally

Install Python 3.11 and VS Code, then open this `custom_yolo_pcb` folder in VS Code. In its PowerShell terminal:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__); print('CUDA' if torch.cuda.is_available() else 'CPU')"
```

Select `.venv` with **Python: Select Interpreter** in VS Code. If PowerShell blocks activation, use `& .\.venv\Scripts\python.exe` wherever a command below says `python`. The recorded environment is Python 3.11.9, PyTorch 2.11.0+cu128, and Ultralytics 8.4.84. A compatible NVIDIA driver is required for CUDA. If CUDA is unavailable, the training and validation entry points automatically select CPU; this can be substantially slower. Neither different hardware nor CPU fallback promises identical numerical results. The original automatic batch selection can also depend on available memory.

## 2. Download and prepare the dataset

Download `pcb_yolo_train_val_v1.0.0.zip` from the private [v1.0.0 Release](https://github.com/bi19110220-gs/yolov8n-pcb-local-comparison/releases/tag/v1.0.0). Access requires repository permission. Keep the ZIP in the repository's ignored `release-assets` folder, or pass its downloaded location explicitly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\prepare_dataset.ps1 -Archive "..\release-assets\pcb_yolo_train_val_v1.0.0.zip"
```

The script verifies the ZIP against the published [SHA-256 sidecar](reproducibility/manifests/dataset/pcb_yolo_train_val_v1.0.0.sha256) before extraction. It rejects unsafe archive members and a nonempty destination. It produces `dataset/pcb_yolo_dataset/images/pool` and `labels/pool`. Four preserved train/validation manifests select the exact model-specific rows, including OHEM repetitions and order. Do not manually split or reorganize the pool. The archive and local dataset remain outside Git. `-Python` can select an explicit interpreter; `-Destination` is available for extraction checks, but training expects the default package-local dataset location.

## 3. Run the comparison

First check all packaged inputs and materialize the authorities without training:

```powershell
python train_local.py --preflight-only
```

To deliberately train a new comparison:

```powershell
python train_local.py
```

The original always trains first from `weights/yolov8n.pt`, then Trial 044 starts from `weights/trial035_parent_best.pt`. Both receive a clean validation pass. The preserved parameters are in [the original config](reproducibility/configs/original/recorded_train_args.json) and [the Trial 044 config](reproducibility/configs/trial044_gpu_adaptation/historical_cpu_trial044.json). Trial 044 uses classification-head-only optimization with MPDIoU, 1024-pixel images, and its exact OHEM authority.

Each invocation uses a fresh timestamped directory under `results/local_vscode_comparison_*`. `--output-dir` selects another fresh directory. Relative custom output paths are resolved from the current terminal directory. No training command accepts a test split. If a new run fails at the exact enhanced-training boundary after completing the original, recover through the same entry point:

```powershell
python train_local.py --output-dir "results\local_vscode_comparison_YOUR_RUN" --resume-enhanced-only
```

Recovery checks the completed original checkpoint, recorded metrics, exact authorities, hashes, and failed stage before restarting Trial 044 from its parent. The archived `last.pt` checkpoints are historical evidence; stripped optimizer state does not guarantee exact optimizer resumption.

## 4. Regenerate the visuals or verify the package

Charts use only saved CSV/JSON evidence and do not run either model:

```powershell
python reproducibility\generate_charts.py
```

To regenerate validation plots with the preserved best checkpoints, choose a new empty output directory:

```powershell
python reproducibility\regenerate_validation.py --output-dir results\local_vscode_comparison_validation_new
```

This runs `split='val'`, original `imgsz=640`, Trial 044 `imgsz=1024`, `conf=0.001`, `iou=0.7`, `max_det=300`, `augment=False`, and `plots=True`. The packaged regenerated outputs are under `results/regenerated_validation`. Each model has checkpoint/authority hashes, timestamps, runtime details, metrics, and output hashes in `provenance.json`. New validation runs do not overwrite the recorded final metrics or the existing report. The report's original numbers remain traceable to their saved clean-validation JSON.

```powershell
python -m pytest reproducibility\tests -q
python reproducibility\scripts\verify\verify_package.py --root ..
python reproducibility\scripts\verify\verify_clone.py --root ..
```

The checks cover checksum extraction, exact authority materialization, guarded recovery, model hashes, relative report links, fresh-clone imports, and held-out-test exclusion. No training is part of the tests. To inspect implementation, start with [model_code](model_code); full configurations, notices, historical evidence, and packaging tools are under [reproducibility](reproducibility).

## Folder guide

| Location | Purpose |
| --- | --- |
| `train_local.py` | The beginner training entry point |
| `prepare_dataset.ps1` | Checksum-verified Release extraction |
| `weights/` | Original/Trial 044 best checkpoints and their two starting checkpoints |
| `model_code/` | Preserved YOLOv8n custom trainers and MPDIoU code |
| `results/charts/` | Balanced figures generated from recorded evidence |
| `results/regenerated_validation/` | Newly regenerated, clearly separated validation plots |
| `results/original/`, `results/trial044_gpu_adaptation/` | Preserved training histories |
| `reproducibility/` | Audit manifests, configs, notices, tests, tools, and archived last checkpoints |

Read the [private-use notice](reproducibility/PRIVATE_USE_NOTICE.md) and [dataset authorization](reproducibility/DATASET_PROVENANCE.md) before sharing. The package does not grant an open-source license or permission for public dataset redistribution.
