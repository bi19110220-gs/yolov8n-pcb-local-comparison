# PCB Quality Inspector — local VS Code guide

This FYP package has one main file: [PCB_Quality_Inspector.ipynb](PCB_Quality_Inspector.ipynb). It shows the recorded Original-versus-Enhanced results, can run the complete training sequence when deliberately enabled, creates `app.py`, and opens the local PCB image inspector.

No Vercel or cloud deployment is used. Training, validation, and uploaded-image inference stay on your Windows PC.

## 1. One-time setup

Install Python 3.11 and VS Code. Open this `custom_yolo_pcb` folder in VS Code, then run these commands in its PowerShell terminal:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The recorded environment used Python 3.11.9, PyTorch 2.11.0+cu128, Ultralytics 8.4.84, and an NVIDIA RTX 3080. The code automatically uses CPU when CUDA is unavailable, but training will be much slower and hardware differences can change numerical results.

## 2. Download the dataset once

Download `pcb_yolo_train_val_v1.0.0.zip` from the private [v1.0.0 Release](https://github.com/bi19110220-gs/yolov8n-pcb-local-comparison/releases/tag/v1.0.0) and place it here:

```text
yolov8n-pcb-local-comparison/
└── release-assets/
    └── pcb_yolo_train_val_v1.0.0.zip
```

The notebook verifies the published SHA-256 before extracting the 1.13 GB dataset. If it is already prepared, it reuses it. The ZIP and extracted dataset remain outside Git.

## 3. Normal workflow: Select Kernel → Run All

1. Open `PCB_Quality_Inspector.ipynb`.
2. Click **Select Kernel** in VS Code.
3. Choose `custom_yolo_pcb\.venv\Scripts\python.exe`.
4. Click **Run All**.
5. Open <http://localhost:8501> if the browser does not open automatically.

The safe defaults are:

```python
RUN_TRAINING = False
RUN_RECOVERY = False
LAUNCH_STREAMLIT = True
RECOVERY_OUTPUT_DIR = None
```

With these defaults, Run All verifies the environment, dataset, model hashes, and validation-only authorities; shows the saved metric table and charts; regenerates the exact committed `app.py`; and starts Streamlit. It does not retrain.

To deliberately retrain, set only `RUN_TRAINING = True`. The guarded order is Original training → Original clean validation → Trial 044 training → Trial 044 clean validation → comparison export. Each run uses a new `results/notebook_runs/YYYYMMDD_HHMMSS/` folder and does not overwrite published evidence.

`RUN_RECOVERY` is only for an exact failed enhanced-stage notebook run. Set `RECOVERY_OUTPUT_DIR` to that existing run directory; all authority and checkpoint guards still apply.

## 4. Use the PCB image inspector

The Streamlit app loads only the included, hash-verified `weights/trial044_best.pt`. It detects all six classes:

- missing hole
- mouse bite
- open circuit
- short
- spur
- spurious copper

Upload one JPG, JPEG, or PNG image, set confidence from `0.05` to `0.95`, and click **Inspect PCB**. The fixed inference settings are 1024-pixel input, IoU 0.70, maximum 300 detections, and no test-time augmentation.

The app shows the annotated result first, the original image second, total detections, classes found, mean confidence, inference time, top defect, six per-class counts, and a confidence-sorted detection table. You can download an annotated PNG and CSV.

Uploads are limited to 10 MB and 20 megapixels. EXIF orientation is corrected, images are converted to RGB, and upload bytes stay in memory instead of being saved to disk. **No defects detected** only means no box met the chosen threshold; it is not a quality-control pass decision.

To start the same committed app without opening the notebook:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

## 5. Review results and verify the package

Read [RESULTS_REPORT.md](RESULTS_REPORT.md) for the saved training/validation curves, final metrics, per-class chart, confusion matrices, PR/F1/precision/recall curves, and validation examples. Trial 044 finished normally through early stopping: its best epoch was 12 and training stopped at epoch 27 after the configured patience was exhausted.

Run lightweight checks from this folder:

```powershell
& .\.venv\Scripts\python.exe -m pytest reproducibility\tests -q
& .\.venv\Scripts\python.exe reproducibility\scripts\verify\verify_package.py --root ..
& .\.venv\Scripts\python.exe reproducibility\scripts\verify\verify_clone.py --root ..
```

These checks do not train models or access a held-out test split.

## Important comparison limitation

Original and Trial 044 use different recorded training authorities, starting checkpoints, schedules, and image sizes. The results are a reproducible descriptive comparison, not a controlled same-data ablation. This package preserves the completed Trial 044 RTX 3080 adaptation. Held-out test evaluation was not run; all reported accuracy is validation-only, and future test evaluation remains a separate, explicitly authorized release event.

See [reproducibility details](reproducibility/README.md), the [private-use notice](reproducibility/PRIVATE_USE_NOTICE.md), and [dataset authorization](reproducibility/DATASET_PROVENANCE.md) before sharing.
