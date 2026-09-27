# PCB Quality Inspector — local VS Code guide

This Windows package compares the original YOLOv8n baseline with Trial 040, can deliberately rerun both training recipes, and provides a local Streamlit PCB inspector using the hash-verified Trial 040 checkpoint. It is local only; Vercel is not used.

## 1. One-time setup

Open this folder in VS Code and run:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The recorded environment used Python 3.11.9, PyTorch 2.11.0+cu128, Ultralytics 8.4.84, and an NVIDIA RTX 3080. CPU fallback is supported but training is much slower.

## 2. Prepare the dataset

Download `pcb_yolo_train_val_v1.0.0.zip` from the private [v1.0.0 release](https://github.com/bi19110220-gs/yolov8n-pcb-local-comparison/releases/tag/v1.0.0) and place it at `release-assets/pcb_yolo_train_val_v1.0.0.zip`. The notebook verifies its SHA-256 before extraction. The ZIP and extracted data remain outside Git.

## 3. Normal workflow

Open `PCB_Quality_Inspector.ipynb`, click **Select Kernel**, choose `custom_yolo_pcb\.venv\Scripts\python.exe`, and click **Run All**. Safe defaults do not train:

```python
RUN_TRAINING = False
RUN_RECOVERY = False
LAUNCH_STREAMLIT = True
RECOVERY_OUTPUT_DIR = None
```

Set `RUN_TRAINING = True` only when a full rerun is intended. Trial 040 starts from the packaged official `yolov8n.pt`; it does not start from another trial's best checkpoint. A recovery may resume only the same failed run from its own state. Final comparison validation uses the same grouped-v1 validation manifest and settings for both models, so it is fair for model selection but not a one-variable ablation.

## 4. Use the inspector

The app loads `weights/trial040_best.pt` after verifying SHA-256. It detects missing hole, mouse bite, open circuit, short, spur, and spurious copper. Input size is 1024, IoU is 0.70, maximum detections are 300, and test-time augmentation is off.

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

“No defects detected” means only that no prediction met the chosen confidence threshold; it is not a quality-control pass.

## 5. Verify

```powershell
& .\.venv\Scripts\python.exe -m pytest reproducibility\tests -q
& .\.venv\Scripts\python.exe reproducibility\scripts\verify\verify_package.py --root ..
& .\.venv\Scripts\python.exe reproducibility\scripts\verify\verify_clone.py --root ..
```

These checks do not train models or access a held-out test split. See [RESULTS_REPORT.md](RESULTS_REPORT.md) and [reproducibility details](reproducibility/README.md).
