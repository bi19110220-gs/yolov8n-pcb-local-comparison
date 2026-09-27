# YOLOv8n PCB defect detection comparison

A private, local Windows FYP project comparing the original YOLOv8n baseline with Trial 040.

## Start here

1. Open [PCB_Quality_Inspector.ipynb](custom_yolo_pcb/PCB_Quality_Inspector.ipynb) in VS Code.
2. Select the project `.venv` Python kernel.
3. Click **Run All** to verify the package, view the recorded comparison, generate `app.py`, and start the local PCB inspector.

The [beginner guide](custom_yolo_pcb/README.md) explains setup. The [results report](custom_yolo_pcb/RESULTS_REPORT.md) contains the matched validation results, training curves, per-class metrics, confusion matrices, and examples.

Private `v1.1.0` introduced the notebook and local app. The unchanged private `v1.0.0` release remains the source of the dataset archive.

Trial 040 and the original were evaluated on the same frozen 3,416-image grouped-v1 validation manifest with identical settings. Trial 040 reaches 0.814235 mAP50-95 versus 0.529946 for the original. Their training recipes differ, so this is a fair model comparison but not a one-variable ablation. No held-out test split was evaluated.
