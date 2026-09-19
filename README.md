# YOLOv8n PCB defect detection comparison

A private, local Windows FYP project comparing Original YOLOv8n with the Enhanced Trial 044 model.

## Start here

1. Open [PCB_Quality_Inspector.ipynb](custom_yolo_pcb/PCB_Quality_Inspector.ipynb) in VS Code.
2. Select the project `.venv` Python kernel.
3. Click **Run All** to verify the package, view the recorded comparison, generate `app.py`, and start the local PCB inspector.

The [beginner guide](custom_yolo_pcb/README.md) explains the one-time setup. The [results report](custom_yolo_pcb/RESULTS_REPORT.md) contains training curves, validation metrics, per-class results, confusion matrices, and examples.

Private `v1.1.0` contains the notebook and local Streamlit application. The unchanged private [v1.0.0 Release](https://github.com/bi19110220-gs/yolov8n-pcb-local-comparison/releases/tag/v1.0.0) remains the single source for the 1.13 GB dataset ZIP; it is not uploaded again.

The models use different recorded training authorities, so the comparison is descriptive rather than a controlled same-data ablation. Held-out test evaluation was not run; all reported performance is validation-only.
