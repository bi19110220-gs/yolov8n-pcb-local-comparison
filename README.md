# AW09 PCB Quality Inspector

A local Streamlit PCB inspection dashboard using **AW09 YOLOv8n**. This GitHub repository is public; the dashboard runs on your own computer.

## Start here

1. Open `custom_yolo_pcb` and follow the [setup guide](custom_yolo_pcb/README.md).
2. Run `python app.py` with the project Python environment.
3. The dashboard opens locally in your browser. Upload a JPG/PNG, choose confidence and click **Inspect PCB**. Stop it with Ctrl+C in the terminal.

No dataset download, notebook execution, training or cloud deployment is required. The app includes hash-verified AW09 weights, six-class detection, a short-circuit indicator, annotated PNG/CSV downloads and recorded validation/test results.

AW09 uses AdamW, learning rate 0.01 and weight decay 0.001. Its epoch-38 checkpoint ranks highest by clean validation mAP50–95 in the completed 18-run balanced6000 grid: **53.89% validation** and **46.74% test**. See the [model card](custom_yolo_pcb/AW09_MODEL_CARD.md) for provenance and limits.

## Historical comparison notebook

[PCB_Quality_Inspector.ipynb](custom_yolo_pcb/PCB_Quality_Inspector.ipynb) preserves the earlier Original/Trial040 grouped-v1 comparison. It also regenerates the current AW09 dashboard. Optional historical training remains disabled by default and is separate from dashboard use.

The [historical results report](custom_yolo_pcb/RESULTS_REPORT.md) compares Original and Trial040 on 3,416 grouped-v1 validation images: mAP50–95 of 0.529946 and 0.814235, respectively. Their training recipes differ; this is not a one-variable ablation. That historical comparison did not evaluate a held-out test split. Its metrics must not be relabelled as AW09 results, which use a different dataset preparation. The earlier v1.0.0 dataset release is needed only for the historical notebook workflow.
