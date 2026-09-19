"""Build the deterministic beginner-facing VS Code notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PACKAGE = Path(__file__).resolve().parents[2]
OUTPUT = PACKAGE / "PCB_Quality_Inspector.ipynb"


def markdown(source: str, cell_id: str):
    return new_markdown_cell(source.strip() + "\n", id=cell_id)


def code(source: str, cell_id: str):
    return new_code_cell(source.strip() + "\n", id=cell_id)


def build_notebook():
    cells = [
        markdown(
            """
# PCB Quality Inspector — Enhanced YOLOv8n

This is the main local Windows/VS Code workflow for the FYP comparison. It verifies the recorded Original YOLOv8n and Enhanced Trial 044 evidence, shows the saved validation results, optionally runs the complete guarded comparison, generates `app.py`, and starts the local six-class PCB inspector.

**Scientific boundary:** the two models use different recorded training authorities, so this is not a controlled same-data ablation. Reported metrics are validation-only. The held-out test split is not used.
            """,
            "goal",
        ),
        code(
            """
# Safe beginner defaults. Change only these values, then click Run All.
RUN_TRAINING = False
RUN_RECOVERY = False
LAUNCH_STREAMLIT = True
RECOVERY_OUTPUT_DIR = None

# Internal verification mode is used only by the repository's notebook check.
import os
PCB_NOTEBOOK_VERIFY_ONLY = os.environ.get("PCB_NOTEBOOK_VERIFY_ONLY") == "1"
            """,
            "configuration",
        ),
        markdown(
            """
## 1. Setup and kernel check

In VS Code, click **Select Kernel** and choose `custom_yolo_pcb\\.venv\\Scripts\\python.exe`. The next cell stops with a friendly message if another interpreter is selected.
            """,
            "setup-heading",
        ),
        code(
            """
import sys
from pathlib import Path
from IPython.display import Markdown, display
import pandas as pd

PACKAGE = Path.cwd().resolve()
if not (PACKAGE / "model_code").is_dir():
    raise RuntimeError("Open PCB_Quality_Inspector.ipynb from the custom_yolo_pcb folder in VS Code.")
sys.path.insert(0, str(PACKAGE / "model_code"))

from generate_app import write_generated_app
from notebook_workflow import (
    DATASET_RELEASE_URL,
    STREAMLIT_URL,
    DatasetUnavailable,
    display_recorded_results,
    ensure_dataset_ready,
    environment_summary,
    run_complete_comparison,
    run_enhanced_recovery,
    start_streamlit,
    stop_streamlit,
    verify_project_kernel,
    verify_publication_inputs,
)

verify_project_kernel(PACKAGE)
print("Project .venv kernel verified.")
            """,
            "kernel-check",
        ),
        code(
            """
environment = environment_summary()
visible_environment = {
    key: value for key, value in environment.items() if key != "python_executable"
}
display(pd.DataFrame(visible_environment.items(), columns=["Environment", "Value"]))
            """,
            "environment-summary",
        ),
        markdown(
            """
## 2. Dataset and publication checks

The notebook reuses a prepared dataset. If it is missing, it verifies and extracts `release-assets/pcb_yolo_train_val_v1.0.0.zip`. If neither is available, it shows the private v1.0.0 Release link and stops before training. No GitHub token is requested or stored.
            """,
            "checks-heading",
        ),
        code(
            """
try:
    dataset_status = ensure_dataset_ready(PACKAGE)
    print(f"Dataset status: {dataset_status['status']}")
except DatasetUnavailable as error:
    dataset_status = None
    print(error)
    if not PCB_NOTEBOOK_VERIFY_ONLY:
        raise
            """,
            "dataset-check",
        ),
        code(
            """
publication_hashes = verify_publication_inputs(PACKAGE)
display(pd.DataFrame(publication_hashes.items(), columns=["Verified input", "SHA-256 / status"]))
            """,
            "publication-check",
        ),
        markdown(
            """
## 3. Recorded comparison results

These are the packaged validation results and charts. Opening the notebook does not retrain either model.
            """,
            "results-heading",
        ),
        code(
            """
recorded = display_recorded_results(PACKAGE)
metric_columns = [
    "model", "training_authority", "precision", "recall", "f1",
    "map50", "map50_95", "short_ap50_95", "inference_latency_ms",
    "parameters", "model_size_mb", "training_seconds",
]
metrics = pd.DataFrame(recorded["metrics"])[metric_columns]
for column in ("precision", "recall", "f1", "map50", "map50_95", "short_ap50_95"):
    metrics[column] = pd.to_numeric(metrics[column]).round(4)
display(metrics)
            """,
            "metric-table",
        ),
        code(
            """
for chart in recorded["charts"]:
    title = chart.stem.replace("_", " ").title()
    relative_chart = chart.relative_to(PACKAGE).as_posix()
    display(Markdown(f"### {title}"))
    display(Markdown(f"![{title}]({relative_chart})"))
            """,
            "recorded-charts",
        ),
        markdown(
            """
## 4. Optional complete local training

Leave `RUN_TRAINING = False` to use the included evidence. Setting it to `True` runs the preserved sequence once: Original training → Original clean validation → Enhanced Trial 044 training → Enhanced clean validation → comparison export. Output goes to a new `results/notebook_runs/YYYYMMDD_HHMMSS/` directory, and detailed logs stay in a sibling local log file instead of filling this notebook.
            """,
            "training-heading",
        ),
        code(
            """
if RUN_TRAINING:
    if dataset_status is None:
        raise RuntimeError("Prepare the private v1.0.0 dataset before enabling training.")
    completed_output = run_complete_comparison(PACKAGE)
    print(f"Complete comparison saved to: {completed_output.relative_to(PACKAGE)}")
else:
    print("Training skipped (RUN_TRAINING=False). Included weights and results remain unchanged.")
            """,
            "optional-training",
        ),
        code(
            """
if RUN_RECOVERY:
    recovered_output = run_enhanced_recovery(PACKAGE, RECOVERY_OUTPUT_DIR)
    print(f"Enhanced recovery completed in: {recovered_output.relative_to(PACKAGE)}")
else:
    print("Recovery skipped (RUN_RECOVERY=False).")
            """,
            "optional-recovery",
        ),
        markdown(
            """
## 5. Generate and start the local inspector

The notebook is the canonical source for `app.py`. The generator reproduces the committed file byte-for-byte and refuses to overwrite unrelated handwritten content. The app loads only the included, hash-verified `weights/trial044_best.pt` checkpoint.
            """,
            "app-heading",
        ),
        code(
            """
generated_app = write_generated_app(PACKAGE / "app.py")
print(f"app.py SHA-256: {generated_app.sha256}")
print("app.py was regenerated." if generated_app.replaced else "app.py already matches the notebook generator.")
            """,
            "generate-app",
        ),
        code(
            """
if LAUNCH_STREAMLIT and not PCB_NOTEBOOK_VERIFY_ONLY:
    import webbrowser
    streamlit_server = start_streamlit(PACKAGE)
    display(Markdown(f"**Streamlit:** [{STREAMLIT_URL}]({STREAMLIT_URL}) — {streamlit_server['status']}"))
    webbrowser.open(STREAMLIT_URL)
elif PCB_NOTEBOOK_VERIFY_ONLY:
    print("Streamlit launch skipped only for notebook verification; LAUNCH_STREAMLIT remains True.")
else:
    print("Streamlit launch skipped (LAUNCH_STREAMLIT=False).")
            """,
            "launch-streamlit",
        ),
        markdown(
            """
You can also start the identical app later from the VS Code terminal with:

```powershell
& .\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Open <http://localhost:8501>, upload a JPG/JPEG/PNG image, choose the confidence threshold, and click **Inspect PCB**. The app keeps the upload in memory and provides annotated-PNG and CSV downloads.
            """,
            "direct-app",
        ),
        code(
            """
# Optional stop control. Change to True and run only this cell when finished.
STOP_STREAMLIT = False
if STOP_STREAMLIT:
    print(stop_streamlit())
else:
    print("Streamlit remains available. Set STOP_STREAMLIT=True and run this cell to stop only the notebook-started process.")
            """,
            "stop-streamlit",
        ),
        markdown(
            """
## 6. Interpretation

- **Defects detected — review required** means one or more boxes met the selected confidence threshold.
- **No defects detected** means no boxes met that threshold; it is not a statement that the PCB passed quality control.
- The app uses Enhanced Trial 044 only. The Original model remains in the notebook comparison and training sequence.
- Uploaded-image predictions do not change the recorded validation metrics.
            """,
            "interpretation",
        ),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )
    nbformat.validate(notebook)
    return notebook


def main() -> int:
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT.name} with {len(notebook.cells)} cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
