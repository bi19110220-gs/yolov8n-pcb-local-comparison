# AW09 PCB Quality Inspector — local guide

Use the included AW09 YOLOv8n model to inspect images locally. No dataset download or model training is required.

## One-time setup

Open this folder in a terminal. With Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The app uses CUDA device 0 when PyTorch reports CUDA available, otherwise CPU. CPU inspection is supported but can be slower. The recorded environment used Python 3.11.9, PyTorch 2.11.0+cu128, Ultralytics 8.4.84 and an RTX 3080. For that NVIDIA environment, install the matching PyTorch build before the requirements:

```powershell
& .\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Run the dashboard

```powershell
& .\.venv\Scripts\python.exe app.py
```

With the environment activated, simply use `python app.py`. This starts Streamlit on loopback and opens your browser, normally at http://127.0.0.1:8501. Keep the terminal running; press Ctrl+C to stop. If the browser does not open automatically, use the URL printed in the terminal.

For an occupied port: `python app.py --server.port=8502`. Standard Streamlit launch also works: `python -m streamlit run app.py --server.address=127.0.0.1`.

## Inspect an image

1. Upload one JPG/JPEG/PNG, up to 10 MiB and 20 megapixels decoded.
2. Set confidence between 0.05 and 0.95 (default 0.25).
3. Click **Inspect PCB** to view boxes, six-class counts, a short-circuit indicator and detection details.
4. Compare original and annotated images and download the annotated PNG or detection CSV.

After changing confidence, click **Inspect PCB** again. A notice identifies results generated at an earlier threshold. Uploading a new image clears the previous result. Images and downloads remain in session memory rather than an upload directory.

No detections means no boxes met the chosen threshold; it is not a quality-control pass or electrical-continuity measurement. Prediction time includes preprocessing, prediction and postprocessing, excluding first model load, image decoding and display rendering.

## Model and evidence

The only dashboard model is `weights/aw09_best.pt`, checked against its fixed SHA-256 before loading. Restore the published file if verification fails. The model panel displays saved results from `weights/aw09_metadata.json`; it does not measure accuracy on your uploaded image. See [AW09_MODEL_CARD.md](AW09_MODEL_CARD.md).

The older notebook and [RESULTS_REPORT.md](RESULTS_REPORT.md) preserve the historical Original/Trial040 grouped-v1 comparison, separate from AW09's balanced6000 experiment. Optional training defaults remain off. The notebook generation step uses `model_code/app_template.py`, so regenerating `app.py` retains AW09. Use the direct launch command when you only need the dashboard.

## Developer checks

```powershell
& .\.venv\Scripts\python.exe model_code\generate_app.py --output app.py
& .\.venv\Scripts\python.exe -m pytest reproducibility\tests\test_inspector_core.py reproducibility\tests\test_app_generation.py reproducibility\tests\test_aw09_dashboard.py reproducibility\tests\test_notebook_contract.py -q
```

These checks do not train or evaluate dataset splits. Edit the canonical template rather than generated `app.py` alone. Keep historical Trial040 APIs used by the comparison notebook.
