# YOLOv8n PCB Local Comparison

This private repository packages a reproducible Windows and VS Code workflow
for comparing a fresh grouped-v1 YOLOv8n baseline with a Trial 044 OHEM RTX
3080 adaptation on a six-class PCB defect dataset.

## Interpretation boundary

The models use **different recorded training authorities** and this comparison
is **not a controlled same-data ablation**. The historical Trial 044 stage ran
on CPU. The packaged enhanced run is an **RTX 3080 adaptation**, so
hardware-level numerical differences may change its metrics.

**Held-out test evaluation was not run** for version 1.0.0. All reported model
selection evidence is validation-only. There is **no globally held-out image
set** in the combined release: the two split schemes assign different roles
to some of the same images. Test evaluation and test manifests remain excluded.

## Documentation

- [Examiner overview](docs/examiner_overview.md)
- [Methodology](docs/methodology.md)
- [Experiment ledger](docs/experiment_ledger.md)
- [Dataset provenance](DATASET_PROVENANCE.md)
- [Private-use notice](PRIVATE_USE_NOTICE.md)

## Supported environment

- Windows 11
- Python 3.11
- VS Code
- NVIDIA RTX 3080 or another CUDA-capable GPU
- PyTorch 2.11.0+cu128 in the recorded environment
- Ultralytics 8.4.84

Google Colab is intentionally unsupported.

## Installation

Open PowerShell in the repository root:

~~~powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install "torch==2.11.0+cu128" torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

These commands install the recorded CUDA 12.8 PyTorch build from the official
PyTorch index. The NVIDIA driver must support that runtime. Verify it before
training:

~~~powershell
& .\.venv\Scripts\python.exe -c "import sys, torch, ultralytics; assert sys.version_info[:2] == (3, 11); assert torch.__version__ == '2.11.0+cu128'; assert ultralytics.__version__ == '8.4.84'; assert torch.cuda.is_available(); print(sys.executable); print(torch.cuda.get_device_name(0))"
~~~

VS Code tasks and PowerShell launchers use this repository's
`.venv\Scripts\python.exe` explicitly and fail if it is missing; activating
another shell environment does not change the selected interpreter. Package
installer integrity is handled by pip; this repository does not supply a
hash-pinned dependency lock file.

## Dataset Release asset

Version 1.0.0 uses a private Release asset named
`pcb_yolo_train_val_v1.0.0.zip`. It contains the union of the four exact
model-specific training and validation authorities. Download it into `release-assets`, verify it against
`manifests/dataset/pcb_yolo_train_val_v1.0.0.sha256`, and extract it to the
ignored `dataset` directory. The resulting layout is
`dataset/pcb_yolo_dataset/images/pool` and `dataset/pcb_yolo_dataset/labels/pool`.
The four tracked authority manifests select the original train/validation,
enhanced OHEM train, and enhanced standard-validation records; OHEM duplicate
entries and their order remain unchanged.

After downloading the private Release asset, run:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\prepare_dataset.ps1
~~~

This verifies the ZIP's SHA-256 against the tracked checksum **before**
extracting it. It rejects unexpected member paths and an existing nonempty
dataset directory. Use `-Archive <downloaded-zip-path>` if the file is elsewhere.

For publishers rebuilding the archive, use the original research dataset and
original source authorities, replacing the two explicit placeholders below.
Do not pass the published pool manifests: they describe the extracted Release,
not the original physical source tree.

~~~powershell
$SourceDataset = "<original research dataset directory containing images and labels>"
$SourceRun = "<completed source comparison run directory containing authority>"
& .\.venv\Scripts\python.exe scripts\package\build_dataset_release.py --dataset-root $SourceDataset --original-train-manifest "$SourceRun\authority\original_grouped_v1_train.txt" --original-val-manifest "$SourceRun\authority\original_grouped_v1_val.txt" --enhanced-ohem-train-manifest "$SourceRun\authority\enhanced_trial044_ohem_train.txt" --enhanced-standard-val-authority "$SourceDataset\images\val" --output release-assets\pcb_yolo_train_val_v1.0.0.zip --manifest-dir manifests\dataset
~~~

The command stages the archive and its inventory/checksum before publishing
them. Test evaluation and test manifests are excluded; image membership is
determined by the model-specific training and validation authorities.

## VS Code workflow

Use **Terminal → Run Task**:

1. `Verify reproducibility package`
2. `Train original then Trial 044 GPU adaptation`
3. `Resume Trial 044 enhanced stage only` only when the recorded run state is
   the exact failed enhanced-training boundary.

The original model always runs first in a new comparison. The enhanced-only
recovery command reuses a completed original and cannot rerun it.

## Command-line verification

~~~powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe scripts\verify\verify_package.py --root .
& .\.venv\Scripts\python.exe scripts\verify\verify_clone.py --root .
& .\.venv\Scripts\python.exe -m tools.run_local_vscode_comparison --output-dir results\local_vscode_comparison_preflight --preflight-only
~~~

The verification gate rejects absolute user paths, credential-like values,
broken relative documentation links, oversized Git candidates, non-false
`test_split_used` records, and test-directory content. The clone check verifies
Git-visible files and Python imports without training. Preflight requires the
extracted dataset and materializes all four exact authorities without CUDA or
training. Use a fresh output directory for each preflight or training run.

The canonical Python import context is the repository root: invoke
`python -m tools.run_local_vscode_comparison`, not a script copied elsewhere.
The root `pcb_mpdiou_loss.py` supports the custom trainer import contract.
Published historical YAML, authority records, and training arguments are
sanitized evidence and may contain redaction placeholders. The launcher reads
the four portable text manifests and builds runnable local YAML in its new
output directory; do not launch directly from the historical YAML files.
`models/official/yolov8n.pt` and `models/trial035_parent/best.pt` are the starting
checkpoints. Both trained model directories include `best.pt` and the recorded
`last.pt`; these historical final files may have optimizer state stripped by
Ultralytics and do not promise exact mid-epoch resumption. The VS Code enhanced
recovery task restarts the failed enhanced stage from its Trial 035 authority.

## Results

Final validation metrics, plots, checkpoints, hashes, and comparison tables are
copied only after both stages reach a terminal completed state. The packaging
command refuses to run while training is incomplete.

Comparison training duration is the final cumulative `time` value from each
training `results.csv`; it excludes the separate clean-validation pass. Source
records retain their complete stage start/end timestamps independently.
