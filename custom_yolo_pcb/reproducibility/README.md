# Reproducibility details

The beginner interface is [one level above](../README.md), with [the notebook](../PCB_Quality_Inspector.ipynb) as the primary controller. This folder retains the exact configurations, authorities, provenance, notices, tests, internal notebook builder, and historical context behind the reported comparison.

`manifests/hashes/artifact_provenance.json` preserves source and publication SHA-256 values; `previous_destination` records relocation without altering the source evidence. `checkpoints` retains each final `last.pt` once. Trained best weights and starting weights are in `../weights`. Historical tooling snapshots are inactive reference material, not setup instructions.

Run commands from the `custom_yolo_pcb` directory. Verification:

```powershell
python -m pytest reproducibility\tests -q
python reproducibility\scripts\verify\verify_package.py --root ..
python reproducibility\scripts\verify\verify_clone.py --root ..
```

Publisher-only dataset rebuild uses the original authorized research dataset and original source authorities, never the published pool manifests:

```powershell
python reproducibility\scripts\package\build_dataset_release.py --dataset-root "<source dataset>" --original-train-manifest "<source run>\authority\original_grouped_v1_train.txt" --original-val-manifest "<source run>\authority\original_grouped_v1_val.txt" --enhanced-ohem-train-manifest "<source run>\authority\enhanced_trial044_ohem_train.txt" --enhanced-standard-val-authority "<source dataset>\images\val" --output ..\release-assets\pcb_yolo_train_val_v1.0.0.zip --manifest-dir reproducibility\manifests\dataset
```

The preserved result exporter `scripts/package/package_comparison.py` accepts `--source-root`, `--run-root`, and `--destination-root`. The destination means a package directory (the equivalent of `custom_yolo_pcb`), and must be a separate staging location for a new export. It requires a terminal completed run and copies evidence; it does not train, publish a Release, or overwrite this report automatically. Historical path placeholders are source provenance, not executable paths.

Exported runtimes include `PCB_Quality_Inspector.ipynb`, generated `app.py`, the notebook/app helper modules, `train_local.py`, `prepare_dataset.ps1`, requirements, and the published dataset checksum. The generator must reproduce `app.py` byte-for-byte. The full-repository verification wrapper remains in this checkout because a staging export does not contain the complete Git repository or test suite. Run the verification commands above against this checked-in package, not an evidence-only staging export.

See [experiment context](docs/experiment_context.md), [experiment ledger](docs/experiment_ledger.md), [dataset provenance](DATASET_PROVENANCE.md), and [private-use notice](PRIVATE_USE_NOTICE.md).
