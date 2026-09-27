# Reproducibility details

The [notebook](../PCB_Quality_Inspector.ipynb) is the beginner-facing controller. This folder contains exact configurations, authorities, provenance, verification tests, and regeneration scripts.

`manifests/hashes/artifact_provenance.json` binds published files to SHA-256 values. Best weights are in `../weights`; final same-run checkpoints are retained once in `checkpoints`.

Run from `custom_yolo_pcb`:

```powershell
python -m pytest reproducibility\tests -q
python reproducibility\scripts\verify\verify_package.py --root ..
python reproducibility\scripts\verify\verify_clone.py --root ..
```

Publisher-only dataset rebuild:

```powershell
python reproducibility\scripts\package\build_dataset_release.py --dataset-root "<source dataset>" --original-train-manifest "<source>\original_grouped_v1_train.txt" --original-val-manifest "<source>\original_grouped_v1_val.txt" --enhanced-ohem-train-manifest "<source>\trial040_oversampled_train.txt" --enhanced-standard-val-authority "<source validation manifest>" --output ..\release-assets\pcb_yolo_train_val_v1.0.0.zip --manifest-dir reproducibility\manifests\dataset
```

The final comparison uses the original grouped-v1 validation manifest for both checkpoints. Test evaluation remains outside this package.
