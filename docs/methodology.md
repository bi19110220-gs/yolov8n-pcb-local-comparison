# Methodology

## Original model

The original model starts from the official YOLOv8n checkpoint and trains for
100 epochs on the grouped-v1 training manifest at 640-pixel resolution. Model
selection and reporting use the grouped-v1 validation manifest.

## Enhanced model

The enhanced model starts from the Trial 035 parent checkpoint. Trial 044
preserves the recorded OHEM training view, 1024-pixel resolution,
classification-head-only optimization scope, and MPDIoU implementation. The
recorded CPU device is adapted to CUDA device 0 for the RTX 3080 run; other
listed training hyperparameters remain unchanged.

## Evaluation

Each best checkpoint receives a clean validation pass. Precision, recall, F1,
mAP50, mAP50-95, per-class AP, model size, parameter count, latency, and
training duration are recorded. No held-out test evaluation or test manifest
is packaged. There is no globally held-out image set in the combined release:
the original and enhanced authorities assign different roles to some images.
Their metrics are not a controlled same-data ablation.

## Reproducibility

Publication files use repository-relative paths. When a source record contains
a machine-specific path, the published copy is normalized and its original
SHA-256 is retained in the provenance manifest.
