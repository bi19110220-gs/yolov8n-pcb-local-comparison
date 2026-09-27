# Experiment context

## Original model

The original model starts from official YOLOv8n and trains for 100 epochs on the grouped-v1 training manifest at 640-pixel resolution.

## Selected Enhanced YOLOv8n

The Selected Enhanced YOLOv8n starts from official `yolov8n.pt`, not another candidate checkpoint. It trains for 330 epochs at 1024 pixels with AdamW and a Short x1.25 oversampled training view. Recovery may continue only the same run from its own last checkpoint. The internal Trial 040 identifier is retained for provenance.

## Evaluation

Both frozen best checkpoints are evaluated on the same grouped-v1 validation manifest with identical settings. Precision, recall, F1, mAP50, mAP50-95, per-class AP, model size, parameter count, and latency are recorded. No held-out test evaluation is packaged. Different training authorities mean the result is a model comparison, not a one-variable ablation.
