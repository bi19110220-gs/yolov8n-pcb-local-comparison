# Examiner overview

## Research question

Does the selected Trial 040 YOLOv8n model outperform the original YOLOv8n baseline on one shared validation authority?

## Controlled final evaluation

- Both are six-class YOLOv8n detectors.
- Both frozen best checkpoints use the same 3,416 validation images.
- Image size, batch, confidence, IoU, maximum detections, augmentation, workers, and evaluation code are identical.
- All reported comparison metrics are validation-only.

## Not controlled

The training recipes differ in training view, image size, schedule, and hyperparameters. The comparison establishes which final model performs better under the shared evaluation contract; it does not isolate the cause.

Trial 040 starts from official `yolov8n.pt`. It does not continue from another trial's best checkpoint.
