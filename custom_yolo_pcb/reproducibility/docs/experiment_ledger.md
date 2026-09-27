# Experiment ledger

| Entry | Role | Initialization | Training authority | Publication state |
| --- | --- | --- | --- | --- |
| Original grouped-v1 YOLOv8n | Baseline | Official `yolov8n.pt` | grouped-v1 train | Completed and packaged |
| Selected Enhanced YOLOv8n | Enhanced model; internal candidate Trial 040 | Official `yolov8n.pt` | Short x1.25 oversampled train | 330 epochs completed and packaged |
| Shared validation | Final comparison | Frozen best checkpoints | grouped-v1 validation, 3,416 images | Completed without training or test access |

The Selected Enhanced YOLOv8n is a single model. Another candidate's checkpoint is not used as its initialization; the internal Trial 040 identifier is retained only for traceability.
