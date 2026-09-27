# Experiment ledger

| Entry | Role | Initialization | Training authority | Publication state |
| --- | --- | --- | --- | --- |
| Original grouped-v1 YOLOv8n | Baseline | Official `yolov8n.pt` | grouped-v1 train | Completed and packaged |
| Trial 040 | Selected enhanced model | Official `yolov8n.pt` | Short x1.25 oversampled train | 330 epochs completed and packaged |
| Shared validation | Final comparison | Frozen best checkpoints | grouped-v1 validation, 3,416 images | Completed without training or test access |

Trial 040 is a single model. Another trial's checkpoint is not used as its initialization.
