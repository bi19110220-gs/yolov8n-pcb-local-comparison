# Local VS Code YOLOv8n comparison

> Method warning: both checkpoints use the same frozen validation authority and settings, but their training authorities and recipes differ. This is not a one-variable ablation.

| Model | Training authority | Device | Precision | Recall | mAP50 | mAP50-95 | Short AP50-95 | Training hours |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_grouped_v1_yolov8n | grouped-v1 | cuda:0 | 0.963324 | 0.955543 | 0.973787 | 0.529946 | 0.536335 | 3.963 |
| enhanced_trial040 | Trial040 fresh YOLOv8n with Short x1.25 oversampled train view | cuda:0 | 0.995439 | 0.996155 | 0.994761 | 0.814235 | 0.838600 | 0.030 |

Held-out test evaluation was not run.

Generated: 2026-09-27T05:41:26.575684+00:00
