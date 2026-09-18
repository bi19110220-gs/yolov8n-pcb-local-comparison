# Local VS Code YOLOv8n comparison

> Method warning: the original and enhanced runs intentionally use different recorded training authorities. This is a reproduction comparison, not a same-data controlled ablation.

| Model | Training authority | Device | Precision | Recall | mAP50 | mAP50-95 | Short AP50-95 | Training hours |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_grouped_v1_yolov8n | grouped-v1 | cuda:0 | 0.976060 | 0.973197 | 0.981800 | 0.555061 | 0.566816 | 3.963 |
| enhanced_trial044_gpu_adaptation | Trial044 OHEM train view plus standard validation | cuda:0 | 0.994192 | 0.991924 | 0.992828 | 0.758482 | 0.763543 | 2.056 |

Held-out test evaluation was not run.

Generated: 2026-09-17T15:50:26.653438+00:00
