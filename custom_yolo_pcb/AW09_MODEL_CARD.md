# AW09 dashboard model

AW09 ranks highest by clean validation mAP50–95 in the recorded 18-configuration YOLOv8n grid. This reporting criterion differs from the archived campaign's earlier constrained designation. The dashboard uses the checkpoint unchanged, with no training or metric recomputation.

| Property | Recorded value |
| --- | --- |
| Architecture | Six-class YOLOv8n |
| Optimizer | AdamW |
| Initial learning rate | 0.01 |
| Requested weight decay | 0.001 |
| Best epoch / completed epochs | 38 / 100 |
| Training image size / batch | 1024 / 3 |
| Dataset | balanced6000_20261007 |
| Images | 4,800 train / 600 validation / 600 test |
| Checkpoint | weights/aw09_best.pt |
| File size | 6,340,010 bytes |
| SHA-256 | 21650e4249bded2bcc974922c705a1ea29c066671017a00177ca8eca27a6a1c0 |

| Split | Precision | Recall | F1 | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| Validation | 96.74% | 97.72% | 97.23% | 98.85% | 53.89% |
| Test | 95.05% | 94.81% | 94.93% | 95.67% | 46.74% |

Full-precision overall and class measurements are in [aw09_metadata.json](weights/aw09_metadata.json). F1 is the harmonic mean of the reported overall precision and recall. Validation mAP50–95 is 0.5389454885300056; test is 0.4673596481511472.

## Provenance and inference

The deployment file is a byte-for-byte copy of `results/optimizer_grid18_20261012_193800/runs/AW09/weights/best.pt`. Metadata identifies the original training, clean-validation and final-test JSON records retained in the research workspace; full training runs and datasets are not included in this dashboard publication. Validation and test records identify the same checkpoint hash. The final test pass completed on 21 October 2026.

Class IDs: 0 missing_hole, 1 mouse_bite, 2 open_circuit, 3 short, 4 spur, 5 spurious_copper. Inference uses image size 1024, NMS IoU 0.70, max 300 detections and no augmentation. Decoded RGB PIL images preserve Ultralytics' expected colour conversion. Default display confidence 0.25 differs from the evaluation floor 0.001.

## Limits

Best overall does not mean best on every metric or category. Short-circuit test AP50–95 was 38.38% and recall 83.71%. One run per configuration and prior exposure of the existing test split limit generalization claims. Dataset results do not guarantee accuracy on a new upload. Boxes do not measure electrical continuity or certify a board as defect-free.

Trial040/grouped-v1 scores remain historical evidence and must not be attributed to AW09. The dashboard has no training path, model picker or automatic quality-control pass decision.
