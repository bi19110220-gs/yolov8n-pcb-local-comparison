# Dataset Provenance and Authorization

The repository owner confirmed authorized private redistribution of the PCB
dataset for this private academic project.

Version 1.0.0 distributes the union of the four exact model-specific training
and validation authorities as a checksum-attested private GitHub Release asset.
The models use different recorded training authorities: grouped-v1 for the
original model, and the recorded OHEM train view plus standard validation for
the enhanced model. OHEM order and repeated entries are preserved.

There is no globally held-out image set in this combined release: an image
held out under one split scheme may be used for training or validation under
the other. The release is not evidence of an independent common test set;
test evaluation and test manifests remain excluded. No held-out test evaluation
was run for this comparison.

The archive places each distinct authorized image and label once in
`pcb_yolo_dataset/images/pool` and `pcb_yolo_dataset/labels/pool`. Extract it
under `dataset`. Four manifests in `manifests/dataset/authority` select exact
model-specific roles from this pool. Original source hashes and normalized
publication hashes are recorded in `manifests/hashes/artifact_provenance.json`.
The standard-validation source hash identifies its sorted relative filename
inventory, because its original authority was a directory. The dataset Release
inventory separately attests image and label bytes.

The dataset archive must not be made public or redistributed beyond the
confirmed authorization without a separate rights review.
