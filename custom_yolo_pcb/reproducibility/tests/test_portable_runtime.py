from pathlib import Path

import pytest


def test_materialization_preserves_order_duplicates_in_canonical_pool(tmp_path):
    from tools import run_local_vscode_comparison as runner
    dataset = tmp_path / "extracted" / "pcb_yolo_dataset"
    for split, name in (("pool", "a.jpg"), ("pool", "b.jpg")):
        image = dataset / "images" / split / name
        label = dataset / "labels" / split / (Path(name).stem + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
        label.touch()
    manifest = tmp_path / "published.txt"
    manifest.write_text("pcb_yolo_dataset/images/pool/a.jpg\npcb_yolo_dataset/images/pool/b.jpg\npcb_yolo_dataset/images/pool/a.jpg\n", encoding="utf-8")
    target = tmp_path / "runtime.txt"
    runner.materialize_manifest(manifest, target, dataset_root=dataset)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines == [str((dataset / "images" / "pool" / name).resolve()) for name in ("a.jpg", "b.jpg", "a.jpg")]


@pytest.mark.parametrize("entry", [
    "pcb_yolo_dataset/images/" + "test/a.jpg",
    "pcb_yolo_dataset/images/train/../val/a.jpg",
    "pcb_yolo_dataset/images/train/a.jpg:stream",
    "pcb_yolo_dataset/images/train/missing.jpg",
    "unrelated/train/a.jpg",
])
def test_materialization_fails_closed_for_invalid_or_missing_inputs(tmp_path, entry):
    from tools import run_local_vscode_comparison as runner
    manifest = tmp_path / "published.txt"
    manifest.write_text(entry + "\n", encoding="utf-8")
    target = tmp_path / "runtime.txt"
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        runner.materialize_manifest(manifest, target, dataset_root=tmp_path)
    assert not target.exists()


def test_portable_inputs_exist_and_match_published_provenance():
    from tools import run_local_vscode_comparison as runner
    assert runner.DATASET_ROOT == runner.ROOT / "dataset" / "pcb_yolo_dataset"
    assert runner.verify_packaged_inputs()["test_split_used"] is False
