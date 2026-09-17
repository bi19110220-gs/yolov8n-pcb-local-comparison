import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.package import build_dataset_release as release_builder


def _write_sample(root: Path, split: str, name: str, image: bytes, label: bytes) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "images" / split / name).write_bytes(image)
    (root / "labels" / split / f"{Path(name).stem}.txt").write_bytes(label)


def _write_manifest(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _authority_inputs(root: Path, workspace: Path) -> dict[str, Path]:
    _write_sample(root, "train", "alpha.jpg", b"alpha-image", b"alpha-label")
    _write_sample(root, "train", "beta.jpg", b"beta-image", b"beta-label")
    _write_sample(root, "val", "delta.jpg", b"delta-image", b"delta-label")
    _write_sample(root, "val", "gamma.jpg", b"gamma-image", b"gamma-label")
    authority = workspace / "authority"
    return {
        "original_train_manifest": _write_manifest(
            authority / "original_grouped_v1_train.txt",
            ["images/train/alpha.jpg", "images/train/beta.jpg"],
        ),
        "original_val_manifest": _write_manifest(
            authority / "original_grouped_v1_val.txt", ["images/val/gamma.jpg"]
        ),
        "enhanced_ohem_train_manifest": _write_manifest(
            authority / "enhanced_trial044_ohem_train.txt",
            ["images/train/beta.jpg", "images/train/alpha.jpg", "images/train/beta.jpg"],
        ),
        "enhanced_standard_val_authority": root / "images" / "val",
    }


def _build(root: Path, workspace: Path, **overrides: Path) -> tuple[dict, Path, Path]:
    inputs = _authority_inputs(root, workspace)
    inputs.update(overrides)
    archive = workspace / "release-assets" / "pcb_yolo_train_val_v1.0.0.zip"
    manifest_dir = workspace / "manifests" / "dataset"
    record = release_builder.build_dataset_release(root, archive, manifest_dir, **inputs)
    return record, archive, manifest_dir


def test_release_builds_neutral_deduplicated_pool_and_ordered_authorities(tmp_path: Path):
    root = tmp_path / "dataset"

    record, archive, manifest_dir = _build(root, tmp_path)

    assert record["schema"] == "pcb_yolo_authority_pool_dataset/v1"
    assert record["test_split_included"] is False
    assert record["pool"]["sample_count"] == 4
    assert record["pool"]["image_count"] == 4
    assert record["pool"]["label_count"] == 4
    assert list(record["authority_memberships"]) == [
        "original_grouped_v1_train.txt",
        "original_grouped_v1_val.txt",
        "enhanced_trial044_ohem_train.txt",
        "enhanced_standard_val.txt",
    ]
    assert record["authority_memberships"]["enhanced_trial044_ohem_train.txt"]["members"] == [
        {"image": "images/pool/beta.jpg", "label": "labels/pool/beta.txt"},
        {"image": "images/pool/alpha.jpg", "label": "labels/pool/alpha.txt"},
        {"image": "images/pool/beta.jpg", "label": "labels/pool/beta.txt"},
    ]
    assert record["authority_memberships"]["enhanced_standard_val.txt"]["members"] == [
        {"image": "images/pool/delta.jpg", "label": "labels/pool/delta.txt"},
        {"image": "images/pool/gamma.jpg", "label": "labels/pool/gamma.txt"},
    ]
    ohem = record["authority_memberships"]["enhanced_trial044_ohem_train.txt"]
    assert len(ohem["source_sha256"]) == 64
    assert len(ohem["published_sha256"]) == 64
    assert ohem["source_row_count"] == 3
    assert ohem["published_row_count"] == 3
    with ZipFile(archive) as handle:
        assert handle.namelist() == [
            "pcb_yolo_dataset/images/pool/alpha.jpg",
            "pcb_yolo_dataset/images/pool/beta.jpg",
            "pcb_yolo_dataset/images/pool/delta.jpg",
            "pcb_yolo_dataset/images/pool/gamma.jpg",
            "pcb_yolo_dataset/labels/pool/alpha.txt",
            "pcb_yolo_dataset/labels/pool/beta.txt",
            "pcb_yolo_dataset/labels/pool/delta.txt",
            "pcb_yolo_dataset/labels/pool/gamma.txt",
        ]
    published = json.loads(
        (manifest_dir / "pcb_yolo_train_val_v1.0.0.manifest.json").read_text(encoding="utf-8")
    )
    assert published == record
    assert all("source_path" not in entry for entry in record["pool"]["images"])


def test_pool_manifest_attests_exact_zip_member_bytes(tmp_path: Path):
    root = tmp_path / "dataset"

    record, archive, _ = _build(root, tmp_path)

    with ZipFile(archive) as handle:
        for kind in ("images", "labels"):
            for entry in record["pool"][kind]:
                payload = handle.read(f"pcb_yolo_dataset/{entry['path']}")
                assert len(payload) == entry["size_bytes"]
                assert hashlib.sha256(payload).hexdigest() == entry["source_sha256"]
                assert entry["published_sha256"] == entry["source_sha256"]


def test_repeated_authority_pool_archives_are_byte_identical(tmp_path: Path):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    first_archive = tmp_path / "first.zip"
    second_archive = tmp_path / "second.zip"

    release_builder.build_dataset_release(root, first_archive, tmp_path / "one", **inputs)
    release_builder.build_dataset_release(root, second_archive, tmp_path / "two", **inputs)

    assert first_archive.read_bytes() == second_archive.read_bytes()


def test_generated_standard_val_manifest_is_sorted_for_runtime_materialization(tmp_path: Path):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    generated = _write_manifest(
        tmp_path / "authority" / "generated-standard-val.txt",
        ["images/val/gamma.jpg", "images/val/delta.jpg"],
    )

    record, _, _ = _build(
        root,
        tmp_path,
        enhanced_standard_val_authority=generated,
    )

    assert record["authority_memberships"]["enhanced_standard_val.txt"]["members"] == [
        {"image": "images/pool/delta.jpg", "label": "labels/pool/delta.txt"},
        {"image": "images/pool/gamma.jpg", "label": "labels/pool/gamma.txt"},
    ]


def test_release_rejects_missing_label_and_path_escape(tmp_path: Path):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    (root / "labels" / "train" / "alpha.txt").unlink()

    with pytest.raises(FileNotFoundError, match="required label"):
        release_builder.build_dataset_release(root, tmp_path / "out.zip", tmp_path / "out", **inputs)

    _write_sample(root, "train", "alpha.jpg", b"alpha-image", b"alpha-label")
    escaped = _write_manifest(tmp_path / "authority" / "escaped.txt", ["../outside.jpg"])
    with pytest.raises(ValueError, match="escapes dataset root"):
        release_builder.build_dataset_release(
            root,
            tmp_path / "out.zip",
            tmp_path / "out",
            **{**inputs, "original_train_manifest": escaped},
        )


def test_release_allows_authorized_rows_from_physical_test_folder_without_publishing_test_paths(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    _write_sample(root, "test", "heldout.jpg", b"test-image", b"test-label")
    authorized_manifest = _write_manifest(
        tmp_path / "authority" / "authorized-train.txt", ["images/test/heldout.jpg"]
    )

    record, archive, _ = _build(
        root,
        tmp_path,
        original_train_manifest=authorized_manifest,
    )

    assert {"image": "images/pool/heldout.jpg", "label": "labels/pool/heldout.txt"} in record[
        "authority_memberships"
    ]["original_grouped_v1_train.txt"]["members"]
    published_json = json.dumps(record)
    assert "images/test" not in published_json
    assert "labels/test" not in published_json
    with ZipFile(archive) as handle:
        assert not any("test" in name.lower() for name in handle.namelist())


def test_release_rejects_test_named_manifest_and_unequal_basename_collisions(tmp_path: Path):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    test_manifest = _write_manifest(tmp_path / "authority" / "test.txt", ["images/train/alpha.jpg"])

    with pytest.raises(PermissionError, match="held-out test authority"):
        release_builder.build_dataset_release(
            root,
            tmp_path / "out.zip",
            tmp_path / "out",
            **{**inputs, "original_train_manifest": test_manifest},
        )

    _write_sample(root, "val", "alpha.jpg", b"different-image", b"alpha-label")
    collision_manifest = _write_manifest(
        tmp_path / "authority" / "collision.txt", ["images/val/alpha.jpg"]
    )
    with pytest.raises(ValueError, match="basename collision"):
        release_builder.build_dataset_release(
            root,
            tmp_path / "out.zip",
            tmp_path / "out",
            **{**inputs, "original_val_manifest": collision_manifest},
        )


def test_release_allows_authorized_hard_links_and_rejects_symlinks(tmp_path: Path):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    hard_link = root / "images" / "val" / "alpha.jpg"
    os.link(root / "images" / "train" / "alpha.jpg", hard_link)
    (root / "labels" / "val" / "alpha.txt").write_bytes(b"alpha-label")
    hard_link_manifest = _write_manifest(tmp_path / "authority" / "hard-link.txt", ["images/val/alpha.jpg"])

    record, archive, _ = _build(root, tmp_path, original_val_manifest=hard_link_manifest)
    assert record["pool"]["image_count"] == 4
    assert record["authority_memberships"]["original_grouped_v1_val.txt"]["members"] == [
        {"image": "images/pool/alpha.jpg", "label": "labels/pool/alpha.txt"}
    ]
    with ZipFile(archive) as handle:
        assert handle.namelist().count("pcb_yolo_dataset/images/pool/alpha.jpg") == 1

    linked_label = root / "labels" / "train" / "symlink.txt"
    linked_label.write_bytes(b"alpha-label")
    linked_image = root / "images" / "train" / "symlink.jpg"
    try:
        linked_image.symlink_to(root / "images" / "train" / "alpha.jpg")
    except OSError as error:
        pytest.skip(f"symlinks are unavailable in this environment: {error}")
    symlink_manifest = _write_manifest(
        tmp_path / "authority" / "symlink.txt", ["images/train/symlink.jpg"]
    )
    with pytest.raises(PermissionError, match="symbolic link"):
        release_builder.build_dataset_release(
            root,
            tmp_path / "out.zip",
            tmp_path / "out",
            **{**inputs, "original_train_manifest": symlink_manifest},
        )


@pytest.mark.parametrize(
    "linked_path",
    [
        "dataset_root",
        "manifest",
        "authority",
        "image_directory",
        "label_directory",
    ],
)
def test_release_rejects_symlink_components_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_path: str
):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    targets = {
        "dataset_root": root,
        "manifest": inputs["original_train_manifest"],
        "authority": inputs["enhanced_standard_val_authority"],
        "image_directory": root / "images" / "train",
        "label_directory": root / "labels" / "train",
    }
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def mark_selected_path_as_symlink(path: Path) -> bool:
        return path == targets[linked_path] or original_is_symlink(path)

    def reject_selected_path_resolution(path: Path, *args, **kwargs) -> Path:
        if path == targets[linked_path]:
            raise AssertionError(f"linked path was resolved: {path}")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", mark_selected_path_as_symlink)
    monkeypatch.setattr(Path, "resolve", reject_selected_path_resolution)

    with pytest.raises(PermissionError, match="symbolic link"):
        release_builder.build_dataset_release(
            root, tmp_path / "out.zip", tmp_path / "out", **inputs
        )


def test_release_rejects_derived_label_that_resolves_outside_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    label = root / "labels" / "train" / "alpha.txt"
    outside = tmp_path / "outside" / "alpha.txt"
    outside.parent.mkdir()
    outside.write_bytes(b"outside")
    original_resolve = Path.resolve

    def resolve_label_outside(path: Path, *args, **kwargs) -> Path:
        if path == label:
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_label_outside)

    with pytest.raises(ValueError, match="label path escapes dataset root"):
        release_builder.build_dataset_release(
            root, tmp_path / "out.zip", tmp_path / "out", **inputs
        )


def test_release_rejects_mutation_before_staging_and_leaves_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    archive = tmp_path / "out.zip"
    original_write = release_builder._write_dataset_zip

    def mutate_then_write(output_zip: Path, pool: dict):
        (root / "images" / "train" / "alpha.jpg").write_bytes(b"mutated")
        original_write(output_zip, pool)

    monkeypatch.setattr(release_builder, "_write_dataset_zip", mutate_then_write)

    with pytest.raises(RuntimeError, match="dataset changed after inventory"):
        release_builder.build_dataset_release(root, archive, tmp_path / "out", **inputs)

    assert not archive.exists()
    assert not list(tmp_path.rglob(".*.tmp"))


@pytest.mark.parametrize("failed_name", ["pcb_yolo_train_val_v1.0.0.manifest.json", "pcb_yolo_train_val_v1.0.0.sha256"])
def test_release_restores_previous_complete_set_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
):
    root = tmp_path / "dataset"
    _, archive, manifest_dir = _build(root, tmp_path)
    inventory = manifest_dir / "pcb_yolo_train_val_v1.0.0.manifest.json"
    checksum = manifest_dir / "pcb_yolo_train_val_v1.0.0.sha256"
    previous = {path: path.read_bytes() for path in (archive, inventory, checksum)}
    (root / "images" / "train" / "alpha.jpg").write_bytes(b"updated-image")
    inputs = _authority_inputs(root, tmp_path)
    original_replace = release_builder.os.replace
    failed_destination = manifest_dir / failed_name

    def fail_selected_promotion(source, destination):
        if Path(destination) == failed_destination and Path(source).suffix == ".tmp":
            raise OSError("injected promotion failure")
        return original_replace(source, destination)

    monkeypatch.setattr(release_builder.os, "replace", fail_selected_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        release_builder.build_dataset_release(root, archive, manifest_dir, **inputs)

    assert {path: path.read_bytes() for path in previous} == previous
    assert not [path for path in tmp_path.rglob(".*") if path.suffix in {".tmp", ".bak"}]


def test_release_preserves_backup_when_rollback_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "dataset"
    _, archive, manifest_dir = _build(root, tmp_path)
    inventory = manifest_dir / "pcb_yolo_train_val_v1.0.0.manifest.json"
    previous_archive = archive.read_bytes()
    (root / "images" / "train" / "alpha.jpg").write_bytes(b"updated-image")
    inputs = _authority_inputs(root, tmp_path)
    original_replace = release_builder.os.replace

    def fail_promotion_and_restore(source, destination):
        source = Path(source)
        destination = Path(destination)
        if destination == inventory and source.suffix == ".tmp":
            raise OSError("injected promotion failure")
        if destination == archive and source.suffix == ".bak":
            raise OSError("injected rollback restore failure")
        return original_replace(source, destination)

    monkeypatch.setattr(release_builder.os, "replace", fail_promotion_and_restore)

    with pytest.raises(RuntimeError, match="recovery backup retained"):
        release_builder.build_dataset_release(root, archive, manifest_dir, **inputs)

    backups = list(archive.parent.glob(f".{archive.name}.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous_archive


def test_release_cleans_registered_temp_when_later_temp_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "dataset"
    inputs = _authority_inputs(root, tmp_path)
    original_temporary_sibling = release_builder._temporary_sibling
    attempts = 0

    def fail_on_second_temp(destination: Path, suffix: str = ".tmp") -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("injected tempfile creation failure")
        return original_temporary_sibling(destination, suffix)

    monkeypatch.setattr(release_builder, "_temporary_sibling", fail_on_second_temp)
    with pytest.raises(OSError, match="injected tempfile creation failure"):
        release_builder.build_dataset_release(root, tmp_path / "out.zip", tmp_path / "out", **inputs)

    assert not list(tmp_path.rglob(".*.tmp"))
