import hashlib
import zipfile
from pathlib import Path

import pytest


def make_archive(tmp_path, name="pcb_yolo_dataset/images/pool/a.jpg"):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(name, b"image")
        handle.writestr("pcb_yolo_dataset/labels/pool/a.txt", b"label")
    checksum = tmp_path / "dataset.sha256"
    checksum.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  dataset.zip\n")
    return archive, checksum


def test_extract_verifies_checksum_and_produces_exact_pool_layout(tmp_path):
    from scripts.package.extract_dataset import extract_dataset
    archive, checksum = make_archive(tmp_path)
    destination = tmp_path / "repo with spaces" / "dataset"
    extract_dataset(archive, checksum, destination)
    assert (destination / "pcb_yolo_dataset/images/pool/a.jpg").read_bytes() == b"image"
    assert (destination / "pcb_yolo_dataset/labels/pool/a.txt").read_bytes() == b"label"


def test_checksum_mismatch_leaves_destination_absent(tmp_path):
    from scripts.package.extract_dataset import extract_dataset
    archive, checksum = make_archive(tmp_path)
    checksum.write_text("0" * 64 + "  dataset.zip\n")
    destination = tmp_path / "dataset"
    with pytest.raises(PermissionError, match="SHA-256"):
        extract_dataset(archive, checksum, destination)
    assert not destination.exists()


@pytest.mark.parametrize("member", ["../escape.jpg", "pcb_yolo_dataset/images/" + "test/a.jpg", "pcb_yolo_dataset/images/pool/a.jpg:stream"])
def test_extract_rejects_unapproved_member_paths(tmp_path, member):
    from scripts.package.extract_dataset import extract_dataset
    archive, checksum = make_archive(tmp_path, member)
    with pytest.raises(PermissionError):
        extract_dataset(archive, checksum, tmp_path / "dataset")


def test_extract_never_overwrites_existing_dataset(tmp_path):
    from scripts.package.extract_dataset import extract_dataset
    archive, checksum = make_archive(tmp_path)
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "preserve").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        extract_dataset(archive, checksum, destination)
    assert (destination / "preserve").read_bytes() == b"keep"
