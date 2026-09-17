"""Build a deterministic authority-aware train/validation PCB dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
AUTHORITY_NAMES = (
    "original_grouped_v1_train.txt",
    "original_grouped_v1_val.txt",
    "enhanced_trial044_ohem_train.txt",
    "enhanced_standard_val.txt",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_filesystem_alias(path: Path) -> None:
    if path.is_symlink():
        raise PermissionError(f"symbolic link is forbidden in dataset release: {path}")


def _reject_symlink_components(path: Path, stop_at: Path | None = None) -> None:
    current = Path(path)
    boundary = Path(stop_at) if stop_at is not None else None
    while True:
        _reject_filesystem_alias(current)
        if boundary is not None and current == boundary:
            return
        parent = current.parent
        if parent == current:
            return
        current = parent


def _read_attested(path: Path, size_bytes: int, sha256: str) -> bytes:
    _reject_symlink_components(path)
    payload = path.read_bytes()
    if len(payload) != size_bytes or sha256_bytes(payload) != sha256:
        raise RuntimeError(f"dataset changed after inventory: {path}")
    return payload


def _read_manifest_rows(manifest: Path, authority_name: str) -> tuple[list[str], str]:
    manifest = Path(manifest)
    _reject_symlink_components(manifest)
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"required authority manifest is missing: {manifest}")
    if "test" in manifest.name.lower():
        raise PermissionError(f"held-out test authority is forbidden: {manifest}")
    raw = manifest.read_bytes()
    rows = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"authority manifest is empty: {authority_name}")
    return rows, sha256_bytes(raw)


def _resolve_image(dataset_root: Path, row: str) -> tuple[Path, Path]:
    candidate = Path(row)
    source = candidate if candidate.is_absolute() else dataset_root / candidate
    _reject_symlink_components(source, stop_at=dataset_root)
    try:
        resolved = source.resolve(strict=False)
        relative = resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"authority path escapes dataset root: {row}") from error
    if not relative.parts or relative.parts[0].lower() != "images":
        raise ValueError(f"authority row must reference an images path: {row}")
    if not resolved.is_file():
        raise FileNotFoundError(f"required image is missing: {resolved}")
    label_relative = Path("labels", *relative.parts[1:]).with_suffix(".txt")
    label_source = dataset_root / label_relative
    _reject_symlink_components(label_source, stop_at=dataset_root)
    label = label_source.resolve(strict=False)
    try:
        label.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"label path escapes dataset root: {label_source}") from error
    if not label.is_file():
        raise FileNotFoundError(f"required label is missing: {label}")
    return resolved, label


def _sample_from_row(dataset_root: Path, row: str) -> dict[str, Any]:
    image, label = _resolve_image(dataset_root, row)
    image_payload = image.read_bytes()
    label_payload = label.read_bytes()
    return {
        "image_source": image,
        "label_source": label,
        "image_path": f"images/pool/{image.name}",
        "label_path": f"labels/pool/{image.stem}.txt",
        "image_size_bytes": len(image_payload),
        "label_size_bytes": len(label_payload),
        "image_sha256": sha256_bytes(image_payload),
        "label_sha256": sha256_bytes(label_payload),
    }


def _rows_from_standard_val_authority(
    dataset_root: Path, authority: Path
) -> tuple[list[str], str, str]:
    authority = Path(authority)
    _reject_symlink_components(authority)
    authority = authority.resolve()
    if authority.is_dir():
        try:
            relative = authority.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(f"authority path escapes dataset root: {authority}") from error
        if not relative.parts or relative.parts[0].lower() != "images":
            raise ValueError(f"standard validation directory must be beneath images: {authority}")
        rows = [
            path.relative_to(dataset_root).as_posix()
            for path in sorted(authority.rglob("*"))
            if path.is_file()
        ]
        for row in rows:
            _resolve_image(dataset_root, row)
        if not rows:
            raise ValueError(f"standard validation directory is empty: {authority}")
        return rows, sha256_bytes(("\n".join(rows) + "\n").encode("utf-8")), "directory"
    rows, source_sha256 = _read_manifest_rows(authority, AUTHORITY_NAMES[3])
    return sorted(rows), source_sha256, "manifest"


def _add_to_pool(pool: dict[str, Any], sample: dict[str, Any]) -> None:
    existing = pool["samples"].get(sample["image_path"])
    if existing is not None:
        if (
            existing["image_sha256"] != sample["image_sha256"]
            or existing["label_sha256"] != sample["label_sha256"]
        ):
            raise ValueError(f"basename collision has unequal content or labels: {sample['image_path']}")
        return
    existing_label = pool["labels"].get(sample["label_path"])
    if existing_label is not None and existing_label["label_sha256"] != sample["label_sha256"]:
        raise ValueError(f"basename collision has unequal content or labels: {sample['label_path']}")
    pool["samples"][sample["image_path"]] = sample
    pool["labels"][sample["label_path"]] = sample


def _membership(
    dataset_root: Path,
    pool: dict[str, Any],
    authority_name: str,
    rows: list[str],
    source_sha256: str,
    source_type: str,
) -> dict[str, Any]:
    members = []
    for row in rows:
        sample = _sample_from_row(dataset_root, row)
        _add_to_pool(pool, sample)
        members.append({"image": sample["image_path"], "label": sample["label_path"]})
    return {
        "source_type": source_type,
        "source_sha256": source_sha256,
        "published_sha256": sha256_bytes(
            (json.dumps(members, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        ),
        "source_row_count": len(rows),
        "published_row_count": len(members),
        "published_unique_member_count": len({(item["image"], item["label"]) for item in members}),
        "members": members,
    }


def inventory_authority_pool(
    dataset_root: Path,
    *,
    original_train_manifest: Path,
    original_val_manifest: Path,
    enhanced_ohem_train_manifest: Path,
    enhanced_standard_val_authority: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a publishable record and private validated sources for ZIP writing."""
    dataset_root = Path(dataset_root)
    _reject_symlink_components(dataset_root)
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root is missing: {dataset_root}")
    pool: dict[str, Any] = {"samples": {}, "labels": {}}
    authority_memberships: dict[str, Any] = {}
    manifest_inputs = (
        (AUTHORITY_NAMES[0], original_train_manifest),
        (AUTHORITY_NAMES[1], original_val_manifest),
        (AUTHORITY_NAMES[2], enhanced_ohem_train_manifest),
    )
    for authority_name, source_manifest in manifest_inputs:
        rows, source_sha256 = _read_manifest_rows(source_manifest, authority_name)
        authority_memberships[authority_name] = _membership(
            dataset_root, pool, authority_name, rows, source_sha256, "manifest"
        )
    rows, source_sha256, source_type = _rows_from_standard_val_authority(
        dataset_root, enhanced_standard_val_authority
    )
    authority_memberships[AUTHORITY_NAMES[3]] = _membership(
        dataset_root, pool, AUTHORITY_NAMES[3], rows, source_sha256, source_type
    )
    images = []
    labels = []
    for image_path, sample in sorted(pool["samples"].items()):
        images.append(
            {
                "path": image_path,
                "size_bytes": sample["image_size_bytes"],
                "source_sha256": sample["image_sha256"],
                "published_sha256": sample["image_sha256"],
            }
        )
    for label_path, sample in sorted(pool["labels"].items()):
        labels.append(
            {
                "path": label_path,
                "size_bytes": sample["label_size_bytes"],
                "source_sha256": sample["label_sha256"],
                "published_sha256": sample["label_sha256"],
            }
        )
    record = {
        "schema": "pcb_yolo_authority_pool_dataset/v1",
        "test_split_included": False,
        "authority_memberships": authority_memberships,
        "pool": {
            "sample_count": len(pool["samples"]),
            "image_count": len(images),
            "label_count": len(labels),
            "images": images,
            "labels": labels,
        },
    }
    return record, pool


def _write_dataset_zip(output_zip: Path, pool: dict[str, Any]) -> None:
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for image_path, sample in sorted(pool["samples"].items()):
            payload = _read_attested(
                sample["image_source"], sample["image_size_bytes"], sample["image_sha256"]
            )
            info = ZipInfo(f"pcb_yolo_dataset/{image_path}", FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=6)
        for label_path, sample in sorted(pool["labels"].items()):
            payload = _read_attested(
                sample["label_source"], sample["label_size_bytes"], sample["label_sha256"]
            )
            info = ZipInfo(f"pcb_yolo_dataset/{label_path}", FIXED_ZIP_TIME)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=6)


def _temporary_sibling(destination: Path, suffix: str = ".tmp") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        return Path(handle.name)


def build_dataset_release(
    dataset_root: Path,
    output_zip: Path,
    manifest_dir: Path,
    *,
    original_train_manifest: Path,
    original_val_manifest: Path,
    enhanced_ohem_train_manifest: Path,
    enhanced_standard_val_authority: Path,
) -> dict[str, Any]:
    """Atomically publish a neutral pool ZIP and authority-membership sidecars."""
    output_zip = Path(output_zip).resolve()
    manifest_dir = Path(manifest_dir).resolve()
    record, pool = inventory_authority_pool(
        dataset_root,
        original_train_manifest=original_train_manifest,
        original_val_manifest=original_val_manifest,
        enhanced_ohem_train_manifest=enhanced_ohem_train_manifest,
        enhanced_standard_val_authority=enhanced_standard_val_authority,
    )
    release_name = output_zip.stem
    inventory_path = manifest_dir / f"{release_name}.manifest.json"
    checksum_path = manifest_dir / f"{release_name}.sha256"
    temporary_paths: list[Path] = []
    backup_paths: dict[Path, Path] = {}
    published_destinations: list[Path] = []
    preserve_backups = False
    try:
        temporary_archive = _temporary_sibling(output_zip)
        temporary_paths.append(temporary_archive)
        temporary_inventory = _temporary_sibling(inventory_path)
        temporary_paths.append(temporary_inventory)
        temporary_checksum = _temporary_sibling(checksum_path)
        temporary_paths.append(temporary_checksum)
        _write_dataset_zip(temporary_archive, pool)
        record.update(
            archive=output_zip.name,
            archive_size_bytes=temporary_archive.stat().st_size,
            archive_sha256=sha256_file(temporary_archive),
        )
        temporary_inventory.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_checksum.write_text(
            f"{record['archive_sha256']} *{output_zip.name}\n", encoding="utf-8"
        )
        publications = (
            (temporary_archive, output_zip),
            (temporary_inventory, inventory_path),
            (temporary_checksum, checksum_path),
        )
        for _, destination in publications:
            if destination.exists():
                backup = _temporary_sibling(destination, suffix=".bak")
                backup_paths[destination] = backup
                copyfile(destination, backup)
        try:
            for temporary_path, destination in publications:
                os.replace(temporary_path, destination)
                temporary_paths.remove(temporary_path)
                published_destinations.append(destination)
        except BaseException as promotion_error:
            recovery_paths: list[Path] = []
            for destination in reversed(published_destinations):
                backup = backup_paths.get(destination)
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    try:
                        os.replace(backup, destination)
                    except OSError:
                        recovery_paths.append(backup)
                    else:
                        backup_paths.pop(destination)
            if recovery_paths:
                preserve_backups = True
                paths = ", ".join(str(path) for path in recovery_paths)
                raise RuntimeError(
                    f"release rollback failed; recovery backup retained: {paths}"
                ) from promotion_error
            raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        if not preserve_backups:
            for backup_path in backup_paths.values():
                backup_path.unlink(missing_ok=True)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-train-manifest", type=Path, required=True)
    parser.add_argument("--original-val-manifest", type=Path, required=True)
    parser.add_argument("--enhanced-ohem-train-manifest", type=Path, required=True)
    parser.add_argument("--enhanced-standard-val-authority", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests/dataset"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    return args


def main() -> int:
    args = parse_args()
    inputs = {
        "original_train_manifest": args.original_train_manifest,
        "original_val_manifest": args.original_val_manifest,
        "enhanced_ohem_train_manifest": args.enhanced_ohem_train_manifest,
        "enhanced_standard_val_authority": args.enhanced_standard_val_authority,
    }
    record = (
        inventory_authority_pool(args.dataset_root, **inputs)[0]
        if args.dry_run
        else build_dataset_release(args.dataset_root, args.output, args.manifest_dir, **inputs)
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
