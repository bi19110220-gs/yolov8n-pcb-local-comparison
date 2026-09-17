"""Verify the Release checksum before extracting a canonical dataset pool."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path


def extract_dataset(archive: Path, checksum: Path, destination: Path) -> None:
    archive, checksum, destination = Path(archive), Path(checksum), Path(destination)
    match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?([^\r\n]+)\s*", checksum.read_text(encoding="utf-8").strip())
    if not match or match[2] != archive.name:
        raise PermissionError("checksum must name this exact Release archive")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != match[1].lower():
        raise PermissionError("Release archive SHA-256 mismatch; nothing extracted")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("dataset destination must be absent or empty")
    if destination.is_symlink():
        raise PermissionError("dataset destination must not be a filesystem alias")
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        seen = set()
        for member in members:
            name = member.filename
            if (not re.fullmatch(r"pcb_yolo_dataset/(?:images|labels)/pool/[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
                or name.lower() in seen or stat.S_ISLNK(member.external_attr >> 16)):
                raise PermissionError(f"unapproved or duplicate Release member: {name}")
            seen.add(name.lower())
        if not members:
            raise PermissionError("Release archive is empty")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pcb-dataset-stage-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            for member in members:
                target = stage / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
            destination.mkdir(exist_ok=True)
            os.replace(stage / "pcb_yolo_dataset", destination / "pcb_yolo_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    extract_dataset(args.archive, args.checksum, args.destination)
    print("Release SHA-256 verified; dataset pool extracted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
