"""Verify that the repository is portable, private-safe, and validation-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote


MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", ".venv", "release-assets"}
ABSOLUTE_USER_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+")
HELD_OUT_CONTENT = re.compile(
    r"(?i)(?:[\\/](?:images|labels)[\\/]test(?:[\\/]|$)|"
    r"[\\/]test[\\/](?:images|labels)(?:[\\/]|$))"
)
MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _candidate_files(root: Path) -> Iterable[Path]:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root, capture_output=True, check=False,
        )
    except FileNotFoundError:
        probe = None
    if probe is not None and probe.returncode == 0 and probe.stdout.strip() == b"true":
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, check=True,
        )
        for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
            path = root / name
            if path.is_file() or path.is_symlink():
                yield path
        return
    # Extracted clone simulations and isolated fixtures have no Git metadata.
    for path in sorted(root.rglob("*")):
        if not (path.is_file() or path.is_symlink()) or any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        yield path


def _has_held_out_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "test" in parts and bool(parts.intersection({"images", "labels"}))


def _json_test_split_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "test_split_used":
                yield item
            yield from _json_test_split_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_test_split_values(item)


def _verify_markdown_links(path: Path, text: str, root: Path) -> list[str]:
    failures = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative = unquote(target.split("#", 1)[0])
        if relative and not (path.parent / relative).resolve().is_relative_to(root.resolve()):
            failures.append(f"link escapes repository: {path}: {target}")
        elif relative and not (path.parent / relative).exists():
            failures.append(f"broken link: {path}: {target}")
    return failures


def verify_repository(root: Path) -> list[str]:
    root = Path(root).resolve()
    failures: list[str] = []
    for path in _candidate_files(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            failures.append(f"symbolic link is not allowed: {relative}")
            continue
        if path.stat().st_size >= MAX_GIT_FILE_BYTES:
            failures.append(f"file exceeds 100 MiB Git limit: {relative}")
        if _has_held_out_path(relative):
            failures.append(f"held-out test content path is forbidden: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"declared text file is not UTF-8: {relative}")
            continue
        if ABSOLUTE_USER_PATH.search(text):
            failures.append(f"absolute path found: {relative}")
        if HELD_OUT_CONTENT.search(text):
            failures.append(f"held-out test content reference found: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"credential-like value found: {relative}")
                break
        if path.suffix.lower() == ".md":
            failures.extend(_verify_markdown_links(path, text, root))
        if path.suffix.lower() in {".json", ".ipynb"}:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                failures.append(f"invalid JSON: {relative}: {error}")
            else:
                if any(value is not False for value in _json_test_split_values(payload)):
                    failures.append(f"test_split_used must be false: {relative}")
    package = root / 'custom_yolo_pcb'
    manifest = package / 'reproducibility/manifests/hashes/artifact_provenance.json'
    if manifest.is_file():
        for record in json.loads(manifest.read_text(encoding='utf-8'))['artifacts']:
            target = package / record['destination']
            if not target.resolve().is_relative_to(package.resolve()):
                failures.append(f"artifact escapes package: {record['destination']}")
            elif not target.is_file():
                failures.append(f"missing attested artifact: {record['destination']}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != record['published_sha256']:
                failures.append(f"artifact hash mismatch: {record['destination']}")
    return sorted(set(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    failures = verify_repository(parse_args().root)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("Package verification passed with zero failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
