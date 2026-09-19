"""Write the canonical notebook-generated Streamlit application safely."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from app_template import APP_SOURCE, GENERATED_MARKER


class GeneratedAppConflict(PermissionError):
    """Raised when app.py is unrelated handwritten content."""


@dataclass(frozen=True)
class GeneratedApp:
    sha256: str
    replaced: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_generated_app(path: Path) -> GeneratedApp:
    path = Path(path)
    desired = APP_SOURCE.encode("utf-8")
    if path.exists():
        current = path.read_bytes()
        if current == desired:
            return GeneratedApp(sha256=_sha256(desired), replaced=False)
        if not current.decode("utf-8", errors="ignore").startswith(GENERATED_MARKER):
            raise GeneratedAppConflict(
                "Refusing to replace app.py because it is not the known generated file."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(desired)
    return GeneratedApp(sha256=_sha256(desired), replaced=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_generated_app(args.output)
    action = "updated" if result.replaced else "already current"
    print(f"app.py {action}; SHA-256 {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
