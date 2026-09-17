import json
import subprocess
from pathlib import Path

from scripts.verify.verify_package import verify_repository


def test_verifier_rejects_absolute_user_path_and_test_content(tmp_path: Path):
    (tmp_path / "bad.txt").write_text(
        "C:" + "\\Users\\Admin\\secret\\test\\images", encoding="utf-8"
    )

    failures = verify_repository(tmp_path)

    assert any("absolute path" in failure for failure in failures)
    assert any("held-out test content" in failure for failure in failures)


def test_verifier_reports_broken_relative_markdown_link(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "[missing](docs/missing.md)", encoding="utf-8"
    )

    failures = verify_repository(tmp_path)

    assert any("broken link" in failure for failure in failures)


def test_verifier_rejects_true_test_split_usage(tmp_path: Path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"test_split_used": True}), encoding="utf-8"
    )

    failures = verify_repository(tmp_path)

    assert any("test_split_used must be false" in failure for failure in failures)


def test_verifier_accepts_portable_validation_only_package(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[guide](docs/guide.md)", encoding="utf-8"
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps({"test_split_used": False}), encoding="utf-8"
    )

    assert verify_repository(tmp_path) == []


def test_verifier_rejects_json_escaped_windows_absolute_paths(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"path": "D:" + "\\Users\\Researcher\\input"}), encoding="utf-8")
    assert any("absolute path" in failure for failure in verify_repository(tmp_path))


def test_verifier_rejects_exactly_100_mib(tmp_path):
    with (tmp_path / "large.pt").open("wb") as handle:
        handle.truncate(100 * 1024 * 1024)
    assert any("100 MiB" in failure for failure in verify_repository(tmp_path))


def test_git_verification_checks_only_git_candidates_including_tracked_ignored_files(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/dataset/\n/release-assets/\n/results/local_vscode_comparison_*/\n", encoding="utf-8")
    for name in ("dataset/local.json", "release-assets/local.json", "results/local_vscode_comparison_preflight/authority/local.json", "release-assets/tracked.json", "unignored.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path": "C:" + "\\Users\\Researcher\\data"}), encoding="utf-8")
    subprocess.run(["git", "add", "-f", "release-assets/tracked.json"], cwd=tmp_path, check=True)

    failures = verify_repository(tmp_path)

    assert len(failures) == 2, failures
    assert any("tracked.json" in failure for failure in failures)
    assert any("unignored.json" in failure for failure in failures)
    assert all("local.json" not in failure for failure in failures)


def test_git_verification_does_not_traverse_ignored_directories(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "portable.txt").write_text("portable", encoding="utf-8")

    def forbid_walk(*args, **kwargs):
        raise AssertionError("Git verification must enumerate candidates without rglob")

    monkeypatch.setattr(Path, "rglob", forbid_walk)
    assert verify_repository(tmp_path) == []
