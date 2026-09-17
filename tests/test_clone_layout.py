import subprocess
import sys
from pathlib import Path

from scripts.verify import verify_clone


ROOT = Path(__file__).resolve().parents[1]


def test_clone_layout_comes_from_git_files_not_empty_worktree_directories(tmp_path):
    clone = tmp_path / "fresh clone with spaces"
    verify_clone.copy_git_candidates(ROOT, clone)
    assert not verify_clone.verify_layout(clone)
    assert not (clone / "dataset").exists()
    assert not (clone / "src").exists()
    result = subprocess.run([sys.executable, "-m", "tools.run_local_vscode_comparison", "--help"], cwd=clone, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--preflight-only" in result.stdout


def test_layout_rejects_empty_model_and_config_directories(tmp_path):
    for name in ("models/original", "models/trial035_parent", "configs/original"):
        (tmp_path / name).mkdir(parents=True)
    failures = verify_clone.verify_layout(tmp_path)
    assert any("models/original/best.pt" in failure for failure in failures)
    assert any("configs/original/recorded_train_args.json" in failure for failure in failures)


def test_git_attributes_preserve_lf_artifact_bytes_with_windows_autocrlf(tmp_path):
    import json
    import shutil
    from scripts.package.package_comparison import TEXT_SUFFIXES, sha256_file
    shutil.copy2(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=tmp_path, check=True)
    records = json.loads((ROOT / "manifests/hashes/artifact_provenance.json").read_text())["artifacts"]
    for record in records:
        source = ROOT / record["destination"]
        if source.suffix not in TEXT_SUFFIXES:
            continue
        assert b"\r" not in source.read_bytes(), record["destination"]
        assert sha256_file(source) == record["published_sha256"]
        output = subprocess.check_output(["git", "check-attr", "eol", "--", record["destination"]], cwd=tmp_path, text=True)
        assert output.strip().endswith(": eol: lf"), output
    output = subprocess.check_output(["git", "check-attr", "text", "--", "models/original/best.pt"], cwd=tmp_path, text=True)
    assert output.strip().endswith(": text: unset")
