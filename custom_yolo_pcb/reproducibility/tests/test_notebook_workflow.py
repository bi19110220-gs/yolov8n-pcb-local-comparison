import sys
from datetime import datetime
from pathlib import Path

import pytest

import notebook_workflow as workflow


def test_project_kernel_guard_requires_package_venv(tmp_path):
    package = tmp_path / "custom_yolo_pcb"
    expected = package / ".venv" / "Scripts" / "python.exe"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"")
    workflow.verify_project_kernel(package, expected)
    with pytest.raises(workflow.ProjectKernelError, match="Select.*Python kernel"):
        workflow.verify_project_kernel(package, tmp_path / "python.exe")


def test_new_notebook_run_dir_is_fresh_and_timestamped(tmp_path):
    package = tmp_path / "custom_yolo_pcb"
    output = workflow.new_notebook_run_dir(
        package, datetime(2026, 9, 19, 12, 34, 56)
    )
    assert output == package / "results" / "notebook_runs" / "20260919_123456"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        workflow.new_notebook_run_dir(
            package, datetime(2026, 9, 19, 12, 34, 56)
        )


def test_complete_comparison_delegates_to_guarded_entry_point(tmp_path, monkeypatch):
    package = tmp_path / "custom_yolo_pcb"
    package.mkdir()
    (package / "train_local.py").write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, cwd, check, stdout, stderr):
        calls.append((command, cwd, check, stdout, stderr))
        return 0

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    output = workflow.run_complete_comparison(
        package, datetime(2026, 9, 19, 12, 34, 56)
    )
    assert output == package / "results" / "notebook_runs" / "20260919_123456"
    command, cwd, check, stdout, stderr = calls[0]
    assert command == [
        sys.executable,
        str(package / "train_local.py"),
        "--output-dir",
        str(output),
    ]
    assert cwd == package
    assert check is True
    assert stdout is not None
    assert stderr is workflow.subprocess.STDOUT
    assert (output.parent / "20260919_123456.training.log").is_file()


def test_recovery_requires_existing_explicit_directory(tmp_path, monkeypatch):
    package = tmp_path / "custom_yolo_pcb"
    package.mkdir()
    (package / "train_local.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="RECOVERY_OUTPUT_DIR"):
        workflow.run_enhanced_recovery(package, None)
    output = package / "results" / "notebook_runs" / "failed"
    output.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda command, cwd, check, stdout, stderr: calls.append(
            (command, cwd, check, stdout, stderr)
        ),
    )
    workflow.run_enhanced_recovery(package, output)
    assert calls[0][0][-1] == "--resume-enhanced-only"
    assert calls[0][3] is not None
    assert calls[0][4] is workflow.subprocess.STDOUT
    assert (output.parent / "failed.recovery.log").is_file()
