import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.package.package_comparison import (
    copy_artifact,
    normalize_text,
    require_completed_run,
)


def test_exported_runtime_has_all_launcher_targets_and_training_help(tmp_path):
    from scripts.package import package_comparison as exporter

    package = Path(__file__).resolve().parents[2]
    exported = tmp_path / 'export with spaces' / 'custom_yolo_pcb'
    exporter._copy_items(package, exported, exporter.RUNTIME_ITEMS)

    required = (
        'train_local.py', 'prepare_dataset.ps1', 'requirements.txt',
        'model_code/tools/run_local_vscode_comparison.py',
        'reproducibility/scripts/package/extract_dataset.py',
        'reproducibility/manifests/dataset/pcb_yolo_train_val_v1.0.0.sha256',
    )
    for target in required:
        assert (exported / target).is_file(), target
    # This wrapper requires a complete Git checkout, not an evidence-only export.
    assert not (exported / 'model_code/tools/verify_package.ps1').exists()
    wrappers = list((exported / 'model_code/tools').glob('*.ps1'))
    assert {path.name for path in wrappers} == {'prepare_dataset.ps1', 'run_local_vscode_comparison.ps1'}
    for wrapper in wrappers:
        text = wrapper.read_text(encoding='utf-8')
        for target in re.findall(r"Join-Path \$package '([^']+)'", text):
            assert (exported / target).is_file(), (wrapper.name, target)
    preparation = (exported / 'prepare_dataset.ps1').read_text(encoding='utf-8')
    for target in re.findall(r'Join-Path \$PSScriptRoot "(reproducibility[^"\n]+)"', preparation):
        assert (exported / target.replace('\\', '/')).is_file(), target
    result = subprocess.run([sys.executable, str(exported / 'train_local.py'), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--preflight-only' in result.stdout


def test_normalizer_removes_absolute_root_and_hashes_source(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "record.txt"
    source.write_text(f"root={source_root}\\data", encoding="utf-8")

    text, record = normalize_text(source, source_root)

    assert str(source_root) not in text
    assert text == "root=__SOURCE_ROOT__\\data"
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["normalized"] is True


@pytest.mark.parametrize("ending", [b"\r\n", b"\r\r\n", b"\r"])
def test_normalizer_converts_windows_line_endings_and_records_change(tmp_path, ending):
    source = tmp_path / "record.yaml"
    raw = ending.join([b"train: train.txt", b"val: val.txt", b""])
    source.write_bytes(raw)
    text, record = normalize_text(source, tmp_path)
    assert text == "train: train.txt\nval: val.txt\n"
    assert record["normalized"] is True
    assert record["source_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r\r\n"])
def test_copied_text_is_exact_lf_bytes_without_windows_write_translation(tmp_path, ending):
    source = tmp_path / "record.csv"
    source.write_bytes(ending.join([b"epoch,time", b"1,1.5", b""]))
    destination_root = tmp_path / "publication"
    target = destination_root / "record.csv"
    record = copy_artifact(source, target, source_root=tmp_path, destination_root=destination_root)
    expected = b"epoch,time\n1,1.5\n"
    assert target.read_bytes() == expected
    assert record["published_sha256"] == hashlib.sha256(expected).hexdigest()
    assert record["normalized"] is (ending != b"\n")


@pytest.mark.parametrize("escaped", [False, True])
def test_normalizer_sanitizes_windows_paths_including_json(tmp_path, escaped):
    root = "D:" + "\\Users\\Researcher\\project"
    value = root + "\\dataset\\image.jpg"
    source = tmp_path / ("record.json" if escaped else "record.txt")
    source.write_text(json.dumps({"path": value}) if escaped else value, encoding="utf-8")
    text, record = normalize_text(source, Path(root))
    decoded = json.loads(text)["path"] if escaped else text
    assert decoded == "__SOURCE_ROOT__\\dataset\\image.jpg"
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_normalizer_sanitizes_other_windows_drive_and_unc_paths(tmp_path):
    source = tmp_path / "record.json"
    source.write_text(json.dumps({"drive": "E:" + "\\outside\\model.pt", "unc": "\\\\server\\share\\model.pt"}), encoding="utf-8")
    text, _ = normalize_text(source, tmp_path)
    payload = json.loads(text)
    assert payload["drive"] == "__ABSOLUTE_ROOT__/outside/model.pt"
    assert payload["unc"] == "__NETWORK_ROOT__/server/share/model.pt"


def test_completion_gate_rejects_running_state(tmp_path: Path):
    state = tmp_path / "run_state.json"
    state.write_text(
        json.dumps({"status": "RUNNING", "test_split_used": False}),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="COMPLETED"):
        require_completed_run(state)


def test_completion_gate_accepts_validation_only_completed_state(tmp_path: Path):
    state = tmp_path / "run_state.json"
    payload = {
        "status": "COMPLETED",
        "current_stage": "completed",
        "original_completed": True,
        "enhanced_completed": True,
        "test_split_used": False,
    }
    state.write_text(json.dumps(payload), encoding="utf-8")

    assert require_completed_run(state) == payload


def test_copy_artifact_rejects_destination_escape(tmp_path: Path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "destination"
    source_root.mkdir()
    destination_root.mkdir()
    source = source_root / "record.txt"
    source.write_text("safe", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside destination root"):
        copy_artifact(
            source,
            destination_root.parent / "escape.txt",
            source_root=source_root,
            destination_root=destination_root,
        )


def test_copy_artifact_rejects_held_out_test_content(tmp_path: Path):
    source_root = tmp_path / "source"
    source = source_root / "images" / "test" / "image.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")

    with pytest.raises(PermissionError, match="held-out test content"):
        copy_artifact(
            source,
            tmp_path / "destination" / "image.jpg",
            source_root=source_root,
            destination_root=tmp_path / "destination",
        )


def test_portable_authority_export_keeps_order_duplicates_and_exact_validation(tmp_path):
    from scripts.package import package_comparison as exporter
    source = tmp_path / "research"
    run = tmp_path / "run"
    destination = tmp_path / "publication"
    (run / "authority").mkdir(parents=True)
    names = ("original_grouped_v1_train.txt", "original_grouped_v1_val.txt", "enhanced_trial044_ohem_train.txt")
    for name in names:
        (run / "authority" / name).write_text("pcb_yolo_dataset/images/train/z.jpg\npcb_yolo_dataset/images/val/a.jpg\npcb_yolo_dataset/images/train/z.jpg\n", encoding="utf-8")
    val = source / "pcb_yolo_dataset/images/val"
    val.mkdir(parents=True)
    (val / "z.jpg").touch()
    (val / "a.jpg").touch()
    records = exporter.export_portable_authorities(source, run, destination)
    published = destination / "reproducibility/manifests/dataset/authority"
    assert (published / names[2]).read_text().splitlines() == ["pcb_yolo_dataset/images/pool/z.jpg", "pcb_yolo_dataset/images/pool/a.jpg", "pcb_yolo_dataset/images/pool/z.jpg"]
    assert (published / "enhanced_standard_val.txt").read_text().splitlines() == ["pcb_yolo_dataset/images/pool/a.jpg", "pcb_yolo_dataset/images/pool/z.jpg"]
    assert len(records) == 4
    assert records[0]["source_sha256"] == hashlib.sha256((run / "authority" / names[0]).read_bytes()).hexdigest()
