import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from tools import run_local_vscode_comparison as runner


def test_device_selection_has_cpu_fallback():
    assert hasattr(runner, 'select_device')
    with patch('torch.cuda.is_available', return_value=False):
        assert runner.select_device() == 'cpu'
        assert runner.original_train_args(Path('a'), Path('b'))['device'] == 'cpu'
        assert runner.runtime_environment()['cuda_available'] is False
    with patch('torch.cuda.is_available', return_value=True):
        assert runner.select_device() == 0


def test_chart_source_records_actual_epochs():
    path = Path(__file__).resolve().parents[1] / 'generate_charts.py'
    assert path.is_file()
    spec = importlib.util.spec_from_file_location('charts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = module.load_evidence()
    assert len(evidence['original']['rows']) == 100
    assert len(evidence['trial044']['rows']) == 27
    assert evidence['trial044']['best_epoch'] == 12


def test_validation_plan_pins_val_only_and_both_resolutions():
    path = Path(__file__).resolve().parents[1] / 'regenerate_validation.py'
    assert path.is_file()
    spec = importlib.util.spec_from_file_location('regenerate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODELS['original']['imgsz'] == 640
    assert module.MODELS['trial044']['imgsz'] == 1024
    assert module.VALIDATION_SETTINGS == dict(split='val', conf=0.001, iou=0.7, max_det=300, augment=False, plots=True, workers=0)


def test_packager_uses_beginner_destinations():
    from scripts.package import package_comparison as packager
    destinations = [destination for _, destination in packager.SOURCE_ITEMS + packager.RUN_ITEMS]
    assert 'weights/original_best.pt' in destinations
    assert 'weights/trial044_best.pt' in destinations
    assert 'reproducibility/checkpoints/original_last.pt' in destinations
    assert all(not path.startswith(('models/', 'configs/', 'manifests/')) for path in destinations)


def test_verifier_checks_artifact_hashes(tmp_path):
    from scripts.verify import verify_package
    package = tmp_path / 'custom_yolo_pcb'
    manifest = package / 'reproducibility/manifests/hashes/artifact_provenance.json'
    manifest.parent.mkdir(parents=True)
    checkpoint = package / 'weights/model.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'tampered')
    manifest.write_text(json.dumps({'artifacts': [{'destination': 'weights/model.pt', 'published_sha256': '0' * 64}]}))
    assert any('hash mismatch' in failure for failure in verify_package.verify_repository(tmp_path))
