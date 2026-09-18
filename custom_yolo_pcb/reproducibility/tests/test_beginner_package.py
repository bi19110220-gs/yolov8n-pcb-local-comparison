"""Acceptance contract for the simplified portable FYP package."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())
PACKAGE = ROOT / 'custom_yolo_pcb'


def test_only_four_tracked_root_entries():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    names = [n for n in names if (ROOT / n).is_file()]
    assert {n.split('/')[0] for n in names} == {'.gitignore', '.gitattributes', 'README.md', 'custom_yolo_pcb'}


def test_beginner_files_and_four_visible_weights():
    for name in ('README.md', 'RESULTS_REPORT.md', 'train_local.py', 'prepare_dataset.ps1', 'requirements.txt'):
        assert (PACKAGE / name).is_file(), name
    assert {p.name for p in (PACKAGE / 'weights').glob('*.pt')} == {'original_best.pt', 'trial044_best.pt', 'yolov8n.pt', 'trial035_parent_best.pt'}
    assert {p.name for p in (PACKAGE / 'reproducibility/checkpoints').glob('*.pt')} == {'original_last.pt', 'trial044_last.pt'}


def test_no_duplicate_checkpoints_or_obsolete_presentation():
    candidates = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    weights = [ROOT / name for name in candidates if name.endswith('.pt') and (ROOT / name).is_file()]
    assert len(weights) == 6
    assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}) == 6
    for name in ('.github', '.vscode', 'docs', 'PRIVATE_USE_NOTICE.md', 'DATASET_PROVENANCE.md'):
        assert not (ROOT / name).exists()
    assert not list(PACKAGE.rglob('methodology.md'))
    assert not any('results' in path.relative_to(PACKAGE).parts for path in weights)


def test_training_entry_help_works_from_unrelated_directory(tmp_path):
    assert (PACKAGE / 'train_local.py').is_file()
    result = subprocess.run([sys.executable, str(PACKAGE / 'train_local.py'), '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--preflight-only' in result.stdout
    assert '--resume-enhanced-only' in result.stdout


def test_results_have_balanced_charts_and_validation_provenance():
    for model in ('original', 'trial044'):
        assert (PACKAGE / 'results/charts' / f'{model}_training.png').is_file()
        target = PACKAGE / 'results/regenerated_validation' / model
        for name in ('confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 'provenance.json'):
            assert (target / name).is_file(), (model, name)
    for name in ('final_comparison.png', 'per_class.png', 'source_hashes.json'):
        assert (PACKAGE / 'results/charts' / name).is_file()
