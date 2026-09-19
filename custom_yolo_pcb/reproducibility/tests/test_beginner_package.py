"""Acceptance contract for the simplified portable FYP package."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(
    subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'],
        cwd=Path(__file__).resolve().parent,
        text=True,
    ).strip()
)
PACKAGE = ROOT / 'custom_yolo_pcb'


def test_only_four_tracked_root_entries():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    names = [n for n in names if (ROOT / n).is_file()]
    assert {n.split('/')[0] for n in names} == {'.gitignore', '.gitattributes', 'README.md', 'custom_yolo_pcb'}


def test_beginner_files_and_four_visible_weights():
    for name in (
        'PCB_Quality_Inspector.ipynb', 'app.py', 'README.md', 'RESULTS_REPORT.md',
        'train_local.py', 'prepare_dataset.ps1', 'requirements.txt',
    ):
        assert (PACKAGE / name).is_file(), name
    assert {p.name for p in (PACKAGE / 'weights').glob('*.pt')} == {'original_best.pt', 'trial044_best.pt', 'yolov8n.pt', 'trial035_parent_best.pt'}
    assert {p.name for p in (PACKAGE / 'reproducibility/checkpoints').glob('*.pt')} == {'original_last.pt', 'trial044_last.pt'}


def test_notebook_and_streamlit_dependencies_are_pinned():
    requirements = (PACKAGE / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    expected = {
        'streamlit==1.64.0', 'Pillow==12.3.0', 'pandas==3.0.6',
        'matplotlib==3.11.2', 'ipykernel==7.3.0', 'nbformat==5.11.1',
        'nbclient==0.11.0', 'nbconvert==7.17.1',
    }
    assert expected.issubset(set(requirements))


def test_beginner_docs_make_notebook_primary_and_preserve_release_boundary():
    root_readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    package_readme = (PACKAGE / 'README.md').read_text(encoding='utf-8')
    for text in ('PCB_Quality_Inspector.ipynb', 'Select Kernel', 'Run All', 'streamlit run app.py'):
        assert text in package_readme
    assert 'v1.1.0' in root_readme
    assert 'v1.0.0' in root_readme
    assert 'Vercel' in package_readme and 'not' in package_readme


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
