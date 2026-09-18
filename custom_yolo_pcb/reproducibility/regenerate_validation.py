"""Generate new validation-only plots from the preserved best checkpoints."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'model_code'))
from tools import run_local_vscode_comparison as runner

MODELS = {'original': dict(checkpoint='original_best.pt', authority='original', imgsz=640, batch=8), 'trial044': dict(checkpoint='trial044_best.pt', authority='enhanced', imgsz=1024, batch=3)}
VALIDATION_SETTINGS = dict(split='val', conf=0.001, iou=0.7, max_det=300, augment=False, plots=True, workers=0)


def regenerate(output):
    from ultralytics import YOLO
    import yaml

    output = Path(output).resolve()
    runner.require_new_output_directory(output)
    runner.verify_packaged_inputs()
    environment = runner.runtime_environment()
    if environment['ultralytics'] != '8.4.84':
        raise RuntimeError('Recorded Ultralytics 8.4.84 is required')
    runtime = PACKAGE / 'reproducibility/runtime' / ('validation_' + time.strftime('%Y%m%d_%H%M%S'))
    runner.require_new_output_directory(runtime)
    authorities = runner.prepare_data_authorities(runtime)
    for name, model in MODELS.items():
        checkpoint = PACKAGE / 'weights' / model['checkpoint']
        runner.require_published_artifact(checkpoint)
        authority = authorities[model['authority']]
        data_yaml = Path(authority['data_yaml'])
        data = yaml.safe_load(data_yaml.read_text())
        if set(data) != {'path', 'nc', 'names', 'train', 'val'} or authority['test_split_used'] is not False:
            raise PermissionError('Authority must contain only the recorded train and val roles')
        started = runner.utc_now()
        timer = time.perf_counter()
        detector = YOLO(str(checkpoint))
        metrics = detector.val(data=str(data_yaml), imgsz=model['imgsz'], batch=model['batch'], device=runner.select_device(), project=str(output), name=name, exist_ok=False, **VALIDATION_SETTINGS)
        destination = output / name
        examples = destination / 'prediction_examples'
        examples.mkdir()
        for image in destination.glob('val_batch*.jpg'):
            shutil.move(str(image), examples / image.name)
        record = {
            'label': 'Regenerated validation plots; separate from recorded final metrics',
            'started_at_utc': started, 'ended_at_utc': runner.utc_now(), 'elapsed_seconds': time.perf_counter() - timer,
            'checkpoint': checkpoint.relative_to(PACKAGE).as_posix(), 'checkpoint_sha256': runner.sha256(checkpoint),
            'authority_manifest': Path(authority['val_manifest']).name,
            'published_authority_sha256': runner.sha256(runner.PUBLISHED_AUTHORITY / Path(authority['val_manifest']).name),
            'runtime_authority_sha256': runner.sha256(Path(authority['val_manifest'])), 'runtime_yaml_sha256': runner.sha256(data_yaml),
            'source_authority_sha256': authority['val_sha256'], 'validation_entries': authority['val_entries'],
            'settings': dict(VALIDATION_SETTINGS, imgsz=model['imgsz'], batch=model['batch'], device=runner.select_device()),
            'environment': {k: v for k, v in environment.items() if k != 'python_executable'},
            'metrics': dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr), map50=float(metrics.box.map50), map50_95=float(metrics.box.map)),
            'test_split_used': False, 'training_performed': False,
        }
        record['generated_files'] = {p.relative_to(destination).as_posix(): runner.sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
        runner.atomic_json(destination / 'provenance.json', record)
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PACKAGE / 'results/regenerated_validation')
    regenerate(parser.parse_args().output_dir)
