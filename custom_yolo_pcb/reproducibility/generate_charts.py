"""Rebuild balanced comparison figures using only recorded CSV/JSON evidence."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MODELS = {'original': ('Original', 'original', 'original'), 'trial044': ('Trial 044', 'trial044_gpu_adaptation', 'enhanced')}
COLORS = {'original': '#2266aa', 'trial044': '#d47126'}


def load_evidence():
    evidence = {}
    for key, (label, folder, metric) in MODELS.items():
        csv_path = PACKAGE / 'results' / folder / 'training/results.csv'
        json_path = PACKAGE / 'results/metrics' / f'{metric}_clean_validation.json'
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        record = json.loads(json_path.read_text())
        if record['test_split_used'] is not False or record['validation']['split'] != 'val':
            raise PermissionError('Only validation evidence is permitted')
        best = max(rows, key=lambda row: row['metrics/mAP50-95(B)'])
        evidence[key] = dict(label=label, rows=rows, record=record, best_epoch=int(best['epoch']), sources=[csv_path, json_path])
    return evidence


def generate():
    evidence = load_evidence()
    output = PACKAGE / 'results/charts'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
    for key, data in evidence.items():
        rows = data['rows']
        epochs = [r['epoch'] for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for ax, loss in zip(axes[0], ('box', 'cls', 'dfl')):
            for split, style in (('train', '-'), ('val', '--')):
                ax.plot(epochs, [r[f'{split}/{loss}_loss'] for r in rows], style, label=split)
            ax.set(title=f'{loss.upper()} loss', xlabel='Epoch', ylabel='Loss', xlim=(1, len(rows)))
            ax.legend()
        for ax, metrics, title in zip(axes[1], (('precision', 'recall'), ('mAP50',), ('mAP50-95',)), ('Precision and recall', 'mAP50', 'mAP50-95')):
            for metric in metrics:
                ax.plot(epochs, [r[f'metrics/{metric}(B)'] for r in rows], label=metric, color=COLORS[key] if len(metrics) == 1 else None)
            ax.axvline(data['best_epoch'], color='#666666', linestyle=':', label=f"Best epoch {data['best_epoch']}")
            ax.set(title=title, xlabel='Epoch', ylabel='Score', ylim=(0, 1), xlim=(1, len(rows)))
            ax.legend(fontsize=8)
        subtitle = '100 completed epochs' if key == 'original' else '27 completed epochs | best epoch 12 | patience=15'
        fig.suptitle(f"{data['label']} recorded training — {subtitle}", fontsize=15)
        fig.savefig(output / f'{key}_training.png')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    metrics = ('precision', 'recall', 'f1', 'map50', 'map50_95')
    x = np.arange(len(metrics))
    for i, (key, data) in enumerate(evidence.items()):
        bars = ax.bar(x + (i - .5) * .35, [data['record']['validation'][m] for m in metrics], .35, label=data['label'], color=COLORS[key])
        ax.bar_label(bars, fmt='%.3f', fontsize=9)
    ax.set(xticks=x, xticklabels=['Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95'], ylim=(0, 1.12), ylabel='Validation score', title='Recorded clean validation — different validation authorities')
    ax.legend(loc='lower right')
    fig.savefig(output / 'final_comparison.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ('precision', 'recall', 'ap50', 'ap50_95'), ('Precision', 'Recall', 'AP50', 'AP50-95')):
        for i, (key, data) in enumerate(evidence.items()):
            rows = data['record']['validation']['per_class']
            ax.bar(np.arange(6) + (i - .5) * .35, [r[metric] for r in rows], .35, label=data['label'], color=COLORS[key])
        ax.set(xticks=np.arange(6), xticklabels=[r['class_name'].replace('_', '\n') for r in rows], ylim=(0, 1.05), title=title, ylabel='Validation score')
        ax.legend(fontsize=8)
    fig.suptitle('Per-class recorded clean validation — different authorities', fontsize=15)
    fig.savefig(output / 'per_class.png')
    plt.close(fig)
    provenance = {'test_split_used': False, 'sources': {p.relative_to(PACKAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for d in evidence.values() for p in d['sources']}, 'epochs': {k: {'completed': len(v['rows']), 'best': v['best_epoch']} for k, v in evidence.items()}}
    (output / 'source_hashes.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    generate()
