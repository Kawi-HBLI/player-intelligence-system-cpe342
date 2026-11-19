"""Task 4: score finished runs out-of-fold, compare TTA view sets, build ensembles.

A "group" is one recipe trained on all 5 folds of the same split, e.g. cnt=cnt_nc_f{fold}
means runs cnt_nc_f0 .. cnt_nc_f4. Every image of train.csv is held out by exactly one
fold, so stacking the five validation sets gives one out-of-fold (OOF) prediction per
training image. All choices here (TTA views, which groups to average) use these OOF
predictions only; test logits are only averaged and written out.

    python Task4/oof.py --group cnt=cnt_nc_f{fold}
    python Task4/oof.py --group cnt=cnt_nc_f{fold} --group cns=cns_nc_f{fold} --views id,hflip
        --write-pred outputs/task4_exp/ens_a/task4_predictions.csv

Needs val_logits.npy / val_idx.npy of every run (written by training) and, for TTA,
val_logits_tta.npy / test_logits_tta.npy (python Task4/train.py --mode tta --run-name <run>).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as t4  # noqa: E402  (constants and paths only; no GPU needed here)

PRESET_VIEW_SETS = {
    'none': ['id'],
    'flip': ['id', 'hflip'],
    'rot8': ['id', 'rot+8', 'rot-8'],
    'zoom': ['id', 'zoom1.12', 'zoom0.9'],
    'flip+rot8': ['id', 'hflip', 'rot+8', 'rot-8'],
    'rot8+zoom': ['id', 'rot+8', 'rot-8', 'zoom1.12', 'zoom0.9'],
    'all': list(t4.TTA_VIEWS),
}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(truth, predicted, average='macro', zero_division=0))


class Run:
    """Validation / test softmax of one finished run, per TTA view."""

    def __init__(self, name: str) -> None:
        self.name, self.dir = name, t4.EXP_ROOT / name
        if not (self.dir / 'val_idx.npy').is_file():
            raise SystemExit(f'{self.dir} has no val_idx.npy: run not finished')
        self.val_idx = np.load(self.dir / 'val_idx.npy')
        if (self.dir / 'val_logits_tta.npy').is_file():
            self.views = json.loads((self.dir / 'tta_views.json').read_text(encoding='utf-8'))
            self.val = softmax(np.load(self.dir / 'val_logits_tta.npy'))
            self.test = softmax(np.load(self.dir / 'test_logits_tta.npy'))
        else:
            self.views = ['id']
            self.val = softmax(np.load(self.dir / 'val_logits.npy'))[None]
            test_file = self.dir / 'test_logits.npy'
            self.test = softmax(np.load(test_file))[None] if test_file.is_file() else None

    def has(self, views: list[str]) -> bool:
        return all(v in self.views for v in views)

    def val_probs(self, views: list[str]) -> np.ndarray:
        return self.val[[self.views.index(v) for v in views]].mean(0)

    def test_probs(self, views: list[str]) -> np.ndarray:
        if self.test is None:
            raise SystemExit(f'{self.name}: no test logits; run --mode tta (or --mode predict) first')
        return self.test[[self.views.index(v) for v in views]].mean(0)


class Group:
    """One recipe on all folds: OOF probabilities over the whole training set."""

    def __init__(self, label: str, pattern: str, folds: list[int], n_images: int) -> None:
        self.label = label
        self.runs = [Run(pattern.format(fold=fold)) for fold in folds]
        self.n_images = n_images

    def available_views(self) -> list[str]:
        common = set(self.runs[0].views)
        for run in self.runs[1:]:
            common &= set(run.views)
        return [v for v in t4.TTA_VIEWS if v in common]

    def oof(self, views: list[str]) -> np.ndarray:
        probs = np.full((self.n_images, t4.NUM_CLASSES), np.nan, dtype=np.float32)
        for run in self.runs:
            probs[run.val_idx] = run.val_probs(views)
        return probs

    def test(self, views: list[str]) -> np.ndarray:
        return np.mean([run.test_probs(views) for run in self.runs], axis=0)


def fold_scores(labels: np.ndarray, probs: np.ndarray, fold_sets: list[np.ndarray]) -> list[float]:
    return [macro_f1(labels[idx], probs[idx].argmax(1)) for idx in fold_sets]


def score_line(name: str, labels: np.ndarray, probs: np.ndarray, fold_sets: list[np.ndarray]) -> str:
    covered = np.concatenate(fold_sets)
    folds = fold_scores(labels, probs, fold_sets)
    wrong = int((probs[covered].argmax(1) != labels[covered]).sum())
    return (f'  {name:34s} OOF {macro_f1(labels[covered], probs[covered].argmax(1)):.4f}  '
            f'wrong {wrong:4d}  folds ' + ' '.join(f'{f:.4f}' for f in folds))


def main() -> None:
    parser = argparse.ArgumentParser(description='Task 4 out-of-fold scores, TTA and ensembles')
    parser.add_argument('--group', action='append', default=[],
                        help='label=run-name pattern with {fold}, e.g. cnt=cnt_nc_f{fold} (repeatable)')
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--views', default='none',
                        help=f'TTA views to average: a preset ({", ".join(PRESET_VIEW_SETS)}) or a comma list')
    parser.add_argument('--weights', default='', help='comma weights per group for the ensemble (default equal)')
    parser.add_argument('--search', action='store_true',
                        help='also score every pair/triple of groups (OOF only) to see which mix helps')
    parser.add_argument('--extra-test', default='',
                        help='comma list of runs added to the test average only (e.g. scene-split runs)')
    parser.add_argument('--write-pred', default='', help='write the ensemble test predictions here')
    parser.add_argument('--data-root', default='')
    args = parser.parse_args()
    if args.data_root:
        t4.DATA_DIR = Path(args.data_root).resolve()
    labels = pd.read_csv(t4.DATA_DIR / 'train.csv')['label'].to_numpy(dtype=int)
    folds = [int(f) for f in args.folds.split(',')]
    groups = []
    for spec in args.group:
        label, _, pattern = spec.partition('=')
        groups.append(Group(label, pattern, folds, len(labels)))
    if not groups:
        raise SystemExit('give at least one --group label=pattern')
    fold_sets = [run.val_idx for run in groups[0].runs]
    for group in groups[1:]:
        for a, b in zip(fold_sets, group.runs):
            if not np.array_equal(np.sort(a), np.sort(b.val_idx)):
                raise SystemExit(f'{group.label}: {b.name} holds out other images than {groups[0].label}; '
                                 'groups must share the same folds')
    views = PRESET_VIEW_SETS.get(args.views, args.views.split(','))

    print(f'== single groups ({len(folds)} folds, {sum(len(f) for f in fold_sets):,} OOF images) ==')
    for group in groups:
        available = group.available_views()
        for preset, preset_views in PRESET_VIEW_SETS.items():
            if all(v in available for v in preset_views):
                print(score_line(f'{group.label} [{preset}]', labels, group.oof(preset_views), fold_sets))
        if len(available) > 1:
            singles = '  '.join(f'{v} {macro_f1(labels[np.concatenate(fold_sets)], group.oof([v])[np.concatenate(fold_sets)].argmax(1)):.4f}'
                                for v in available)
            print(f'    single views: {singles}')

    usable = [g for g in groups if all(v in g.available_views() for v in views)]
    if len(usable) < len(groups):
        print(f'note: {len(groups) - len(usable)} group(s) lack views {views}; ensemble uses view "id" for all')
        views = ['id']
    weights = [float(w) for w in args.weights.split(',')] if args.weights else [1.0] * len(groups)
    oofs = [g.oof(views) for g in groups]
    ensemble = sum(w * p for w, p in zip(weights, oofs)) / sum(weights)
    print(f'\n== ensemble of {len(groups)} group(s), views {views}, weights {weights} ==')
    print(score_line('ensemble', labels, ensemble, fold_sets))
    if args.search and len(groups) > 2:
        for size in range(2, len(groups)):
            for combo in combinations(range(len(groups)), size):
                mix = np.mean([oofs[i] for i in combo], axis=0)
                print(score_line('+'.join(groups[i].label for i in combo), labels, mix, fold_sets))

    covered = np.concatenate(fold_sets)
    predicted = ensemble[covered].argmax(1)
    matrix = confusion_matrix(labels[covered], predicted, labels=list(range(t4.NUM_CLASSES)))
    per_class = f1_score(labels[covered], predicted, average=None, labels=list(range(t4.NUM_CLASSES)))
    print('  per class F1: ' + '  '.join(f'{t4.TEAM_LABELS[c]} {per_class[c]:.4f}' for c in range(t4.NUM_CLASSES)))
    print('  confusion (rows = true, cols = predicted):')
    for c in range(t4.NUM_CLASSES):
        print(f'    {t4.TEAM_LABELS[c]:10s} ' + ' '.join(f'{v:6d}' for v in matrix[c]))

    if args.write_pred:
        test_probs = sum(w * g.test(views) for w, g in zip(weights, groups)) / sum(weights)
        extra = [name for name in args.extra_test.split(',') if name]
        if extra:
            # Extra runs count like one more group, each run equally inside it.
            extra_probs = np.mean([Run(name).test_probs(['id'] if not Run(name).has(views) else views)
                                   for name in extra], axis=0)
            test_probs = (test_probs * sum(weights) + extra_probs) / (sum(weights) + 1)
        out = Path(args.write_pred)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out.with_suffix('.probs.npy'), test_probs.astype(np.float32))
        test = pd.read_csv(t4.DATA_DIR / 'test.csv')
        stems = test['file_name'].str.replace(r'\.[A-Za-z0-9]+$', '', regex=True)
        pd.DataFrame({'player_id': stems, 'prediction': [t4.TEAM_LABELS[i] for i in test_probs.argmax(1)]}).to_csv(
            out, index=False)
        print(f'\nwrote {out} ({len(test):,} test rows) and {out.with_suffix(".probs.npy").name}')


if __name__ == '__main__':
    main()
