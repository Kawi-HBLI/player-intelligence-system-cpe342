"""Task 1 round 4: re-score the near-miss experiments on the Kaggle metric.

Kaggle scores Task 1 with F2 of the cheater class (label 1). Rounds 1-3 were decided on
macro F2 with a threshold near 0.38; since commit 080d9a0 `_threshold` maximises cheater F2
and picks about 0.18, where the ranking of the variants may differ.

Same folds as the README tables (StratifiedKFold 3, seed 342), the same features
(pis/task1_features.py) and the same CatBoost settings as the pipeline, 12 threads.

Decision rule (as in Task 3): a variant counts only if it beats 'current' by >= +0.0015
cheater F2, wins on all 3 folds (threshold tuned on each fold's OOF) and still wins on the
3-fold split with seed 343. Variants that pass are then run with 10 folds (as Task1/train.py).

Every single-seed CatBoost OOF is cached under outputs/experiments/task1_round4/parts, so seed
averages reuse earlier seeds and an interrupted run resumes. Each variant's OOF is saved as
outputs/experiments/task1_round4/<name>.npy (gitignored). The test file is not read.

Usage: python Task1/experiments_round4.py ../cpe342-kaggle-main [--threads 12] [--fresh]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pis.tabular import _threshold  # noqa: E402
from pis.task1_features import add_imputed, add_range_rules, fit_imputers  # noqa: E402
from experiments_round2 import cv_catboost  # noqa: E402
from experiments_round3 import EXTRA_IMPUTED, fit_extra  # noqa: E402

SEED = 342
OUT = ROOT / 'outputs' / 'experiments' / 'task1_round4'
MIN_GAIN = 0.0015
CURRENT_10FOLD_F2 = 0.8388  # commit 080d9a0, Task1/train.py (10 folds)


class Runner:
    """CatBoost / HGB OOF with an on-disk cache of every single-seed run and its runtime."""

    def __init__(self, x_sets, y, threads):
        self.x_sets, self.y, self.threads = x_sets, y, threads
        (OUT / 'parts').mkdir(parents=True, exist_ok=True)
        self.times_file = OUT / 'parts' / 'times.json'
        self.times = json.loads(self.times_file.read_text()) if self.times_file.exists() else {}

    def _cached(self, key, compute):
        path = OUT / 'parts' / f'{key}.npy'
        if path.exists() and key in self.times:
            return np.load(path), self.times[key]
        started = time.time()
        oof = compute()
        self.times[key] = time.time() - started
        np.save(path, oof)
        self.times_file.write_text(json.dumps(self.times, indent=1))
        return oof, self.times[key]

    def catboost(self, features, folds, split, seed_index=0, **extra):
        tag = '_'.join(f'{k}{v}' for k, v in sorted(extra.items())) or 'base'
        key = f'cb_{features}_{tag}_{split}_s{seed_index}'
        # cv_catboost seeds fold f with 342 + f + 1000 * s: s = 0 is exactly the pipeline's seed.
        return self._cached(key, lambda: cv_catboost(self.x_sets[features], self.y, folds, seeds=(seed_index,),
                                                     thread_count=self.threads, **extra))

    def catboost_avg(self, features, folds, split, n_seeds, **extra):
        runs = [self.catboost(features, folds, split, s, **extra) for s in range(n_seeds)]
        return np.mean([r[0] for r in runs], axis=0), sum(r[1] for r in runs)

    def hgb(self, folds, split):
        def compute():
            x, oof = self.x_sets['full'], np.zeros(len(self.y))
            with threadpool_limits(limits=self.threads):
                for fold, (a, b) in enumerate(folds, 1):  # settings of Task1/experiments_round2.py
                    model = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                           early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
                                                           random_state=SEED + fold).fit(x.iloc[a], self.y[a])
                    oof[b] = model.predict_proba(x.iloc[b])[:, 1]
            return oof
        return self._cached(f'hgb_full_{split}', compute)


def variants(run):
    """name -> function(folds, split) returning (oof, runtime in seconds)."""
    def blend(w):
        def f(folds, split):
            cb, t1 = run.catboost_avg('full', folds, split, 3)
            hgb, t2 = run.hgb(folds, split)
            return w * cb + (1 - w) * hgb, t1 + t2
        return f
    return {
        'current': lambda folds, split: run.catboost('full', folds, split),
        'seeds3': lambda folds, split: run.catboost_avg('full', folds, split, 3),
        'seeds5': lambda folds, split: run.catboost_avg('full', folds, split, 5),
        'drop_weak': lambda folds, split: run.catboost('weak_dropped', folds, split),
        'drop_weak_seeds5': lambda folds, split: run.catboost_avg('weak_dropped', folds, split, 5),
        'cb3_hgb_70_30': blend(0.7),
        'cb3_hgb_50_50': blend(0.5),
        'cb3_hgb_85_15': blend(0.85),
        'spw_1.5': lambda folds, split: run.catboost('full', folds, split, scale_pos_weight=1.5),
        'spw_2.0': lambda folds, split: run.catboost('full', folds, split, scale_pos_weight=2.0),
        'depth8': lambda folds, split: run.catboost('full', folds, split, depth=8, l2_leaf_reg=5),  # as round 1
        'lr0.03_3000': lambda folds, split: run.catboost('full', folds, split, learning_rate=0.03, iterations=3000,
                                                         early_stopping_rounds=200),  # as round 1
        'extra_imputed': lambda folds, split: run.catboost('extra_imputed', folds, split),
    }


def score(y, oof, folds):
    threshold, f2 = _threshold(y, oof)
    per_fold = [_threshold(y[b], oof[b])[1] for _, b in folds]
    return {'AUC': roc_auc_score(y, oof), 'F2': f2, 'threshold': threshold, 'per_fold': per_fold,
            'macro_F2_ref': fbeta_score(y, (oof >= threshold).astype(int), beta=2, average='macro')}


def line(name, s, ref=None, runtime=None):
    gain = f"{s['F2'] - ref['F2']:+.4f}" if ref else '   ref '
    wins = ''
    if ref and len(s['per_fold']) == len(ref['per_fold']):
        wins = f"{sum(a > b for a, b in zip(s['per_fold'], ref['per_fold']))}/{len(s['per_fold'])}"
    folds = ' / '.join(f'{v:.4f}' for v in s['per_fold']) if len(s['per_fold']) <= 3 else f"{len(s['per_fold'])} folds"
    rt = f'{runtime:5.0f}s' if runtime is not None else ''
    return (f"{name:18s} AUC {s['AUC']:.4f}  F2 {s['F2']:.4f} ({gain})  thr {s['threshold']:.3f}  "
            f"folds {folds}  wins {wins:3s}  macro F2 {s['macro_F2_ref']:.4f}  {rt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--fresh', action='store_true', help='delete cached OOF and recompute everything')
    args = parser.parse_args()
    if args.fresh and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_root / '01_binary_classification' / 'train.csv')
    labelled = train[train['is_cheater'].notna()].reset_index(drop=True)
    y = labelled['is_cheater'].astype(int).to_numpy()
    raw = labelled.drop(columns=['player_id', 'is_cheater'])
    x = add_range_rules(add_imputed(raw, fit_imputers(raw, SEED)))
    # Weak raw features exactly as in round 2 (selection uses all labels: mildly optimistic).
    weak = [c for c in raw.columns if abs(roc_auc_score(y[raw[c].notna()], raw.loc[raw[c].notna(), c]) - 0.5) < 0.02]
    extra = add_imputed(raw, fit_extra(raw, EXTRA_IMPUTED))
    x_sets = {'full': x, 'weak_dropped': x.drop(columns=weak),
              'extra_imputed': x.join(extra[[f'{c}_imp' for c in EXTRA_IMPUTED]])}
    print(f'{len(y):,} labelled rows | {x.shape[1]} features | {len(weak)} weak raw features dropped in drop_weak | '
          f'threads {args.threads}', flush=True)

    run = Runner(x_sets, y, args.threads)
    table = variants(run)

    def split(k, seed):
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(x, y))

    # Stage 1: every variant on the README folds.
    folds = split(3, SEED)
    results = {}
    print('\nStage 1: StratifiedKFold 3, seed 342 (cheater-class F2, threshold tuned on the OOF)', flush=True)
    for name, fn in table.items():
        oof, runtime = fn(folds, 's342k3')
        np.save(OUT / f'{name}.npy', oof)
        results[name] = (score(y, oof, folds), runtime)
        print(line(name, results[name][0], results['current'][0] if name != 'current' else None, runtime), flush=True)
        if name == 'current':
            pipeline = ROOT / 'outputs' / 'experiments' / 'task1_oof.csv'
            if pipeline.exists():
                same = np.allclose(pd.read_csv(pipeline)['probability_1'].to_numpy(), oof, atol=1e-12)
                print(f'  current == pipeline OOF (Task1/eda/make_oof.py, 12 threads): {same}', flush=True)

    ref = results['current'][0]
    first = [n for n, (s, _) in results.items() if n != 'current' and s['F2'] - ref['F2'] >= MIN_GAIN
             and all(a > b for a, b in zip(s['per_fold'], ref['per_fold']))]
    print(f'\nPass stage 1 (>= +{MIN_GAIN} and all 3 folds): {first or "none"}', flush=True)

    # Stage 2: split seed 343, only for variants that passed stage 1.
    second = []
    if first:
        folds343 = split(3, 343)
        print('\nStage 2: StratifiedKFold 3, seed 343', flush=True)
        cur343 = score(y, table['current'](folds343, 's343k3')[0], folds343)
        print(line('current', cur343), flush=True)
        for name in first:
            oof, runtime = table[name](folds343, 's343k3')
            np.save(OUT / f'{name}_split343.npy', oof)
            s = score(y, oof, folds343)
            print(line(name, s, cur343, runtime), flush=True)
            if s['F2'] > cur343['F2']:
                second.append(name)
        print(f'Pass stage 2 (still wins on split seed 343): {second or "none"}', flush=True)

    # Stage 3: 10 folds, as Task1/train.py.
    if second:
        folds10 = split(10, SEED)
        print(f'\nStage 3: 10 folds (commit 080d9a0 reports current 10-fold F2 {CURRENT_10FOLD_F2})', flush=True)
        cur10 = score(y, table['current'](folds10, 's342k10')[0], folds10)
        np.save(OUT / 'current_10fold.npy', table['current'](folds10, 's342k10')[0])
        print(line('current', cur10), flush=True)
        for name in second:
            oof, runtime = table[name](folds10, 's342k10')
            np.save(OUT / f'{name}_10fold.npy', oof)
            print(line(name, score(y, oof, folds10), cur10, runtime), flush=True)

    summary = {n: {**{k: v for k, v in s.items()}, 'runtime_s': rt} for n, (s, rt) in results.items()}
    (OUT / 'stage1.json').write_text(json.dumps({'stage1': summary, 'pass_stage1': first, 'pass_stage2': second}, indent=1))
    print(f'\nSaved OOF and stage1.json under {OUT}', flush=True)


if __name__ == '__main__':
    main()
