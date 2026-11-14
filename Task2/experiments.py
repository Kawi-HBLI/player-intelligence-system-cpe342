"""Task 2 experiments on the pipeline folds (StratifiedKFold 3, seed 342).

The 'baseline' row reproduces `python Task2/train.py --folds 3`. OOF probability tables are
saved to <out_dir> so blends can be recomputed without retraining.

Usage: python Task2/experiments.py <data_root> <out_dir> [variant ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.task2_naive_bayes import SmoothNaiveBayes, geometric_blend  # noqa: E402

SEED, FOLDS = 342, 3
BASE = dict(iterations=1000, depth=6, learning_rate=0.05, early_stopping_rounds=80)
CATBOOST = {
    'baseline': (BASE, False),
    'baseline, no random_metric': (BASE, True),
    'depth 6, 4000 iters': ({**BASE, 'iterations': 4000, 'early_stopping_rounds': 200}, True),
    'depth 3, lr 0.1': (dict(iterations=6000, depth=3, learning_rate=0.1, early_stopping_rounds=300), True),
    'depth 2, lr 0.1': (dict(iterations=8000, depth=2, learning_rate=0.1, early_stopping_rounds=300), True),
    'depth 1, lr 0.15': (dict(iterations=10000, depth=1, learning_rate=0.15, early_stopping_rounds=300), True),
}


def report(name, y, p, started=None):
    took = f'  {time.time() - started:.0f}s' if started else ''
    per_class = ' '.join(f'{v:.3f}' for v in f1_score(y, p.argmax(1), average=None))
    print(f'{name:38s} macro F1 {f1_score(y, p.argmax(1), average="macro"):.4f}  logloss {log_loss(y, p):.4f}  '
          f'[{per_class}]{took}', flush=True)


def prepare(frame, drop_noise):
    x = frame.drop(columns=['player_id', 'segment'], errors='ignore')
    if drop_noise:
        x = x[[c for c in x.columns if not c.startswith('random_metric')]]
    cats = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    x = x.copy()
    for c in cats:
        x[c] = x[c].fillna('__MISSING__').astype(str)
    return x, cats


def cv_catboost(train, y, folds, params, drop_noise):
    x, cats = prepare(train, drop_noise)
    oof = np.zeros((len(y), 4))
    best = []
    for fold, (a, b) in enumerate(folds, 1):
        model = CatBoostClassifier(loss_function='MultiClass', random_seed=SEED + fold, verbose=False,
                                   allow_writing_files=False, thread_count=-1, **params)
        model.fit(x.iloc[a], y[a], cat_features=cats, eval_set=(x.iloc[b], y[b]), use_best_model=True)
        oof[b] = model.predict_proba(x.iloc[b])
        best.append(model.get_best_iteration() + 1)
    return oof, best


def cv_naive_bayes(train, y, folds, sigma):
    x = train.drop(columns=['player_id', 'segment'])
    x = x[[c for c in x.columns if not c.startswith('random_metric')]]
    oof = np.zeros((len(y), 4))
    for a, b in folds:
        oof[b] = SmoothNaiveBayes(sigma=sigma).fit(x.iloc[a], y[a]).predict_proba(x.iloc[b])
    return oof


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[3:]
    train = pd.read_csv(root / '02_multiclass_classification' / 'train.csv')
    y = train['segment'].astype(int).to_numpy()
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(train, y))
    tables = {}
    for sigma in (5, 10, 20):
        name = f'naive Bayes sigma={sigma}'
        if wanted and name not in wanted:
            continue
        t = time.time(); tables[name] = cv_naive_bayes(train, y, folds, sigma); report(name, y, tables[name], t)
    for name, (params, drop_noise) in CATBOOST.items():
        if wanted and name not in wanted:
            continue
        t = time.time(); tables[name], best = cv_catboost(train, y, folds, params, drop_noise)
        report(name + f' {best}', y, tables[name], t)
    for name, table in tables.items():
        np.save(out / ('task2_oof_' + name.replace(' ', '_').replace(',', '').replace('=', '') + '.npy'), table)
    nb = min((n for n in tables if n.startswith('naive')), key=lambda n: log_loss(y, tables[n]), default=None)
    cb = min((n for n in tables if not n.startswith('naive')), key=lambda n: log_loss(y, tables[n]), default=None)
    if nb and cb:
        print(f'blending "{cb}" with "{nb}"')
        for w in (0.3, 0.5, 0.7):
            report(f'geometric blend, NB weight {w}', y, geometric_blend(tables[cb], tables[nb], w))


if __name__ == '__main__':
    main()
