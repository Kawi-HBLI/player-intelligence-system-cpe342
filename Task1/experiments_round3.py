"""Task 1 round 3: more imputed columns, and how much more training data helps.

Part A uses the pipeline's folds (StratifiedKFold 3, seed 342), so its rows compare
directly with the 0.7807 reference. Part B is a learning curve: the same model with 3, 5
and 10 folds. Each fold model then trains on 67%, 80% or 90% of the labelled rows. Two
extra 3-fold splits (seeds 343, 344) show how much the score moves from the split alone,
so a gain from more folds can be told apart from split noise.

Everything is fitted on train only; the test file is not read.

Usage: python Task1/experiments_round3.py <data_root>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.tabular import _threshold  # noqa: E402
from pis.task1_features import add_imputed, add_range_rules, fit_imputers  # noqa: E402
from experiments_round2 import cv_catboost  # noqa: E402

SEED = 342
# Columns with R2 >= 0.1 from the other features that are not imputed yet (see README).
EXTRA_IMPUTED = ('reaction_time_ms', 'aiming_smoothness', 'spray_control_score', 'kill_death_ratio')


def fit_extra(x, columns, seed=SEED):
    models = {}
    for column in columns:
        known = x[column].notna()
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
        models[column] = model.fit(x.loc[known].drop(columns=column), x.loc[known, column])
    return models


def report(name, y, oof, folds, started):
    threshold, tuned = _threshold(y, oof)
    per_fold = ' / '.join(f'{_threshold(y[b], oof[b])[1]:.4f}' for _, b in folds) if len(folds) == 3 else ''
    print(f'{name:40s} AUC {roc_auc_score(y, oof):.4f}  F2 {tuned:.4f} (thr {threshold:.3f})  {per_fold}  '
          f'{time.time() - started:.0f}s', flush=True)


def main():
    root = Path(sys.argv[1])
    train = pd.read_csv(root / '01_binary_classification' / 'train.csv')
    labelled = train[train['is_cheater'].notna()].reset_index(drop=True)
    y = labelled['is_cheater'].astype(int).to_numpy()
    raw = labelled.drop(columns=['player_id', 'is_cheater'])
    x = add_range_rules(add_imputed(raw, fit_imputers(raw, SEED)))

    def split(k, seed=SEED):
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(x, y))

    folds = split(3)
    print('Part A: pipeline folds')
    t = time.time(); report('reference', y, cv_catboost(x, y, folds), folds, t)
    extra = add_imputed(raw[list(raw.columns)], fit_extra(raw, EXTRA_IMPUTED))
    x_extra = x.join(extra[[f'{c}_imp' for c in EXTRA_IMPUTED]])
    t = time.time(); report('+ 4 more imputed columns', y, cv_catboost(x_extra, y, folds), folds, t)

    print('Part B: learning curve (same model, more training rows per fold)')
    for seed in (343, 344):
        f = split(3, seed)
        t = time.time(); report(f'3 folds, split seed {seed}', y, cv_catboost(x, y, f), f, t)
    for k in (5, 10):
        f = split(k)
        t = time.time(); report(f'{k} folds ({100 - 100 // k}% train per model)', y, cv_catboost(x, y, f), f, t)


if __name__ == '__main__':
    main()
