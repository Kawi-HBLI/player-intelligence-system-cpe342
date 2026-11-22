"""Task 1 experiments: compare feature sets against the baseline on identical folds.

Uses the same split as the pipeline (StratifiedKFold 3, seed 342) and the same CatBoost
settings, so the 'baseline' row reproduces `python Task1/train.py --folds 3`.

Usage: python Task1/experiments.py <data_root> [variant ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.tabular import _threshold  # noqa: E402
from pis.task1_features import add_imputed, add_range_rules, fit_imputers  # noqa: E402

SEED, FOLDS = 342, 3
def ratios(x):
    x = x.copy()
    age = x['account_age_days'].clip(lower=1)
    x['level_per_day'] = x['level'] / age
    x['age_from_speed'] = x['level'] / x['level_progression_speed'].clip(lower=1e-3)
    x['speed_x_age'] = x['level_progression_speed'] * age
    x['friends_per_day'] = x['friend_network_size'] / age
    x['reports_per_level'] = x['reports_received'] / x['level'].clip(lower=1)
    return x


def missing(x):
    x = x.copy()
    base = [c for c in x.columns]
    x['n_missing'] = x[base].isna().sum(axis=1)
    for c in ('reports_received', 'crosshair_placement', 'account_age_days', 'friend_network_size', 'level'):
        x[f'{c}_missing'] = x[c].isna().astype(int)
    return x


def margins(x):
    return add_range_rules(x)


def impute(x):
    """The imputers never see the label, so fitting them on all rows does not leak it."""
    return add_imputed(x, fit_imputers(x, SEED))


VARIANTS = {
    'margins+impute': (lambda x: margins(impute(x)), {}),  # this is what pis/task1_features.build does
    'impute': (impute, {}),
    'baseline': (lambda x: x, {}),
    'ratios': (ratios, {}),
    'missing': (missing, {}),
    'margins': (margins, {}),
    'ratios+missing+margins': (lambda x: margins(missing(ratios(x))), {}),
    'depth8': (lambda x: x, {'depth': 8, 'l2_leaf_reg': 5}),
    'lr003_3000': (lambda x: x, {'learning_rate': 0.03, 'iterations': 3000, 'early_stopping_rounds': 200}),
}


def run(name, x, y):
    build, extra = VARIANTS[name]
    x = build(x)
    params = dict(iterations=1000, depth=6, learning_rate=0.05, verbose=False, allow_writing_files=False,
                  early_stopping_rounds=80, loss_function='Logloss', thread_count=-1)
    params.update(extra)
    oof = np.zeros(len(y))
    start = time.time()
    for fold, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y), 1):
        model = CatBoostClassifier(random_seed=SEED + fold, **params)
        model.fit(x.iloc[a], y[a], eval_set=(x.iloc[b], y[b]), use_best_model=True)
        oof[b] = model.predict_proba(x.iloc[b])[:, 1]
    threshold, tuned = _threshold(y, oof)
    fixed = fbeta_score(y, (oof >= 0.5).astype(int), beta=2, average='macro')
    print(f'{name:26s} AUC {roc_auc_score(y, oof):.4f}  F2@0.5 {fixed:.4f}  F2 tuned {tuned:.4f} '
          f'(thr {threshold:.3f})  features {x.shape[1]}  {time.time() - start:.0f}s', flush=True)
    return oof


def main():
    root = Path(sys.argv[1])
    names = sys.argv[2:] or list(VARIANTS)
    train = pd.read_csv(root / '01_binary_classification' / 'train.csv')
    train = train[train['is_cheater'].notna()].reset_index(drop=True)
    y = train['is_cheater'].astype(int).to_numpy()
    x = train.drop(columns=['player_id', 'is_cheater'])
    for name in names:
        run(name, x, y)


if __name__ == '__main__':
    main()
