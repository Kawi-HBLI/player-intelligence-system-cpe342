"""Task 1 round 2: model-side ideas on top of the task1_features set.

Same folds as the pipeline (StratifiedKFold 3, seed 342). The 'reference' row equals
`python Task1/train.py --folds 3`.

Usage: python Task1/experiments_round2.py <data_root>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.tabular import _threshold  # noqa: E402
from pis.task1_features import IMPUTED, add_imputed, add_range_rules, fit_imputers  # noqa: E402

SEED, FOLDS = 342, 3
CB = dict(iterations=1000, depth=6, learning_rate=0.05, verbose=False, allow_writing_files=False,
          early_stopping_rounds=80, loss_function='Logloss', thread_count=-1)


def report(name, y, oof, started):
    threshold, tuned = _threshold(y, oof)
    fixed = fbeta_score(y, (oof >= 0.5).astype(int), beta=2, average='macro')
    print(f'{name:34s} AUC {roc_auc_score(y, oof):.4f}  F2@0.5 {fixed:.4f}  F2 tuned {tuned:.4f} '
          f'(thr {threshold:.3f})  {time.time() - started:.0f}s', flush=True)


def catboost_fold(xa, ya, xb, yb, seed, **extra):
    model = CatBoostClassifier(random_seed=seed, **{**CB, **extra})
    model.fit(xa, ya, eval_set=(xb, yb), use_best_model=True)
    return model


def cv_catboost(x, y, folds, seeds=(0,), **extra):
    oof = np.zeros(len(y))
    for fold, (a, b) in enumerate(folds, 1):
        for s in seeds:
            model = catboost_fold(x.iloc[a], y[a], x.iloc[b], y[b], SEED + fold + 1000 * s, **extra)
            oof[b] += model.predict_proba(x.iloc[b])[:, 1] / len(seeds)
    return oof


def dense(x, raw_columns, stats=None):
    """Standardised matrix without NaN: imputed copies where they exist, medians elsewhere."""
    cols = {c: (x[f'{c}_imp'] if c in IMPUTED else x[c]) for c in raw_columns}
    frame = pd.DataFrame(cols)
    if stats is None:
        stats = (frame.median(), frame.std().replace(0, 1))
    frame = frame.fillna(stats[0])
    return ((frame - stats[0]) / stats[1]).to_numpy(), stats


def gmm_llr(za, ya, zb, k):
    cheat = GaussianMixture(k, covariance_type='full', reg_covar=1e-3, random_state=SEED).fit(za[ya == 1])
    legit = GaussianMixture(k, covariance_type='full', reg_covar=1e-3, random_state=SEED).fit(za[ya == 0])
    return cheat.score_samples(zb) - legit.score_samples(zb)


def main():
    root = Path(sys.argv[1])
    train = pd.read_csv(root / '01_binary_classification' / 'train.csv')
    labelled = train[train['is_cheater'].notna()].reset_index(drop=True)
    unlabelled = train[train['is_cheater'].isna()].reset_index(drop=True)
    y = labelled['is_cheater'].astype(int).to_numpy()
    raw = labelled.drop(columns=['player_id', 'is_cheater'])
    raw_columns = list(raw.columns)
    imputers = fit_imputers(raw, SEED)
    x = add_range_rules(add_imputed(raw, imputers))
    xu = add_range_rules(add_imputed(unlabelled[raw_columns], imputers))
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y))

    t = time.time(); ref = cv_catboost(x, y, folds); report('reference (current pipeline)', y, ref, t)

    t = time.time(); seeds3 = cv_catboost(x, y, folds, seeds=(0, 1, 2)); report('3-seed average', y, seeds3, t)

    for name, extra in (('depth 4, 3000 iters', dict(depth=4, iterations=3000, early_stopping_rounds=200)),
                        ('depth 6, l2=10, rsm=0.5', dict(l2_leaf_reg=10, rsm=0.5, iterations=2000, early_stopping_rounds=150))):
        t = time.time(); report(name, y, cv_catboost(x, y, folds, **extra), t)

    # Drop the raw features with almost no univariate signal (selection uses all labels: mildly optimistic).
    weak = [c for c in raw_columns if abs(roc_auc_score(y[raw[c].notna()], raw.loc[raw[c].notna(), c]) - 0.5) < 0.02]
    t = time.time(); report(f'drop {len(weak)} weak raw features', y, cv_catboost(x.drop(columns=weak), y, folds), t)

    t = time.time(); hgb = np.zeros(len(y))
    for fold, (a, b) in enumerate(folds, 1):
        model = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
                                               random_state=SEED + fold).fit(x.iloc[a], y[a])
        hgb[b] = model.predict_proba(x.iloc[b])[:, 1]
    report('HistGradientBoosting alone', y, hgb, t)
    report('blend CatBoost 3-seed + HGB (50/50)', y, (seeds3 + hgb) / 2, t)
    report('blend CatBoost 3-seed + HGB (70/30)', y, 0.7 * seeds3 + 0.3 * hgb, t)

    # Pseudo-label confident unlabelled TRAIN rows (never test rows), per fold.
    t = time.time(); pseudo = np.zeros(len(y)); added = []
    for fold, (a, b) in enumerate(folds, 1):
        first = catboost_fold(x.iloc[a], y[a], x.iloc[b], y[b], SEED + fold)
        pu = first.predict_proba(xu)[:, 1]
        keep = (pu > 0.97) | (pu < 0.03)
        added.append(int(keep.sum()))
        xa = pd.concat([x.iloc[a], xu[keep]], ignore_index=True)
        ya = np.r_[y[a], (pu[keep] > 0.5).astype(int)]
        pseudo[b] = catboost_fold(xa, ya, x.iloc[b], y[b], SEED + fold).predict_proba(x.iloc[b])[:, 1]
    report(f'pseudo-label unlabelled (+{int(np.mean(added))} rows)', y, pseudo, t)

    # Generative view: per-class Gaussian mixtures, log-likelihood ratio as score and as feature.
    for k in (1, 3):
        t = time.time(); llr_oof = np.zeros(len(y)); with_llr = np.zeros(len(y))
        for fold, (a, b) in enumerate(folds, 1):
            za, stats = dense(x.iloc[a], raw_columns)
            zb, _ = dense(x.iloc[b], raw_columns, stats)
            llr_b = gmm_llr(za, y[a], zb, k)
            llr_a = np.zeros(len(a))   # out-of-fold inside the training part, so the booster cannot over-trust it
            for ia, ib in StratifiedKFold(3, shuffle=True, random_state=SEED).split(za, y[a]):
                llr_a[ib] = gmm_llr(za[ia], y[a][ia], za[ib], k)
            llr_oof[b] = llr_b
            xa = x.iloc[a].assign(gmm_llr=llr_a); xb = x.iloc[b].assign(gmm_llr=llr_b)
            with_llr[b] = catboost_fold(xa, y[a], xb, y[b], SEED + fold).predict_proba(xb)[:, 1]
        report(f'GMM k={k} log-likelihood ratio alone', y, llr_oof, t)
        report(f'CatBoost + GMM k={k} LLR feature', y, with_llr, t)


if __name__ == '__main__':
    main()
