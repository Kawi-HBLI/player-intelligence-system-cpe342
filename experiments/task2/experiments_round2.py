"""Task 2 round 2: better density estimates for naive Bayes, and more training rows per fold.

Part A uses the pipeline's folds (StratifiedKFold 3, seed 342) so its rows compare directly
with the 0.7567 reference. It tries choosing the smoothing width of each feature's density
separately, by held-out likelihood inside the training fold, instead of one sigma for all.

Part B is a learning curve for the blend that the pipeline uses: 3, 5 and 10 folds, so each
fold model trains on 67%, 80% or 90% of the rows. Two extra 3-fold splits (seeds 343, 344)
show how much the score moves from the split alone.

Everything is fitted on train only; the test file is not read.

Usage: python experiments/task2/experiments_round2.py <data_root>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / 'run.py').is_file())))
from pis.task2_naive_bayes import DISCRETE_MAX_UNIQUE, SmoothNaiveBayes, geometric_blend  # noqa: E402
from experiments import CATBOOST, cv_catboost  # noqa: E402

SEED = 342
WEIGHT = 0.6  # naive Bayes share in the pipeline blend
SIGMAS = (2, 3, 5, 8, 12, 20)


class TunedSigmaNaiveBayes(SmoothNaiveBayes):
    """SmoothNaiveBayes where every continuous feature gets its own sigma.

    For each feature the sigma with the highest held-out log-likelihood of the class-conditional
    density is chosen on a 50/50 split of the training rows, then the density is refitted on
    all training rows with that sigma.
    """

    def fit(self, x, y):
        super().fit(x, y)
        y = np.asarray(y)
        half = np.random.default_rng(SEED).random(len(y)) < 0.5
        self.sigma_ = {}
        for column, (edges, _) in self.continuous_.items():
            v = np.clip(x[column].to_numpy(dtype=float), edges[0], edges[-1])
            known = ~np.isnan(v)
            index = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, self.bins - 1)
            scores = []
            for sigma in SIGMAS:
                total = 0.0
                for k in self.classes_:
                    fit_rows, eval_rows = known & (y == k) & half, known & (y == k) & ~half
                    table = self._log_density(v[fit_rows], edges, sigma)
                    total += table[index[eval_rows]].sum()
                scores.append(total)
            sigma = SIGMAS[int(np.argmax(scores))]
            self.sigma_[column] = sigma
            self.continuous_[column] = (edges, np.vstack(
                [self._log_density(v[known & (y == k)], edges, sigma) for k in self.classes_]))
        return self

    def _log_density(self, values, edges, sigma):
        counts = np.histogram(values, edges)[0].astype(float)
        smooth = gaussian_filter1d(counts, sigma, mode='nearest') + 1e-3
        return np.log(smooth / smooth.sum())


def bayes_columns(train):
    x = train.drop(columns=['player_id', 'segment'])
    return x[[c for c in x.columns if not c.startswith('random_metric')]]


def cv_bayes(x, y, folds, model_class):
    oof = np.zeros((len(y), 4))
    for a, b in folds:
        oof[b] = model_class().fit(x.iloc[a], y[a]).predict_proba(x.iloc[b])
    return oof


def macro_f1(y, p):
    return f1_score(y, p.argmax(1), average='macro')


def report(name, y, p, folds, started):
    per_fold = ' / '.join(f'{macro_f1(y[b], p[b]):.4f}' for _, b in folds) if len(folds) == 3 else ''
    print(f'{name:44s} macro F1 {macro_f1(y, p):.4f}  {per_fold}  {time.time() - started:.0f}s', flush=True)


def main():
    root = Path(sys.argv[1])
    train = pd.read_csv(root / '02_multiclass_classification' / 'train.csv')
    y = train['segment'].astype(int).to_numpy()
    xb = bayes_columns(train)
    params, drop_noise = CATBOOST['baseline']

    def split(k, seed=SEED):
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(train, y))

    folds = split(3)
    print('Part A: pipeline folds')
    t = time.time(); cat = cv_catboost(train, y, folds, params, drop_noise)[0]; report('CatBoost depth 6', y, cat, folds, t)
    t = time.time(); nb = cv_bayes(xb, y, folds, SmoothNaiveBayes); report('naive Bayes sigma 5', y, nb, folds, t)
    report('reference blend (NB 0.6)', y, geometric_blend(cat, nb, WEIGHT), folds, time.time())
    t = time.time(); tuned = cv_bayes(xb, y, folds, TunedSigmaNaiveBayes)
    report('naive Bayes, sigma per feature', y, tuned, folds, t)
    for w in (0.5, 0.6, 0.7, 0.8):
        report(f'blend with sigma per feature (NB {w})', y, geometric_blend(cat, tuned, w), folds, time.time())
    chosen = TunedSigmaNaiveBayes().fit(xb, y).sigma_
    print('sigma chosen on all rows:', pd.Series(chosen).value_counts().sort_index().to_dict(), flush=True)

    print('Part B: learning curve (reference blend, more training rows per fold)')
    for k, seed in ((3, 343), (3, 344), (5, SEED), (10, SEED)):
        f = split(k, seed)
        t = time.time()
        cat_k = cv_catboost(train, y, f, params, drop_noise)[0]
        nb_k = cv_bayes(xb, y, f, SmoothNaiveBayes)
        name = f'{k} folds, split seed {seed}' if k == 3 else f'{k} folds ({100 - 100 // k}% train per model)'
        report(name, y, geometric_blend(cat_k, nb_k, WEIGHT), f, t)
        print(f'{"":4s}CatBoost alone {macro_f1(y, cat_k):.4f}  naive Bayes alone {macro_f1(y, nb_k):.4f}', flush=True)


if __name__ == '__main__':
    main()
