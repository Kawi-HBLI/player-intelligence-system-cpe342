"""Missing-aware generative classifiers for experiments/task1/eda/model_gap_eda.ipynb (analysis only).

GaussianCopulaNB: per class, every feature keeps the smoothed marginal that
pis.task2_naive_bayes.SmoothNaiveBayes uses (200-bin histogram smoothed with a Gaussian
kernel for continuous features, smoothed frequency table for features with <= 40 values).
Each value is mapped to a normal score through that class's marginal CDF, and the class
correlation matrix of the normal scores is estimated on pairwise-complete rows and made
positive definite. A row is scored with its observed features only:

    log p(x_o | k) = sum_j log f_kj(x_j) - 0.5 log|R_oo| - 0.5 z_o' (R_oo^-1 - I) z_o

so missing values are marginalised exactly (the marginal of a Gaussian copula is the copula of
the observed sub-matrix). With R = I this is naive Bayes skipping missing values. `shrink` pulls
R toward the identity (shrink * R + (1 - shrink) * I).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtri

DISCRETE_MAX_UNIQUE = 40


class _Marginal:
    """Smoothed per-class marginal of one feature: log density / mass and mid-CDF."""

    def __init__(self, values, y, classes, bins, sigma, alpha):
        v = np.asarray(values, dtype=float)
        known = ~np.isnan(v)
        self.discrete = len(np.unique(v[known])) <= DISCRETE_MAX_UNIQUE
        if self.discrete:
            self.levels = np.unique(v[known])
            table = []
            for k in classes:
                counts = np.array([np.sum(v[known & (y == k)] == lv) for lv in self.levels], dtype=float)
                table.append((counts + alpha) / (counts.sum() + alpha * len(self.levels)))
            self.prob = np.vstack(table)                       # (classes, levels)
            self.cdf = np.cumsum(self.prob, axis=1)
        else:
            self.low, self.high = np.quantile(v[known], [0.001, 0.999])
            self.edges = np.linspace(self.low, self.high, bins + 1)
            self.width = self.edges[1] - self.edges[0]
            table = []
            for k in classes:
                counts = np.histogram(np.clip(v[known & (y == k)], self.low, self.high), self.edges)[0].astype(float)
                smooth = gaussian_filter1d(counts, sigma, mode='nearest') + 1e-3
                table.append(smooth / smooth.sum())
            self.prob = np.vstack(table)                       # (classes, bins), bin probabilities
            self.cdf = np.cumsum(self.prob, axis=1)

    def score(self, values, ci):
        """(log density or mass, mid-CDF in (0, 1)) for class index ci; NaN stays NaN."""
        v = np.asarray(values, dtype=float)
        out_l, out_u = np.full(len(v), np.nan), np.full(len(v), np.nan)
        known = ~np.isnan(v)
        if self.discrete:
            idx = np.searchsorted(self.levels, v[known])
            idx = np.clip(idx, 0, len(self.levels) - 1)
            exact = self.levels[idx] == v[known]
            p = np.where(exact, self.prob[ci, idx], 1e-6)       # unseen level: tiny mass, no evidence either way
            out_l[known] = np.log(p)
            out_u[known] = self.cdf[ci, idx] - self.prob[ci, idx] / 2
        else:
            x = np.clip(v[known], self.low, self.high)
            b = np.clip(((x - self.low) / self.width).astype(int), 0, self.prob.shape[1] - 1)
            p = self.prob[ci, b]
            out_l[known] = np.log(p / self.width)
            frac = np.clip((x - self.edges[b]) / self.width, 0, 1)
            out_u[known] = self.cdf[ci, b] - p + p * frac
        return out_l, out_u


def _nearest_correlation(r, floor=1e-3):
    r = np.nan_to_num((r + r.T) / 2, nan=0.0)
    np.fill_diagonal(r, 1.0)
    w, v = np.linalg.eigh(r)
    r = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(r))
    return r / np.outer(d, d)


class GaussianCopulaNB:
    def __init__(self, bins=200, sigma=5.0, alpha=0.5, clip=5.0, identity=False, min_pairs=200, shrink=1.0):
        self.bins, self.sigma, self.alpha, self.clip = bins, sigma, alpha, clip
        self.identity, self.min_pairs, self.shrink = identity, min_pairs, shrink

    def _normal_scores(self, x, ci):
        logf = np.full(x.shape, np.nan)
        z = np.full(x.shape, np.nan)
        for j, m in enumerate(self.marginals_):
            logf[:, j], u = m.score(x[:, j], ci)
            z[:, j] = np.clip(ndtri(np.clip(u, 1e-12, 1 - 1e-12)), -self.clip, self.clip)
        return logf, z

    def fit(self, x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.log_prior_ = np.log(np.array([(y == k).mean() for k in self.classes_]))
        self.marginals_ = [_Marginal(x[:, j], y, self.classes_, self.bins, self.sigma, self.alpha) for j in range(x.shape[1])]
        self.corr_ = []
        for ci, k in enumerate(self.classes_):
            if self.identity:
                self.corr_.append(np.eye(x.shape[1]))
                continue
            _, z = self._normal_scores(x[y == k], ci)
            r = pd.DataFrame(z).corr(min_periods=self.min_pairs).to_numpy()
            r = _nearest_correlation(r)
            self.corr_.append(self.shrink * r + (1 - self.shrink) * np.eye(len(r)))
        return self

    def log_scores(self, x):
        x = np.asarray(x, dtype=float)
        n, d = x.shape
        observed = ~np.isnan(x)
        # Group rows by missing pattern so each sub-matrix is factorised once per class.
        keys = np.packbits(observed, axis=1)
        _, pattern = np.unique(keys, axis=0, return_inverse=True)
        pattern = pattern.ravel()
        out = np.zeros((n, len(self.classes_)))
        for ci in range(len(self.classes_)):
            logf, z = self._normal_scores(x, ci)
            total = np.nansum(logf, axis=1) + self.log_prior_[ci]
            if not self.identity:
                r = self.corr_[ci]
                for g in np.unique(pattern):
                    rows = np.flatnonzero(pattern == g)
                    o = observed[rows[0]]
                    if o.sum() < 2:
                        continue
                    sub = r[np.ix_(o, o)]
                    c = np.linalg.cholesky(sub)
                    zo = z[np.ix_(rows, np.flatnonzero(o))]
                    sol = np.linalg.solve(c, zo.T)                       # c^-1 z
                    quad = (sol ** 2).sum(axis=0) - (zo ** 2).sum(axis=1)
                    total[rows] += -np.log(np.diag(c)).sum() - 0.5 * quad
            out[:, ci] = total
        return out

    def predict_proba(self, x):
        s = self.log_scores(x)
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(axis=1, keepdims=True)
