"""Smoothed naive Bayes for Task 2.

EDA showed the Task 2 features are close to independent inside each segment (largest
within-class Spearman correlation 0.04), which is the assumption naive Bayes makes. Each
continuous feature gets a per-class histogram smoothed with a Gaussian kernel; low-cardinality
numeric and categorical features get smoothed frequency tables. A missing value adds nothing
to the score, which is the right treatment when values are missing at random.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

DISCRETE_MAX_UNIQUE = 40


class SmoothNaiveBayes:
    def __init__(self, bins=200, sigma=5.0, alpha=0.5):
        self.bins, self.sigma, self.alpha = bins, sigma, alpha

    def fit(self, x, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.log_prior_ = np.log(np.array([(y == k).mean() for k in self.classes_]))
        self.continuous_, self.discrete_ = {}, {}
        for column in x.columns:
            values = x[column]
            known = values.notna().to_numpy()
            if pd.api.types.is_numeric_dtype(values) and values.nunique() > DISCRETE_MAX_UNIQUE:
                v = values.to_numpy(dtype=float)
                low, high = np.quantile(v[known], [0.001, 0.999])
                edges = np.linspace(low, high, self.bins + 1)
                table = []
                for k in self.classes_:
                    counts = np.histogram(np.clip(v[known & (y == k)], low, high), edges)[0].astype(float)
                    smooth = gaussian_filter1d(counts, self.sigma, mode='nearest') + 1e-3
                    table.append(np.log(smooth / smooth.sum()))
                self.continuous_[column] = (edges, np.vstack(table))
            else:
                v = values.astype(object).to_numpy()
                levels = pd.unique(v[known])
                table = {}
                for i, k in enumerate(self.classes_):
                    counts = pd.Series(v[known & (y == k)]).value_counts()
                    total = counts.sum() + self.alpha * len(levels)
                    for level in levels:
                        table.setdefault(level, np.zeros(len(self.classes_)))[i] = np.log(
                            (counts.get(level, 0) + self.alpha) / total)
                self.discrete_[column] = table
        return self

    def feature_log_likelihoods(self, x):
        """Array (rows, features, classes): each feature's log-likelihood per class, 0 if missing."""
        names = list(self.continuous_) + list(self.discrete_)
        out = np.zeros((len(x), len(names), len(self.classes_)))
        for j, (column, (edges, table)) in enumerate(self.continuous_.items()):
            v = x[column].to_numpy(dtype=float)
            known = ~np.isnan(v)
            index = np.clip(np.searchsorted(edges, v[known], side='right') - 1, 0, table.shape[1] - 1)
            out[known, j] = table[:, index].T
        offset = len(self.continuous_)
        for j, (column, table) in enumerate(self.discrete_.items()):
            v = x[column].astype(object).to_numpy()
            known = pd.notna(x[column]).to_numpy()
            # A level never seen in training carries no evidence.
            add = np.array([table.get(level, np.zeros(len(self.classes_))) for level in v[known]])
            if len(add):
                out[known, offset + j] = add
        self.feature_names_ = names
        return out

    def log_scores(self, x):
        return self.log_prior_ + self.feature_log_likelihoods(x).sum(axis=1)

    def predict_proba(self, x):
        scores = self.log_scores(x)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)


def geometric_blend(p, q, weight):
    """Weighted geometric mean of two probability tables (weight is for q)."""
    blend = np.exp((1 - weight) * np.log(p + 1e-9) + weight * np.log(q + 1e-9))
    return blend / blend.sum(axis=1, keepdims=True)
