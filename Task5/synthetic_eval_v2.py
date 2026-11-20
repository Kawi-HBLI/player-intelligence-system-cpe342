"""Task 5 label-free check for the v2 detector: semi-synthetic injection, v1 vs v2 vs union.

Normal rows: real test rows that v2 gives posterior below 0.01 (about 23.8k rows, the bulk
outside every anomaly group found). Anomalies of four kinds are injected into a random subset, with
families, directions and magnitudes drawn at random per dataset, so the check does not
reuse the exact patterns found in the real data:
  0  coordinated shift of 3-5 families in period 4 (or 3-4), optional location jump + flag
  1  1-D island: whole series of one family moved far below the normal range
  2  multivariate corner group: 2-3 families moved to their joint 1-6% tail corner
  3  subtle shift of 2 random families in period 4 (2.5-5 sd)
Each detector picks its own number of flags (expected Macro-F3 rule), so the numbers rank
whole pipelines, including the choice of k. They are not the leaderboard F3.

Usage: python Task5/synthetic_eval_v2.py <data_root> [out.csv]
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.anomaly import score_anomalies, select_expected_macro_f3  # noqa: E402

FLAGS = ['password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale']
SEEDS = (1, 2, 3)
PREVALENCES = (0.05, 0.08, 0.10)


def _families(x):
    return sorted({re.sub(r'_[1-4]$', '', c) for c in x if re.search(r'_[1-4]$', c)})


def inject(base, prev, rng):
    """Return (frame, labels, kind per row) with `prev` share of injected anomalies."""
    x = base.copy().reset_index(drop=True)
    n = len(x)
    cont = [f for f in _families(x) if f not in ('login_lat', 'login_lon') and x[f'{f}_1'].nunique() > 50]
    lvl = {f: x[[f'{f}_{p}' for p in range(1, 5)]].mean(axis=1) for f in cont}
    positive = [f for f in cont if np.nanquantile(lvl[f], 0.01) > 0.15 * np.nanmedian(lvl[f])]
    m = int(round(prev * n))
    idx = rng.choice(n, m, replace=False)
    kinds = rng.integers(0, 4, m)
    island_fams = rng.choice(positive, 3, replace=False)
    others = [f for f in cont if f not in island_fams]
    mv_sets = [rng.choice(others, rng.integers(2, 4), replace=False) for _ in range(2)]
    mv_dirs = [rng.choice([-1, 1], len(s)) for s in mv_sets]
    shift_sets = [rng.choice(cont, rng.integers(3, 6), replace=False) for _ in range(3)]
    sd = {f: float(np.nanstd(x[f'{f}_4'] - x[[f'{f}_{i}' for i in range(1, 4)]].mean(axis=1))) for f in cont}
    for i, kind in zip(idx, kinds):
        if kind == 0:
            periods = [4] if rng.random() < 0.7 else [3, 4]
            for f in shift_sets[rng.integers(0, 3)]:
                step = rng.uniform(4, 12) * rng.choice([-1, 1]) * sd[f]
                for p in periods:
                    x.at[i, f'{f}_{p}'] += step
            if rng.random() < 0.5:
                dlat, dlon = rng.uniform(15, 50) * rng.choice([-1, 1]), rng.uniform(20, 90) * rng.choice([-1, 1])
                for p in periods:
                    x.at[i, f'login_lat_{p}'] += dlat
                    x.at[i, f'login_lon_{p}'] += dlon
            if rng.random() < 0.6:
                x.at[i, rng.choice(FLAGS)] = 1.0
        elif kind == 1:
            f = island_fams[rng.integers(0, 3)]
            target = np.nanquantile(lvl[f], 0.01) * rng.uniform(0.15, 0.45)
            for p in range(1, 5):
                if pd.notna(x.at[i, f'{f}_{p}']):
                    x.at[i, f'{f}_{p}'] = target * (1 + 0.05 * rng.normal())
        elif kind == 2:
            g = rng.integers(0, 2)
            for f, direction in zip(mv_sets[g], mv_dirs[g]):
                q = rng.uniform(0.94, 0.99) if direction > 0 else rng.uniform(0.01, 0.06)
                move = np.nanquantile(lvl[f], q) - np.nanmean([x.at[i, f'{f}_{p}'] for p in range(1, 5)])
                for p in range(1, 5):
                    x.at[i, f'{f}_{p}'] += move
        else:
            for f in rng.choice(cont, 2, replace=False):
                x.at[i, f'{f}_4'] += rng.uniform(2.5, 5) * rng.choice([-1, 1]) * sd[f]
    y = np.zeros(n, dtype=int)
    y[idx] = 1
    kind_row = np.full(n, -1)
    kind_row[idx] = kinds
    return x, y, kind_row


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    full = pd.read_csv(root / '05_anomaly_detection' / 'test.csv').drop(columns='player_id')
    clean = score_anomalies(full, seed=342, profile='balanced', method='v2')['anomaly_probability'] < 0.01
    base = full[clean].reset_index(drop=True)
    print(f'normal pool: {len(base)} rows with v2 posterior < 0.01', flush=True)
    rows = []
    for seed in SEEDS:
        for prev in PREVALENCES:
            x, y, kinds = inject(base, prev, np.random.default_rng(seed * 100 + int(prev * 100) + 7))
            preds = {}
            for method in ('v1', 'v2'):
                t0 = time.time()
                views = score_anomalies(x, seed=342, profile='balanced', method=method)
                preds[method] = (select_expected_macro_f3(views['anomaly_probability'])[0], time.time() - t0)
            preds['union'] = (np.maximum(preds['v1'][0], preds['v2'][0]), 0.0)
            for name, (pred, secs) in preds.items():
                row = {'seed': seed, 'prev': prev, 'detector': name, 'flags': int(pred.sum()), 'true': int(y.sum()),
                       'false_pos': int(((pred == 1) & (y == 0)).sum()),
                       'macro_f3': round(fbeta_score(y, pred, beta=3, average='macro'), 4), 'secs': round(secs)}
                row.update({f'recall_kind{k}': round(float(pred[kinds == k].mean()), 3) for k in range(4)})
                rows.append(row)
                print(row, flush=True)
    table = pd.DataFrame(rows)
    print(table.pivot_table(index=['seed', 'prev'], columns='detector', values='macro_f3').round(4))
    print(table.groupby('detector')[['flags', 'false_pos', 'macro_f3'] + [f'recall_kind{k}' for k in range(4)]].mean().round(3))
    if out:
        table.to_csv(out, index=False)


if __name__ == '__main__':
    main()
