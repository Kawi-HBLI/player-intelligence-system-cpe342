"""Task 5 label-free check: inject synthetic anomalies, then compare detectors and flag rates.

Synthetic anomalies copy the patterns seen in the real test data: abrupt period-4 shifts in
location, skill, economy and social features, plus security flags. Real (unknown) anomalies in
the base rows act as label noise for every detector equally, so the numbers rank detectors and
flag rates; they are not the leaderboard F3.

Usage: python Task5/synthetic_eval.py <data_root> [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pis.anomaly import shift_score, temporal_features  # noqa: E402

SEED = 342
FLAG_RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
PREVALENCES = (0.05, 0.10, 0.15)
TYPES = {
    'location': ['login_lat_4', 'login_lon_4', 'location_changes_4', 'ip_hash_entropy_4', 'device_count_4'],
    'skill': ['skill_rating_4', 'skill_trajectory_4', 'accuracy_mean_4', 'kda_mean_4', 'reaction_time_mean_4'],
    'economy': ['currency_transfers_4', 'purchase_frequency_4', 'purchase_amount_mean_4', 'refund_count_4'],
    'social': ['friend_removes_4', 'friend_adds_4', 'guild_changes_4', 'communication_frequency_4'],
}
FLAG_FOR_TYPE = {'location': 'vpn_usage', 'skill': 'suspicious_login_time',
                 'economy': 'mass_item_sale', 'social': 'password_changed'}
DOWN = {'reaction_time_mean_4'}  # features that move down when the anomaly appears


def robust_sd(col):
    v = col.dropna()
    return float((v - v.median()).abs().median() * 1.4826) or float(v.std()) or 1.0


def inject(base, prevalence, rng):
    x = base.copy()
    n = len(x)
    idx = rng.choice(n, int(round(prevalence * n)), replace=False)
    sd = {c: robust_sd(x[c]) for cols in TYPES.values() for c in cols}
    names = list(TYPES)
    for i in idx:
        for kind in rng.choice(names, rng.integers(1, 3), replace=False):
            k = rng.uniform(1.5, 8.0)  # mixes subtle (1.5 sd) and blatant (8 sd) shifts
            for c in TYPES[kind]:
                if pd.isna(x.iat[i, x.columns.get_loc(c)]):
                    continue
                step = k * sd[c] * (-1 if c in DOWN else 1)
                if c in ('login_lat_4', 'login_lon_4'):
                    step *= rng.choice([-1, 1])
                x.iat[i, x.columns.get_loc(c)] += step
            if rng.random() < 0.5:
                x.iat[i, x.columns.get_loc(FLAG_FOR_TYPE[kind])] = 1.0
    x['login_lat_4'] = x['login_lat_4'].clip(-90, 90)
    x['login_lon_4'] = ((x['login_lon_4'] + 180) % 360) - 180
    y = np.zeros(n, dtype=int)
    y[idx] = 1
    return x, y


def iforest(values, seed):
    filled = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(values)
    model = IsolationForest(n_estimators=300, max_samples=min(4096, len(values)), random_state=seed, n_jobs=-1)
    return pd.Series(-model.fit(filled).score_samples(filled)).rank(pct=True).to_numpy()


def detectors(x):
    raw = iforest(x, SEED)
    temporal = iforest(temporal_features(x), SEED + 1)
    shift = shift_score(x)
    return {'current_if_raw+temporal': (raw + temporal) / 2,
            'if_temporal_only': temporal,
            'shift_z': shift,
            'if_temporal+shift': (temporal + shift) / 2,
            'if_raw+temporal+shift': (raw + temporal + shift) / 3}


def macro_f3_at(y, score, q):
    pred = np.zeros(len(y), dtype=int)
    pred[np.argsort(-score, kind='stable')[:int(np.ceil(q * len(y)))]] = 1
    return fbeta_score(y, pred, beta=3, average='macro')


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    base = pd.read_csv(root / '05_anomaly_detection' / 'test.csv').drop(columns='player_id')
    rng = np.random.default_rng(SEED)
    results = {}
    for p in PREVALENCES:
        x, y = inject(base, p, rng)
        for name, score in detectors(x).items():
            row = {'auc': round(roc_auc_score(y, score), 4)}
            row.update({f'q{q:.2f}': round(macro_f3_at(y, score, q), 4) for q in FLAG_RATES})
            results[f'p{p:.2f}|{name}'] = row
            print(f'p={p:.2f} {name:26s} ' + ' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
    if out:
        out.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
