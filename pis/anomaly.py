"""Task 5: unsupervised multi-view anomaly detection calibrated for Macro-F3."""
from pathlib import Path
import re
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .common import write_json

LOG = logging.getLogger(__name__)

SECURITY_FLAGS = ('password_changed', 'vpn_usage', 'suspicious_login_time', 'mass_item_sale')
_GEO_FAMILIES = {'login_lat', 'login_lon'}
_STATIC_LEVEL_COLS = ('account_age_days', 'behavior_stability')

_PROFILE_CONFIG = {
    'quick': {'n_splits': 3, 'n_bags': 1, 'n_proj': 90, 'max_iter': 120, 'max_bins': 63, 'n_rep': 2, 'n_eval_null': 1},
    'balanced': {'n_splits': 5, 'n_bags': 2, 'n_proj': 120, 'max_iter': 140, 'max_bins': 63, 'n_rep': 2, 'n_eval_null': 2},
    'strong': {'n_splits': 5, 'n_bags': 3, 'n_proj': 140, 'max_iter': 160, 'max_bins': 63, 'n_rep': 3, 'n_eval_null': 2},
}


def _complete_families(frame):
    """Return sorted base names for complete 4-period feature families."""
    candidates = sorted({re.sub(r'_[1-4]$', '', col) for col in frame if re.search(r'_[1-4]$', col)})
    return [fam for fam in candidates if all(f'{fam}_{i}' in frame for i in range(1, 5))]


def _haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in kilometers between coordinate pairs."""
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    hav = np.sin((la2 - la1) / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def _robust_scale(values):
    """Scale values robustly around the median using MAD with quantile/std fallbacks."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)) * 1.4826)
    if mad <= 1e-9:
        q15, q85 = np.quantile(valid, [0.15, 0.85])
        mad = float((q85 - q15) / 2.073)
    if mad <= 1e-9:
        mad = float(np.std(valid))
    return med, max(mad, 1e-6)


def temporal_features(frame):
    """Extract classic 4-period trajectory deltas and period-to-period geo distances."""
    result = {}
    for family in _complete_families(frame):
        values = frame[[f'{family}_{i}' for i in range(1, 5)]]
        history = values.iloc[:, :3]
        mean = history.mean(axis=1)
        delta = values.iloc[:, 3] - mean
        result[f'{family}_last_delta'] = delta
        result[f'{family}_relative_delta'] = delta / (mean.abs() + 1)
        result[f'{family}_range'] = values.max(axis=1) - values.min(axis=1)
        result[f'{family}_first_last'] = values.iloc[:, 3] - values.iloc[:, 0]
    for i in range(1, 4):
        cols = [f'login_lat_{i}', f'login_lon_{i}', f'login_lat_{i+1}', f'login_lon_{i+1}']
        if all(c in frame for c in cols):
            result[f'location_distance_km_{i}_{i+1}'] = _haversine_km(
                frame[cols[0]], frame[cols[1]], frame[cols[2]], frame[cols[3]]
            )
    return pd.DataFrame(result, index=frame.index)


def shift_score(frame):
    """Percentile rank of abrupt per-period shifts plus security flags.

    For every 4-period family and period j, take the robust z of (value_j - mean of the
    other periods). A row's score is the sum of log1p of its three largest z values plus
    0.5 per raised security flag. Rows whose periods are all missing get z = 0.
    """
    zs = []
    for family in _complete_families(frame):
        v = frame[[f'{family}_{i}' for i in range(1, 5)]].to_numpy(dtype=float)
        for j in range(4):
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                d = v[:, j] - np.nanmean(np.delete(v, j, axis=1), axis=1)
            med = np.nanmedian(d)
            mad = np.nanmedian(np.abs(d - med)) * 1.4826 + 1e-9
            zs.append(np.nan_to_num(np.abs((d - med) / mad)))
    if not zs:
        raise ValueError('shift_score needs at least one complete 4-period feature family')
    top = np.sort(np.vstack(zs).T, axis=1)[:, ::-1][:, :3]
    flags = frame[[c for c in SECURITY_FLAGS if c in frame]].fillna(0).sum(axis=1).to_numpy()
    return pd.Series(np.log1p(top).sum(axis=1) + 0.5 * flags).rank(method='average', pct=True).to_numpy()


def extract_multiview_features(frame):
    """Build level features and MAD-standardized within-player shift magnitudes.

    Multiplicative families (where early baseline magnitude correlates with change size)
    are log1p-transformed automatically. Geographic coordinates are excluded from static
    level means (so region is never flagged) and summarized via max pairwise haversine
    jump in shift magnitudes.
    """
    families = _complete_families(frame)
    level_dict, shift_mag_dict = {}, {}

    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for fam in families:
            v = frame[[f'{fam}_{i}' for i in range(1, 5)]].astype(float)
            p1, p2 = v.iloc[:, 0], v.iloc[:, 1]
            valid12 = p1.notna() & p2.notna()
            is_mult = False
            if int(valid12.sum()) > 20 and float(v.min().min()) >= 0.0 and float(np.nanmedian(v.to_numpy())) > 0.1:
                corr = ((p1[valid12] + p2[valid12]) / 2.0).corr((p2[valid12] - p1[valid12]).abs(), method='spearman')
                is_mult = bool(np.isfinite(corr) and abs(corr) > 0.20)
            lv = np.log1p(v.clip(lower=0.0)) if is_mult else v
            if fam in _GEO_FAMILIES:
                continue
            level_dict[fam] = lv.mean(axis=1, skipna=True)
            d4 = lv.iloc[:, 3] - lv.iloc[:, :3].mean(axis=1, skipna=True)
            dh = lv.iloc[:, 2:4].mean(axis=1, skipna=True) - lv.iloc[:, :2].mean(axis=1, skipna=True)
            mags = []
            for ser in (d4, dh):
                med, mad = _robust_scale(ser.to_numpy(dtype=float))
                mags.append(np.nan_to_num(np.abs((ser.to_numpy(dtype=float) - med) / mad), nan=0.0))
            shift_mag_dict[fam] = np.maximum(mags[0], mags[1])

        if all(f'login_lat_{i}' in frame and f'login_lon_{i}' in frame for i in range(1, 5)):
            h_pairs = [
                _haversine_km(frame[f'login_lat_{i}'], frame[f'login_lon_{i}'], frame[f'login_lat_{j}'], frame[f'login_lon_{j}'])
                for i in range(1, 4) for j in range(i + 1, 5)
            ]
            hmax = pd.concat(h_pairs, axis=1).max(axis=1)
            lh = np.log1p(hmax.to_numpy(dtype=float))
            med, mad = _robust_scale(lh)
            shift_mag_dict['geo_hav'] = np.nan_to_num(np.clip((lh - med) / mad, 0.0, None), nan=0.0)

    levels = pd.DataFrame(level_dict, index=frame.index)
    if not families:
        # Support general non-temporal numeric tables gracefully
        for col in frame.columns:
            if col not in SECURITY_FLAGS:
                levels[col] = frame[col].astype(float)
    else:
        for col in _STATIC_LEVEL_COLS:
            if col in frame:
                levels[col] = frame[col].astype(float)
    shift_mag = pd.DataFrame(shift_mag_dict, index=frame.index)
    return levels, shift_mag


def topological_island_score(frame, bins=None, min_peak_frac=0.15):
    """Score rows belonging to isolated secondary marginal density islands.

    Detects continuous 1D distributions containing a smaller cluster separated from the
    main distribution bulk by a near-empty saddle valley, skipping discrete lattice features.
    """
    n = len(frame)
    if n == 0 or frame.shape[1] == 0:
        return np.zeros(n, dtype=float)
    scores = np.zeros((n, frame.shape[1]), dtype=float)
    kernel = np.array([0.15, 0.70, 0.15], dtype=float)

    for j, col in enumerate(frame.columns):
        s = frame[col].to_numpy(dtype=float)
        valid = np.isfinite(s)
        vals = s[valid]
        if len(vals) < 60 or len(np.unique(np.round(vals, 4))) < 0.05 * len(vals):
            continue
        n_bins = min(60, max(20, len(vals) // 50)) if bins is None else int(bins)
        q01, q25, q75, q99 = np.quantile(vals, [0.005, 0.25, 0.75, 0.995])
        iqr = max(float(q75 - q25), 1e-6)
        lo = max(float(vals.min()), float(q01 - 0.5 * iqr))
        hi = min(float(vals.max()), float(q99 + 0.5 * iqr))
        if hi <= lo:
            continue
        edges = np.linspace(lo - 1e-9, hi + 1e-9, n_bins + 1)
        hist, _ = np.histogram(vals[(vals >= lo) & (vals <= hi)], bins=edges)
        smooth = np.convolve(hist.astype(float), kernel, mode='same')
        p_max = float(smooth.max())
        if p_max <= 0.0:
            continue
        major = smooth >= min_peak_frac * p_max
        maj_idx = np.where(major)[0]
        if len(maj_idx) == 0:
            continue

        left_cap = np.zeros(n_bins, dtype=float)
        cap = 0.0
        for b in range(n_bins):
            cap = p_max if major[b] else min(cap, smooth[b])
            left_cap[b] = cap
        right_cap = np.zeros(n_bins, dtype=float)
        cap = 0.0
        for b in range(n_bins - 1, -1, -1):
            cap = p_max if major[b] else min(cap, smooth[b])
            right_cap[b] = cap
        best_saddle = np.maximum(left_cap, right_cap)

        bin_score = np.zeros(n_bins, dtype=float)
        min_mass = max(15.0, 0.002 * len(vals))
        for start, end in ((0, int(maj_idx[0])), (int(maj_idx[-1]) + 1, n_bins)):
            for b in range(start, end):
                sad = float(best_saddle[b])
                if sad >= 0.03 * p_max or hist[b] == 0:
                    continue
                valid_c = (best_saddle <= sad + 1e-9) & (smooth > sad + 0.5)
                lo_b, hi_b = b, b
                while lo_b > start and valid_c[lo_b - 1]:
                    lo_b -= 1
                while hi_b + 1 < end and valid_c[hi_b + 1]:
                    hi_b += 1
                comp = slice(lo_b, hi_b + 1)
                isl_mass = float(hist[comp].sum())
                isl_peak = float(smooth[comp].max())
                if isl_mass >= min_mass and isl_peak >= max(3.0, 0.01 * p_max) and isl_peak > 4.0 * (sad + 0.5):
                    bin_score[b] = np.log((isl_peak + 0.5) / (sad + 0.5)) * np.log((0.25 * p_max + 1.0) / (sad + 0.5))

        idx = np.clip(np.searchsorted(edges[1:-1], np.clip(s, lo, hi), side='right'), 0, n_bins - 1)
        col_sc = bin_score[idx]
        col_sc[~valid] = 0.0
        if bin_score[0] == 0.0:
            col_sc[valid & (s < lo)] = 0.0
        if bin_score[-1] == 0.0:
            col_sc[valid & (s > hi)] = 0.0
        scores[:, j] = col_sc

    return scores.sum(axis=1)


def calibrate_vs_clean_null(s_real, s_null):
    """Empirical Bayes density-ratio posterior P(anomaly | s) from real vs clean-null scores."""
    s_real = np.asarray(s_real, dtype=float)
    s_null = np.asarray(s_null, dtype=float)
    if len(s_real) == 0 or len(s_null) == 0 or float(np.std(s_real)) < 1e-12:
        return np.zeros(len(s_real), dtype=float), 1.0
    y = np.concatenate([np.ones(len(s_real)), np.zeros(len(s_null))])
    s = np.concatenate([s_real, s_null])
    w = np.concatenate([np.ones(len(s_real)), np.full(len(s_null), float(len(s_real)) / len(s_null))])
    iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, increasing=True, out_of_bounds='clip')
    iso.fit(s, y, sample_weight=w)
    odds = np.maximum(iso.predict(s_real), 1e-4) / np.maximum(1.0 - iso.predict(s_real), 1e-6)
    t30, t75, t85 = np.quantile(s_null, [0.30, 0.75, 0.85])
    mid = (s_real >= t30) & (s_real <= t75)
    raw_bg = float(np.median(odds[mid])) if mid.any() else 1.0
    bg_scale = float(np.clip(raw_bg, 0.85, 1.00))
    p_anom = np.clip(1.0 - bg_scale / np.maximum(odds, 1e-6), 0.0, 1.0)
    p_anom = np.where((s_real <= t85) | (odds <= 1.00) | (p_anom < 0.08), 0.0, p_anom)
    return p_anom, bg_scale


def _permute_columns_with_proj(rng, x_raw, x_imp, indices, proj, perm=None):
    """Permute each column independently over `indices` and append sparse projections."""
    m = len(indices)
    d = x_raw.shape[1]
    if perm is None:
        perm = np.column_stack([rng.permutation(m) for _ in range(d)])
    selected = np.column_stack([indices[perm[:, j]] for j in range(d)])
    cols = np.arange(d)
    xr_n = x_raw[selected, cols]
    xi_n = x_imp[selected, cols]
    return np.hstack([xr_n, xi_n @ proj])


def calibrated_shift_score(shift_mag, n_rep=8, seed=342):
    """Two-pass calibrated top-k hinged shift energy across temporal families."""
    n, d = shift_mag.shape
    if n == 0 or d == 0:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), 1.0
    m = shift_mag.to_numpy(dtype=float)

    def energy(mat):
        hinge = np.clip(mat - 2.1, 0.0, 12.0) ** 2
        k = min(6, mat.shape[1])
        top = np.sort(hinge, axis=1)[:, ::-1][:, :k]
        return np.log1p(top.sum(axis=1))

    e_real = energy(m)
    if n < 60:
        return e_real, np.tanh(e_real / 6.0), 1.0
    rng = np.random.default_rng(seed)
    null1 = np.concatenate([energy(np.column_stack([m[rng.permutation(n), j] for j in range(d)])) for _ in range(4)])
    p1, _ = calibrate_vs_clean_null(e_real, null1)
    bg_rows = m[p1 < 0.65] if int((p1 < 0.65).sum()) >= 50 else m
    m_bg = len(bg_rows)
    null2 = np.concatenate([energy(np.column_stack([bg_rows[rng.permutation(m_bg), j] for j in range(d)])) for _ in range(n_rep)])
    p_sh, bg_scale = calibrate_vs_clean_null(e_real, null2)
    return e_real, p_sh, bg_scale


def residual_density_ratio(features_df, bg_mask, cfg, seed=342):
    """Fit bagged OOF HGB density-ratio estimator with sparse random linear projections.

    Breaks depth-0 axis-aligned symmetry against column-permuted marginals while matching
    held-out out-of-fold background rows and permutations identically across bags.
    """
    n = len(features_df)
    d = features_df.shape[1]
    ref_idx = np.where(np.asarray(bg_mask, dtype=bool))[0]
    m_all = len(ref_idx)
    n_splits = cfg['n_splits']
    if d < 2 or m_all < max(400, n_splits * 20):
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), 1.0

    rng = np.random.default_rng(seed)
    x_raw = features_df.to_numpy(dtype=np.float32)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(x_raw[ref_idx])
    x_raw_imp = imp.transform(x_raw).astype(np.float32)
    scaler = StandardScaler().fit(x_raw_imp[ref_idx])
    x_imp = np.clip(scaler.transform(x_raw_imp), -4.0, 4.0).astype(np.float32)

    n_bags = cfg['n_bags']
    n_proj = cfg['n_proj']
    n_rep = cfg['n_rep']
    n_eval_null = cfg['n_eval_null']

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(x_raw))
    fold_eval_perms = []
    for _, va_idx in splits:
        va_ref_local = np.where(np.isin(ref_idx, va_idx))[0]
        va_ref_rows = ref_idx[va_ref_local]
        mv = len(va_ref_rows)
        perms = [np.column_stack([rng.permutation(mv) for _ in range(d)]) for _ in range(n_eval_null)]
        fold_eval_perms.append((va_ref_local, va_ref_rows, perms))

    oof_real_sum = np.zeros(n, dtype=float)
    oof_null_sum = np.zeros((n_eval_null, m_all), dtype=float)

    for bag in range(n_bags):
        rng_b = np.random.default_rng(seed + bag * 101)
        proj = np.zeros((d, n_proj), dtype=np.float32)
        for j in range(n_proj):
            k_nz = min(d, 2 if j < int(0.6 * n_proj) else 3)
            cols_j = rng_b.choice(d, size=k_nz, replace=False)
            signs = rng_b.choice([-1.0, 1.0], size=k_nz)
            proj[cols_j, j] = signs / np.sqrt(k_nz)
        x_full = np.hstack([x_raw, x_imp @ proj])

        bag_real = np.zeros(n, dtype=float)
        bag_null = np.zeros((n_eval_null, m_all), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr_sub = tr_idx[bg_mask[tr_idx]]
            mb = len(tr_sub)
            null_parts = [_permute_columns_with_proj(rng_b, x_raw, x_imp, tr_sub, proj) for _ in range(n_rep)]
            x_tr = np.vstack([x_full[tr_sub]] + null_parts)
            y_tr = np.concatenate([np.ones(mb), np.zeros(mb * n_rep)])
            w_tr = np.concatenate([np.full(mb, float(n_rep)), np.ones(mb * n_rep)])
            clf = HistGradientBoostingClassifier(
                max_iter=cfg['max_iter'],
                max_depth=5,
                min_samples_leaf=min(25, max(5, mb // 40)),
                l2_regularization=2.0,
                learning_rate=0.07,
                max_bins=cfg['max_bins'],
                random_state=seed + bag * 100 + fold,
                early_stopping=False,
            ).fit(x_tr, y_tr, sample_weight=w_tr)
            p_va = np.clip(clf.predict_proba(x_full[va_idx])[:, 1], 1e-5, 1.0 - 1e-5)
            bag_real[va_idx] = np.log(p_va / (1.0 - p_va))

            va_ref_local, va_ref_rows, perms = fold_eval_perms[fold]
            if len(va_ref_rows) > 0:
                for r in range(n_eval_null):
                    x_nv = _permute_columns_with_proj(rng_b, x_raw, x_imp, va_ref_rows, proj, perm=perms[r])
                    pn = np.clip(clf.predict_proba(x_nv)[:, 1], 1e-5, 1.0 - 1e-5)
                    bag_null[r, va_ref_local] = np.log(pn / (1.0 - pn))
        oof_real_sum += bag_real / n_bags
        oof_null_sum += bag_null / n_bags

    p_anom, bg_scale = calibrate_vs_clean_null(oof_real_sum, oof_null_sum.ravel())
    return oof_real_sum, p_anom, bg_scale


def select_expected_macro_f3(p_anom):
    """Select prediction mask maximizing Expected Macro-F3 from calibrated posteriors."""
    p = np.asarray(p_anom, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=int), 0, 0.0, 0.0
    p_hat = float(p.sum())
    if p_hat < 0.002 * n:
        return np.zeros(n, dtype=int), 0, float(p_hat / n), 0.5
    order = np.argsort(-p, kind='stable')
    ps = p[order]
    p_eff = max(p_hat, 1.0)
    cum_tp = np.cumsum(ps)
    ks = np.arange(1, n + 1)
    f1 = 10.0 * cum_tp / (9.0 * p_eff + ks)
    tn = (n - p_eff) - (ks - cum_tp)
    f0 = 10.0 * tn / (9.0 * (n - p_eff) + (n - ks))
    ef3 = 0.5 * (f1 + f0)
    max_k = max(1, int(0.45 * n))
    valid_k = ks <= max_k
    best_idx = int(np.argmax(np.where(valid_k, ef3, -1.0)))
    k_opt = int(ks[best_idx])
    pred = np.zeros(n, dtype=int)
    pred[order[:k_opt]] = 1
    return pred, k_opt, float(p_hat / n), float(ef3[best_idx])


def score_anomalies(features_df, seed=342, profile='balanced'):
    """Run the 3-view unsupervised pipeline and return scores plus view diagnostics."""
    cfg = _PROFILE_CONFIG[profile]
    levels, shift_mag = extract_multiview_features(features_df)
    s_island = topological_island_score(levels)
    p_island = np.clip(1.0 - np.exp(-0.4 * s_island), 0.0, 1.0)
    e_shift, p_shift, bg_sh = calibrated_shift_score(shift_mag, seed=seed)
    bg_mask = (p_shift < 0.65) & (s_island == 0.0)
    lo_density, p_density, bg_h2 = residual_density_ratio(levels, bg_mask, cfg, seed=seed + 100)

    p_joint = 1.0 - (1.0 - p_shift) * (1.0 - p_island) * (1.0 - p_density)
    aux_shift = np.clip(e_shift / 10.0, 0.0, 1.0)
    aux_dens = 1.0 / (1.0 + np.exp(-np.clip(lo_density, -20.0, 20.0)))
    aux_isl = np.clip(s_island / 30.0, 0.0, 1.0)
    aux = (aux_shift + aux_dens + aux_isl) / 3.0
    score = np.where(
        p_joint >= 0.9999,
        0.9999 + 1e-5 * aux,
        np.where(p_joint == 0.0, 1e-5 * aux, p_joint + 1e-7 * aux),
    )
    return {
        'anomaly_score': score,
        'anomaly_probability': p_joint,
        'shift_prob': p_shift,
        'island_prob': p_island,
        'density_ratio_prob': p_density,
        'shift_energy': e_shift,
        'island_raw': s_island,
        'density_ratio_logit': lo_density,
        'bg_shift_null_scale': float(bg_sh),
        'bg_density_null_scale': float(bg_h2),
        'level_features': list(levels.columns),
        'shift_features': list(shift_mag.columns),
    }


def _validate_contamination(contamination):
    """Validate optional contamination override; accept 'auto' or None."""
    if contamination is None or (isinstance(contamination, str) and contamination.strip().lower() == 'auto'):
        return 'auto'
    rate = float(contamination)
    if not 0.0 < rate < 0.5:
        raise ValueError('contamination must be "auto" or between 0 and 0.5')
    return rate


def train_anomaly(data_root, output_dir, seed=342, profile='balanced', contamination='auto', threads=-1):
    """Fit Task 5 unsupervised multi-view anomaly detector and save predictions."""
    contam_mode = _validate_contamination(contamination)
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / '05_anomaly_detection' / 'test.csv')
    if not frame['player_id'].is_unique:
        raise ValueError('Task 5 player_id must be unique')
    x = frame.drop(columns='player_id').apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    if 'is_anomaly' in x:
        raise ValueError('Task 5 must remain unsupervised; found label column')

    LOG.info('Task 5 scoring multi-view density ratios (%s profile, rows=%d)', profile, len(x))
    views = score_anomalies(x, seed=seed, profile=profile)
    p_joint = views['anomaly_probability']
    score = views['anomaly_score']
    auto_pred, k_opt, p_est, exp_mf3 = select_expected_macro_f3(p_joint)

    if contam_mode == 'auto':
        prediction = auto_pred
        mode_name = 'auto_expected_macro_f3'
    else:
        order = np.argsort(-score, kind='stable')
        prediction = np.zeros(len(frame), dtype=int)
        prediction[order[:int(np.ceil(contam_mode * len(frame)))]] = 1
        mode_name = 'fixed_fraction'

    pd.DataFrame({'player_id': frame.player_id, 'prediction': prediction}).to_csv(out / 'task5_predictions.csv', index=False)
    shift_ref = shift_score(x) if _complete_families(x) else np.zeros(len(frame), dtype=float)
    scored = pd.DataFrame({
        'player_id': frame.player_id,
        'anomaly_score': score,
        'anomaly_probability': p_joint,
        'prediction': prediction,
        'shift_prob': views['shift_prob'],
        'island_prob': views['island_prob'],
        'density_ratio_prob': views['density_ratio_prob'],
        'shift_energy': views['shift_energy'],
        'island_raw': views['island_raw'],
        'density_ratio_logit': views['density_ratio_logit'],
        'shift_rank': shift_ref,
    })
    scored.to_csv(out / 'task5_scores.csv', index=False)
    artifacts = {
        'profile': profile,
        'seed': seed,
        'config': _PROFILE_CONFIG[profile],
        'level_features': views['level_features'],
        'shift_features': views['shift_features'],
        'bg_shift_null_scale': views['bg_shift_null_scale'],
        'bg_density_null_scale': views['bg_density_null_scale'],
        'estimated_prevalence': p_est,
        'auto_optimal_k': k_opt,
    }
    joblib.dump(artifacts, out / 'task5_models.joblib')

    metrics = {
        'task': 5,
        'model': 'Multi-view Empirical Bayes density-ratio ensemble (Topological Islands + Calibrated Shifts + Residual HGB+Proj)',
        'seed': seed,
        'profile': profile,
        'threads': threads,
        'n_test': len(frame),
        'contamination_mode': mode_name,
        'contamination_assumption': float(prediction.mean()) if contam_mode == 'auto' else float(contam_mode),
        'estimated_prevalence': p_est,
        'estimated_expected_macro_f3': exp_mf3,
        'flagged_rows': int(prediction.sum()),
        'macro_f3': None,
        'metric_note': 'No ground-truth labels; local F3 cannot be calculated.',
        'fit_policy': 'Task5-specific transductive unsupervised fit on provided unlabeled test.csv.',
        'view_summary': {
            'shift_prob_gt_50': int((views['shift_prob'] > 0.5).sum()),
            'island_prob_gt_0': int((views['island_prob'] > 0.0).sum()),
            'density_ratio_prob_gt_20': int((views['density_ratio_prob'] > 0.2).sum()),
            'bg_shift_null_scale': views['bg_shift_null_scale'],
            'bg_density_null_scale': views['bg_density_null_scale'],
        },
        'sensitivity_flagged_rows': {str(q): int(np.ceil(q * len(frame))) for q in (.05, .1, .15, .2, .3)},
    }
    write_json(out / 'task5_metrics.json', metrics)
    return metrics
