"""Task 3 V7: V4 with better-calibrated low amount quantiles (lower alphas, coverage calibration, cap).

V4 predicts 0 if p ≤ 0.5, else the (1 − 0.5/p) quantile of the amount distribution. Two
things push that amount UP for uncertain spenders (p just above 0.5), which costs error
on the y = 0 rows they really are (bucket 0, 92.1M THB under V4):
  - V4's CatBoost MultiQuantile model is over-covered at low alphas (coverage 0.075 at
    α = 0.05, 0.127 at α = 0.10): its low quantiles are predicted too high;
  - quantile_combine clips q to ≥ 0.05, so rows with 0.5 < p < 0.526 (q < 0.05) get the
    5% quantile instead of something closer to 0.
Everything else is V4: classifier, features, folds, expected_spend leakage rule, target
log(y / prev) + prev_missing, t = 0.5, s = 1 and the champion rule. All of it is
imported from amount_model.py, not copied. Variants (all on ONE new amount model, the V4
MultiQuantile model with 14 alphas 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, …, 0.50):
  - V7-grid (diagnostic, not a candidate): new alpha grid, V4's rule (q ≥ 0.05, no remap,
    no cap): shows what the extra alphas alone do to the shared columns;
  - V7a: q clipped to ≥ 0.01 instead of 0.05;
  - V7b: coverage calibration (below), q still clipped to ≥ 0.05 before the remap;
  - V7 = V7a + V7b: q ≥ 0.01, then the remap.
V7a / V7b / V7 also clip the final prediction at the largest target in the outer-training
part (the target is capped at 300,000); the rows affected are reported.

Run from the repo root:
    python experiments/task3/model/quantile_calib/quantile_calib.py [--quick] [--promote]
Prints the summary, bucket, coverage and runtime tables; writes
experiments/task3/model/quantile_calib/quantile_calib_results.md (the hand-written
Interpretation section is kept), rows in experiments/task3/LOG.md and OOF/test
predictions in outputs/experiments/task3_<variant>/ (gitignored). --quick (250 trees per
model) only prints. --promote makes the chosen winner (see "Promotion") the champion.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold

# --- Make `pis`, ../amount_model and ../oracle importable from anywhere ---
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
from amount_model import (ALPHAS, AMOUNTS, CLASSIFIERS, COMPONENT_NAMES, MQ_LOSS, N_TEST, OUT,  # noqa: E402
                          TWO_STAGE, add_features, cb_params, current_branch, fold_expected_spend,
                          interp_quantile, load_champion, log_ratio, prev_or_one, promote, quantile_combine,
                          upsert_log, with_prev_flag)
from oracle import (BUCKETS, EXPECTED_FOLDS, EXPECTED_OOF, FOLDS, SEED, TARGET,  # noqa: E402
                    bucket_table, to_markdown)
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402

# --- Settings ---
CLF, AMT = TWO_STAGE['v4']  # V4's classifier ('clf') and its 10-alpha MultiQuantile amount model ('mq')
AMT_CAP = AMOUNTS[AMT][1]  # 3000 trees, early stopping as in V4
ALPHAS7 = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
MQ7_LOSS = 'MultiQuantile:alpha=' + ','.join(f'{a:g}' for a in ALPHAS7)
AMT7_NAME = 'amount log-ratio MultiQuantile, 14 alphas 0.01–0.50, 3000 iters'  # refit.py reads this name
CAL_FOLDS = 5  # inner KFold over the outer-training spenders for the calibration map
TIE = 0.0003  # winners closer than this in OOF count as equal: the simplest one is promoted
SIMPLICITY = ['v7a', 'v7b', 'v7']  # simplest first: V7a needs no extra fits, V7b / V7 need the inner fits
RESULTS = Path(__file__).with_name('quantile_calib_results.md')
INTERPRETATION = '## Interpretation'
SCRIPT = 'experiments/task3/model/quantile_calib/quantile_calib.py'
# variant -> q_min (lower clip of the nominal q), calibrated (remap q), cap (clip at the outer-train max).
VARIANTS = {'v7_grid': dict(q_min=0.05, calibrated=False, cap=False),
            'v7a': dict(q_min=0.01, calibrated=False, cap=True),
            'v7b': dict(q_min=0.05, calibrated=True, cap=True),
            'v7': dict(q_min=0.01, calibrated=True, cap=True)}
NAMES = {'v7_grid': 'V7-grid = V4 rule on the 14-alpha model (diagnostic)',
         'v7a': 'V7a = lower alphas, q ≥ 0.01, cap',
         'v7b': 'V7b = coverage-calibrated q (q ≥ 0.05 before remap), cap',
         'v7': 'V7 = V7a + V7b (q ≥ 0.01, remap, cap)'}
SHORT = {'v7_grid': 'V7-grid', 'v7a': 'V7a', 'v7b': 'V7b', 'v7': 'V7'}
LOG_LABELS = {'v7_grid': 'V7-grid (diagnostic): V4 rule (q ≥ 0.05, no remap, no cap) on a MultiQuantile model with 14 alphas 0.01–0.50',
              'v7a': 'V7a: V4 + 14-alpha MultiQuantile (0.01–0.50), q clipped to ≥ 0.01, prediction capped at outer-train max',
              'v7b': 'V7b: V7a model + in-fold coverage calibration of q (inner KFold(5) OOF), q ≥ 0.05 before remap, cap',
              'v7': 'V7: V7a + V7b (q ≥ 0.01, coverage remap, cap)'}


# --- Coverage calibration (V7b) ---
# Coverage of a quantile column = share of spenders whose true log(y / prev) is ≤ it; a
# calibrated alpha-quantile has coverage alpha. On held-out rows the curve alpha -> coverage
# is monotone (rows are sorted across alphas), so it can be inverted: for a wanted level q,
# alpha' = the modelled level whose coverage is q (linear interpolation between grid points),
# and the model is read at alpha' instead of q. That is a monotone remap of q before the
# row-wise interpolation in quantile_combine. Levels outside the measured coverage range
# are clamped to the grid ends (0.01 / 0.50).
#
# No leakage: the curve is measured on inner out-of-fold predictions of the OUTER-training
# spenders only. For each outer fold, the outer-training spenders are split with
# KFold(5); the 14-alpha amount model is refitted on inner-train exactly like the outer
# one (same features, expected_spend rebuilt inside the inner split with the same rule:
# inner-OOF for inner-train rows, inner-train multipliers for inner-val rows; same seed,
# 3000 trees, early stopping on inner-val), and predicts inner-val. Outer-validation rows
# never enter the map. Test predictions (averaged over the 3 fold models) use the average
# of the 3 fold maps, itself monotone.

def coverage(truth, log_q):
    """Share of rows whose truth is ≤ each row-sorted quantile column."""
    return (truth[:, None] <= np.sort(log_q, axis=1)).mean(axis=0)


def make_map(cov, alphas=ALPHAS7):
    """Monotone q -> alpha' with coverage(alpha') = q, from the coverage curve `cov` over `alphas`."""
    c = np.maximum.accumulate(np.asarray(cov, dtype=float)) + np.arange(len(cov)) * 1e-9  # strictly increasing
    return lambda q: np.interp(q, c, alphas)


def average_map(maps):
    return lambda q: np.mean([m(q) for m in maps], axis=0)


def fit_amount(loss, iterations, seed_fold, x_fit, y_fit, categorical, eval_set=None):
    """V4's amount model (CatBoost, log-ratio target, cb_params) with the given MultiQuantile loss."""
    reg = CatBoostRegressor(loss_function=loss, **cb_params(iterations, seed_fold))
    reg.fit(x_fit, y_fit, cat_features=categorical, eval_set=eval_set, use_best_model=eval_set is not None)
    return reg


def calibration_curve(raw, x, y, categorical, seed_fold, iterations):
    """Inner-OOF coverage curve of the 14-alpha amount model on the spender rows `raw` / `x` / `y`.

    `raw` and `x` are positionally aligned frames of ONE training part's spenders (the outer-training
    part in CV, the full train in a refit). Returns (coverage per alpha, inner-OOF quantiles, best
    iteration per inner fit).
    """
    inner = np.zeros((len(raw), len(ALPHAS7)))
    best = []
    for itr, iva in KFold(n_splits=CAL_FOLDS, shuffle=True, random_state=SEED).split(raw):
        f_raw, v_raw = raw.iloc[itr], raw.iloc[iva]
        es_f, (es_v,) = fold_expected_spend(f_raw, [v_raw])
        xf = with_prev_flag(add_features(x.iloc[itr], f_raw, es_f), f_raw)
        xv = with_prev_flag(add_features(x.iloc[iva], v_raw, es_v), v_raw)
        reg = fit_amount(MQ7_LOSS, iterations, seed_fold, xf, log_ratio(y[itr], f_raw), categorical,
                         eval_set=(xv, log_ratio(y[iva], v_raw)))
        inner[iva] = reg.predict(xv)
        best.append(reg.get_best_iteration() + 1)
    return coverage(log_ratio(y, raw), inner), inner, best


def predict_variant(v, p, log_q, prev, q_map, cap):
    """Final prediction of variant v from p and 14-alpha log-ratio quantiles; also the uncapped one."""
    cfg = VARIANTS[v]
    raw = quantile_combine(p, log_q, prev, alphas=ALPHAS7, q_min=cfg['q_min'],
                           q_map=q_map if cfg['calibrated'] else None)
    return (np.minimum(raw, cap) if cfg['cap'] else raw), raw


def fmt4(v):
    return f'{v:.4f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--quick', action='store_true', help='250 trees per model; prints only (smoke test)')
    parser.add_argument('--promote', action='store_true', help='promote the chosen winner if one beats V4 on all folds')
    args = parser.parse_args()
    if args.promote and args.quick:
        parser.error('--promote needs a full run, not --quick')
    cap_it = lambda full: 250 if args.quick else full  # noqa: E731
    t_start = time.time()

    # --- Data, folds, baseline and champion (same checks as amount_model.py) ---
    train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)
    x, xt, categorical = _prepare(train, test, TARGET)
    y = train[TARGET].to_numpy(dtype=float)
    spend = y > 0
    oof_base = pd.read_csv(ROOT / 'baseline' / 'task3_oof.csv')
    assert (train['player_id'].to_numpy() == oof_base['player_id'].to_numpy()).all(), 'row order differs'
    base = oof_base['prediction'].to_numpy(dtype=float)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(x))
    base_folds = [_score(y[va], base[va]) for _, va in folds]
    if abs(_score(y, base) - EXPECTED_OOF) > 1e-4 or any(abs(a - b) > 1e-4 for a, b in zip(base_folds, EXPECTED_FOLDS)):
        sys.exit(f'Baseline OOF does not reproduce the published fold scores: {base_folds}')
    champ_meta, champ = load_champion(train, base)
    if champ_meta.get('variant') != 'v4':
        sys.exit(f'champion is {champ_meta.get("variant")}; this experiment is defined against V4')
    champ_folds = [_score(y[va], champ[va]) for _, va in folds]

    n, nt = len(train), len(test)
    p_oof, p_tst = np.zeros(n), np.zeros(nt)
    q4_oof, q4_tst = np.zeros((n, len(ALPHAS))), np.zeros((nt, len(ALPHAS)))
    q7_oof, q7_tst = np.zeros((n, len(ALPHAS7))), np.zeros((nt, len(ALPHAS7)))
    best = {'clf': [], 'v4': [], 'v7': [], 'inner': []}
    seconds = {'features': [], 'classifier': [], 'V4 MultiQuantile (10 alphas)': [],
               'V7 MultiQuantile (14 alphas)': [], f'calibration: {CAL_FOLDS} inner fits': []}
    curves, inner_cov, caps = [], [], []

    # --- Outer CV: V4's classifier and amount model exactly as in amount_model.py, the
    # 14-alpha amount model next to it, then the inner fits for the calibration map ---
    for fold, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        fit_raw, val_raw = train.iloc[tr], train.iloc[va]
        es_fit, (es_val, es_test) = fold_expected_spend(fit_raw, [val_raw, test])
        x_fit = add_features(x.iloc[tr], fit_raw, es_fit)
        x_val = add_features(x.iloc[va], val_raw, es_val)
        x_tst = add_features(xt, test, es_test)
        caps.append(float(y[tr].max()))
        seconds['features'].append(time.time() - t0)

        t0 = time.time()
        clf = CatBoostClassifier(loss_function='Logloss', **cb_params(cap_it(1000), fold, **CLASSIFIERS[CLF]))
        clf.fit(x_fit, spend[tr].astype(int), cat_features=categorical,
                eval_set=(x_val, spend[va].astype(int)), use_best_model=True)
        p_oof[va] = clf.predict_proba(x_val)[:, 1]
        p_tst += clf.predict_proba(x_tst)[:, 1] / FOLDS
        best['clf'].append(clf.get_best_iteration() + 1)
        seconds['classifier'].append(time.time() - t0)

        s_fit, s_val = spend[tr], spend[va]
        f3, v3, t3 = with_prev_flag(x_fit, fit_raw), with_prev_flag(x_val, val_raw), with_prev_flag(x_tst, test)
        y3_fit, y3_val = log_ratio(y[tr][s_fit], fit_raw[s_fit]), log_ratio(y[va][s_val], val_raw[s_val])
        for key, loss, q_oof, q_tst, label in [('v4', MQ_LOSS, q4_oof, q4_tst, 'V4 MultiQuantile (10 alphas)'),
                                               ('v7', MQ7_LOSS, q7_oof, q7_tst, 'V7 MultiQuantile (14 alphas)')]:
            t0 = time.time()
            reg = fit_amount(loss, cap_it(AMT_CAP), fold, f3[s_fit], y3_fit, categorical, eval_set=(v3[s_val], y3_val))
            q_oof[va] = reg.predict(v3)
            q_tst += reg.predict(t3) / FOLDS
            best[key].append(reg.get_best_iteration() + 1)
            seconds[label].append(time.time() - t0)

        # Calibration map from the outer-training spenders only (positional subsets of train / x).
        t0 = time.time()
        idx = tr[s_fit]
        cov, _, inner_best = calibration_curve(train.iloc[idx], x.iloc[idx], y[idx], categorical, fold, cap_it(AMT_CAP))
        curves.append(cov)
        best['inner'].append(inner_best)
        seconds[f'calibration: {CAL_FOLDS} inner fits'].append(time.time() - t0)
        print(f'fold {fold}/{FOLDS}: ' + ', '.join(f'{k} {v[-1]:.0f}s' for k, v in seconds.items()), flush=True)

    prev_tr, prev_te = prev_or_one(train), prev_or_one(test)
    maps = [make_map(c) for c in curves]
    test_map = average_map(maps)
    fold_of = np.zeros(n, dtype=int)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f

    # --- V4 re-run must reproduce the champion (champion OOF is rounded to satang) ---
    v4_oof = quantile_combine(p_oof, q4_oof, prev_tr)
    v4_folds = [_score(y[va], v4_oof[va]) for _, va in folds]
    v4_ok = all(abs(a - b) <= 5e-5 for a, b in zip(v4_folds, champ_folds))
    print(f'V4 re-run: {" / ".join(map(fmt4, v4_folds))} vs champion {" / ".join(map(fmt4, champ_folds))} '
          f'-> {"reproduced" if v4_ok else "NOT reproduced"}', flush=True)
    if not v4_ok and not args.quick:
        sys.exit('V4 re-run does not reproduce the champion; comparisons would not be valid')

    # --- Variants: OOF per fold (that fold's map and cap), test with the averaged map ---
    test_cap = float(y.max())  # test predictions average all 3 fold models; the full train's max
    final = {}
    for v in VARIANTS:
        oof, oof_raw = np.zeros(n), np.zeros(n)
        for f, (_, va) in enumerate(folds):
            oof[va], oof_raw[va] = predict_variant(v, p_oof[va], q7_oof[va], prev_tr[va], maps[f], caps[f])
        tst, tst_raw = predict_variant(v, p_tst, q7_tst, prev_te, test_map, test_cap)
        final[v] = dict(label=NAMES[v], oof=oof, oof_raw=oof_raw, test=tst, test_raw=tst_raw)

    # --- Summary: baseline, champion V4, then each variant vs both, fold by fold ---
    def row(label, pred):
        fs = [_score(y[va], pred[va]) for _, va in folds]
        s = _score(y, pred)
        return [label, *fs, s, s - _score(y, base), all(a > b for a, b in zip(fs, base_folds)),
                s - _score(y, champ), all(a > b for a, b in zip(fs, champ_folds)),
                ' / '.join(f'{a - b:+.5f}' for a, b in zip(fs, champ_folds))]

    rows = [row('Baseline (baseline/task3_oof.csv)', base), row(f'Champion: {champ_meta["name"]}', champ)]
    rows += [row(f['label'], f['oof']) for f in final.values()]
    cols = ['variant', 'fold1', 'fold2', 'fold3', 'OOF', 'Δ vs baseline', 'wins all folds vs baseline?',
            'Δ vs champion', 'wins all folds vs champion?', 'Δ per fold vs champion']
    summary = pd.DataFrame(rows, columns=cols)
    shown = summary.copy()
    for c in ['fold1', 'fold2', 'fold3', 'OOF']:
        shown[c] = shown[c].map(fmt4)
    for c in ['Δ vs baseline', 'Δ vs champion']:
        shown[c] = shown[c].map('{:+.4f}'.format)
    for c in ['wins all folds vs baseline?', 'wins all folds vs champion?']:
        shown[c] = shown[c].map({True: 'yes', False: 'no'})
    shown.iloc[0, 6] = '—'
    shown.iloc[1, [8, 9]] = '—'
    stats = {v: summary.iloc[2 + i] for i, v in enumerate(final)}

    # --- Cap: rows affected and what it changes ---
    cap_rows = []
    for v, f in final.items():
        if not VARIANTS[v]['cap']:
            continue
        hit_oof, hit_tst = int((f['oof_raw'] > f['oof']).sum()), int((f['test_raw'] > f['test']).sum())
        cap_rows.append([SHORT[v], f'{hit_oof:,}', f'{int((f["oof_raw"][y > 0] > f["oof"][y > 0]).sum()):,}',
                         f'{hit_tst:,}', fmt4(_score(y, f['oof_raw'])), fmt4(_score(y, f['oof'])),
                         f'{_score(y, f["oof"]) - _score(y, f["oof_raw"]):+.6f}',
                         f'{(np.abs(y - f["oof_raw"]).sum() - np.abs(y - f["oof"]).sum()) / 1e6:+.3f}'])
    cap_table = pd.DataFrame(cap_rows, columns=['variant', 'OOF rows capped', 'of which y > 0', 'test rows capped',
                                                'OOF uncapped', 'OOF capped', 'Δ score from cap',
                                                'error removed (M THB)'])

    # --- Error by bucket vs V4 ---
    errs = {'V4': bucket_table(y, champ)['error_sum']}
    errs.update({SHORT[v]: bucket_table(y, f['oof'])['error_sum'] for v, f in final.items()})
    tot = lambda e: [*e, e.sum()]  # noqa: E731
    bucket_cmp = pd.DataFrame({'bucket': [*BUCKETS, 'total'],
                               'count': [*bucket_table(y, base)['count'].map('{:,}'.format), f'{n:,}']})
    for lbl, e in errs.items():
        bucket_cmp[f'{lbl} (M THB)'] = [f'{v / 1e6:.2f}' for v in tot(e)]
    for lbl in [SHORT[v] for v in final]:
        bucket_cmp[f'{lbl} − V4 (M)'] = [f'{(b - a) / 1e6:+.2f}' for a, b in zip(tot(errs['V4']), tot(errs[lbl]))]

    # --- Coverage before / after calibration (outer OOF spenders) ---
    # "after": each OOF row reads its quantile at its own fold's alpha' = map(alpha).
    truth = log_ratio(y[spend], train[spend])
    q7_sorted = np.sort(q7_oof[spend], axis=1)
    cov_v4 = coverage(truth, q4_oof[spend])
    cov_v7 = coverage(truth, q7_oof[spend])
    fold_sp = fold_of[spend]
    cov_cal = []
    for a in ALPHAS7:
        levels = np.array([maps[f](a) for f in range(FOLDS)])[fold_sp]
        cov_cal.append((truth <= interp_quantile(q7_sorted, np.clip(levels, ALPHAS7[0], ALPHAS7[-1]), ALPHAS7)).mean())
    v4_cov_at = {round(float(a), 3): c for a, c in zip(ALPHAS, cov_v4)}
    cov_table = pd.DataFrame({
        'alpha': [f'{a:g}' for a in ALPHAS7],
        'V4 model (10 alphas)': [f'{v4_cov_at[round(float(a), 3)]:.3f}' if round(float(a), 3) in v4_cov_at else '—'
                                 for a in ALPHAS7],
        'V7 model, raw': [f'{c:.3f}' for c in cov_v7],
        'V7 model, calibrated': [f'{c:.3f}' for c in cov_cal],
        'inner-OOF raw (mean of folds)': [f'{c:.3f}' for c in np.mean(curves, axis=0)],
        **{f"alpha' fold {f + 1}": [f'{maps[f](a):.4f}' for a in ALPHAS7] for f in range(FOLDS)}})

    # --- Which rows the changes touch ---
    q_nom = 1.0 - 0.5 / np.maximum(p_oof, 1e-12)
    pos = p_oof > 0.5
    low = pos & (q_nom < 0.05)
    reach = pd.DataFrame([
        ['OOF rows with p > 0.5 (get an amount)', f'{pos.sum():,}', f'{(~spend[pos]).mean():.3f}'],
        ['… of which 0.5 < p < 0.526 (q < 0.05: V4 / V7b clip them to 0.05)', f'{low.sum():,}',
         f'{(~spend[low]).mean():.3f}'],
        ['… of which 0.526 ≤ p < 0.6', f'{(pos & (p_oof < 0.6) & ~low).sum():,}',
         f'{(~spend[pos & (p_oof < 0.6) & ~low]).mean():.3f}']],
        columns=['rows', 'count', 'share with y = 0'])

    # --- Amount-model diagnostics ---
    med4, med7 = list(ALPHAS).index(0.5), int(np.flatnonzero(np.isclose(ALPHAS7, 0.5))[0])
    sp = lambda lq, j: _score(y[spend], (prev_tr * np.exp(lq[:, j]))[spend])  # noqa: E731
    crossing = lambda lq: (np.diff(lq, axis=1) < 0).any(axis=1).mean()  # noqa: E731
    diag = pd.DataFrame([['V4 MultiQuantile (10 alphas)', fmt4(sp(q4_oof, med4)), f'{crossing(q4_oof):.2%}'],
                         ['V7 MultiQuantile (14 alphas)', fmt4(sp(q7_oof, med7)), f'{crossing(q7_oof):.2%}']],
                        columns=['amount model', 'OOF score on y > 0 rows, α=0.5 column', 'rows with crossing quantiles'])
    iters = pd.DataFrame([[COMPONENT_NAMES[CLF], cap_it(1000), ' / '.join(map(str, best['clf']))],
                          [COMPONENT_NAMES[AMT], cap_it(AMT_CAP), ' / '.join(map(str, best['v4']))],
                          [AMT7_NAME, cap_it(AMT_CAP), ' / '.join(map(str, best['v7']))]],
                         columns=['model', 'limit', 'best iteration per fold'])
    inner_iters = pd.DataFrame([[f'outer fold {f + 1}', ' / '.join(map(str, b))] for f, b in enumerate(best['inner'])],
                               columns=['calibration fits (14-alpha model)', f'best iteration per inner fold ({CAL_FOLDS})'])
    times = pd.DataFrame([[comp, *[f'{s:.0f}' for s in secs], f'{sum(secs):.0f}'] for comp, secs in seconds.items()],
                         columns=['step', 'fold1 (s)', 'fold2 (s)', 'fold3 (s)', 'total (s)'])
    total = time.time() - t_start

    print(f'\nTASK 3 V7 QUANTILE CALIBRATION{" (QUICK: 250 trees)" if args.quick else ""}\n{shown.to_string(index=False)}')
    print(f'\nCap at outer-train max (fold caps {", ".join(f"{c:,.0f}" for c in caps)}; test {test_cap:,.0f})\n'
          f'{cap_table.to_string(index=False)}')
    print(f'\nError by bucket vs V4\n{bucket_cmp.to_string(index=False)}')
    print(f'\nCoverage, y > 0 OOF rows\n{cov_table.to_string(index=False)}')
    print(f'\nRows reached\n{reach.to_string(index=False)}')
    print(f'\nAmount models\n{diag.to_string(index=False)}')
    print(f'\nBest iterations\n{iters.to_string(index=False)}\n{inner_iters.to_string(index=False)}')
    print(f'\nRuntimes\n{times.to_string(index=False)}\nTotal {total:.0f}s')
    if args.quick:
        print('\n--quick: nothing written.')
        return

    # --- OOF and test predictions (outputs/ is gitignored) ---
    oof_frames = {}
    for v, f in final.items():
        folder = OUT / f'task3_{v}'
        folder.mkdir(parents=True, exist_ok=True)
        oof_frames[v] = pd.DataFrame({'player_id': train['player_id'], 'target': y, 'prediction': f['oof']})
        oof_frames[v].to_csv(folder / 'task3_oof.csv', index=False)
        frame = pd.DataFrame({'player_id': test['player_id'], 'prediction': f['test']})
        assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
        frame.to_csv(folder / 'task3_predictions.csv', index=False)

    # --- Promotion ---
    # Candidates: V7a, V7b, V7 (V7-grid is a diagnostic). Each must beat V4 on all 3 folds.
    # Among winners, the simplest one whose OOF is within TIE of the best winner is chosen.
    wins = {v: bool(stats[v]['wins all folds vs champion?']) for v in SIMPLICITY}
    winners = [v for v in SIMPLICITY if wins[v]]
    chosen = None
    if winners:
        top = max(stats[v]['OOF'] for v in winners)
        chosen = next(v for v in SIMPLICITY if v in winners and stats[v]['OOF'] >= top - TIE)
    verdict = {v: (f'{SHORT[v]} beats V4 on all 3 folds' if wins[v] else
                   f'{SHORT[v]} does not beat V4 on all 3 folds ({stats[v]["Δ per fold vs champion"]})')
               for v in SIMPLICITY}
    promoted = None
    if chosen and args.promote:
        r = stats[chosen]
        promote(chosen, NAMES[chosen], oof_frames[chosen], [r['fold1'], r['fold2'], r['fold3']], r['OOF'],
                champ_meta, script=SCRIPT)
        promoted = chosen
        decision = (f'PROMOTED: {SHORT[chosen]} (OOF {r["OOF"]:.4f}, {r["Δ vs champion"]:+.4f} vs V4; simplest winner '
                    f'within {TIE} of the best). ' + '; '.join(verdict.values()) + '.')
    elif chosen:
        decision = '; '.join(verdict.values()) + f'; choice: {SHORT[chosen]}; re-run with --promote to promote it.'
    else:
        decision = 'STOP: ' + '; '.join(verdict.values()) + '; champion stays V4, no refit, no Kaggle file.'
    print(f'\n{decision}')

    # --- quantile_calib_results.md (Interpretation kept) ---
    md = ['# Task 3 V7: calibrated low amount quantiles', '',
          '> **CAUTION:** Early stopping uses the validation fold (as in the baseline and V4) for every model, '
          'including the inner calibration fits (inner-val), so every OOF score here is slightly optimistic; '
          'differences between rows are still comparable.', '',
          f'Generated by `python {SCRIPT}{" --promote" if args.promote else ""}`. Folds `KFold({FOLDS}, shuffle=True, '
          f'random_state={SEED})`; metric `_score` from `pis/tabular.py`. Classifier, features, `expected_spend` rule, '
          'target `log(y / prev)` + `prev_missing`, t = 0.5 and `quantile_combine` are imported from '
          f'`amount_model.py` (V4 re-run here: {" / ".join(map(fmt4, v4_folds))}, reproduces the champion).', '',
          f'- **V7 amount model:** V4\'s CatBoost model with `{MQ7_LOSS}` (14 alphas instead of 10), same '
          f'settings (≤ {AMT_CAP} trees, early stopping on the validation spenders). All V7 rows use it.',
          '- **V7-grid (diagnostic):** V4\'s rule on the new model (q = 1 − 0.5/p clipped to [0.05, 0.5], no remap, '
          'no cap).',
          '- **V7a:** q clipped to [0.01, 0.5].',
          f'- **V7b:** q clipped to [0.05, 0.5], then remapped to alpha\' = the level whose inner-OOF coverage is q '
          f'(linear interpolation of the coverage curve, clamped to [0.01, 0.5]). The curve comes from inner '
          f'`KFold({CAL_FOLDS})` over the outer-training spenders (14-alpha model refitted on inner-train with '
          '`expected_spend` rebuilt inside the split, early stopping on inner-val); outer-validation rows never '
          'enter it. Test uses the mean of the 3 fold maps.',
          '- **V7:** V7a + V7b (q clipped to [0.01, 0.5], then remapped).',
          '- **Cap (V7a / V7b / V7):** final prediction clipped at the largest target of the outer-training part '
          f'(folds: {", ".join(f"{c:,.0f}" for c in caps)}; test: full-train max {test_cap:,.0f}).', '',
          'Promotion: V7a / V7b / V7 must each beat V4 on all 3 folds; among winners the simplest '
          f'(V7a < V7b < V7) within {TIE} OOF of the best is chosen.', '',
          f'**Decision: {decision}**', '',
          '## Summary', '', to_markdown(shown), '',
          '## Cap at the outer-train max', '', to_markdown(cap_table), '',
          '## Error by bucket vs V4', '', to_markdown(bucket_cmp), '',
          '## Quantile coverage before / after calibration', '',
          'Share of OOF spenders whose true `log(y / prev)` is ≤ the predicted (row-sorted) quantile; ≈ alpha if '
          'calibrated. "calibrated" reads each row at its fold\'s alpha\' (columns on the right: alpha\' = map(alpha) '
          'per outer fold). "inner-OOF raw" is the curve the maps were fitted on.', '',
          to_markdown(cov_table), '',
          '## Rows reached by the change', '', to_markdown(reach), '',
          '## Amount models', '',
          'Spenders-only score: the α=0.5 column (`prev × exp(...)`) on y > 0 rows (stage 2 as if stage 1 were perfect).',
          '', to_markdown(diag), '',
          '## Best iterations', '', to_markdown(iters), '',
          '### Calibration fits', '', to_markdown(inner_iters), '',
          '## Runtimes', '', to_markdown(times), '', f'Total {total:.0f}s.', '']
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')

    # --- ../LOG.md ---
    today, branch = datetime.date.today().isoformat(), current_branch()
    secs_cal = sum(seconds[f'calibration: {CAL_FOLDS} inner fits'])
    log_rows = []
    for v in VARIANTS:
        r = stats[v]
        notes = []
        if v == 'v7_grid':
            notes.append('diagnostic row, not a candidate')
        if VARIANTS[v]['cap']:
            notes.append(f'cap at outer-train max: {int((final[v]["oof_raw"] > final[v]["oof"]).sum())} OOF rows / '
                         f'{int((final[v]["test_raw"] > final[v]["test"]).sum())} test rows capped')
        notes.append(f'wins all folds vs baseline: {"yes" if r["wins all folds vs baseline?"] else "no"}')
        notes.append(f'vs champion (V4): {r["Δ vs champion"]:+.4f} ({r["Δ per fold vs champion"]}), '
                     f'wins all folds: {"yes" if r["wins all folds vs champion?"] else "no"}')
        if VARIANTS[v]['calibrated']:
            notes.append(f'calibration: {CAL_FOLDS} inner fits per fold, {secs_cal:.0f}s over 3 folds')
        if v == promoted:
            notes.append('**PROMOTED to champion**')
        elif v in SIMPLICITY and not winners:
            notes.append('STOP: no V7 variant beats V4 on all folds; champion stays V4, no refit, no Kaggle file')
        if v == 'v7':
            notes.append(f'`quantile_calib/`; run {total:.0f}s')
        log_rows.append([today, branch, LOG_LABELS[v], ' / '.join(fmt4(r[f'fold{i}']) for i in (1, 2, 3)),
                         fmt4(r['OOF']), '—', '; '.join(notes)])
    upsert_log(log_rows)
    print(f'Wrote {RESULTS.relative_to(ROOT)}; updated experiments/task3/LOG.md')


if __name__ == '__main__':
    main()
