"""Task 3 amount model: better "how much?" stage (V1 features, V2/V3 two-stage).

The oracle test showed the amount stage is the main bottleneck (perfect amount +0.148
vs perfect zeros +0.054), and the EDA showed spenders' y ≈ prev_month_spending × a
segment×VIP multiplier. This script turns that into a leak-free feature and tries it
in a single-stage model (V1) and in two-stage models (V2: amount in THB, V3: amount
as a log-ratio to prev_month_spending).

Run from the repo root:
    python experiments/task3/amount_model.py [--variants v1,v2,v3] [--quick]
Prints the summary, error-by-bucket and stage diagnostics; rewrites the tables in
experiments/task3/amount_model_results.md (the hand-written interpretation is kept);
upserts one row per variant in experiments/task3/LOG.md; writes test predictions to
outputs/experiments/task3_<variant>/task3_predictions.csv (gitignored).
"""
import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# --- Make `pis` and oracle.py importable no matter where the script is launched from ---
# Reuse the project's data path, preprocessing and metric, and the oracle's multiplier
# code, so the expected_spend feature is exactly the rule scored as scenario D there.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402
from oracle import (EXPECTED_FOLDS, EXPECTED_OOF, FOLDS, PREV, SEED, TARGET,  # noqa: E402
                    bucket_table, fit_multipliers, fmt_bucket, lookup, to_markdown)

# --- Settings ---
# Outer folds, seed and CatBoost settings match pis/tabular.py so every score is
# directly comparable with the baseline fold by fold. INNER_FOLDS only splits the
# outer-training part to build expected_spend for the training rows (see below).
INNER_FOLDS = 5
LEVELS = [['segment', 'vip_status'], ['segment'], []]  # fallback order: segment×VIP -> segment -> global
GRID = np.round(np.arange(0.30, 0.70 + 1e-9, 0.025), 3)  # thresholds tried for p(spend)
VARIANTS = ['v1', 'v2', 'v3']
N_TEST = 25_889
RESULTS = Path(__file__).with_name('amount_model_results.md')
LOG_FILE = Path(__file__).with_name('LOG.md')
OUT = ROOT / 'outputs' / 'experiments'
INTERPRETATION = '## Interpretation'


def cb_params(iterations, fold):
    """Baseline CatBoost settings (pis/tabular.py), seed shifted per fold like the baseline."""
    return dict(iterations=iterations, depth=6, learning_rate=0.05, random_seed=SEED + fold,
                thread_count=-1, verbose=False, allow_writing_files=False, early_stopping_rounds=80)


# --- Shared feature: expected_spend = prev_month_spending × group multiplier ---
# The multiplier is a target statistic (median of y / prev among spenders), so it must
# never be computed with the row's own target, otherwise the model learns to trust a
# feature that is far better on train than it can be on test:
#   - outer-validation and test rows: multipliers fitted on the whole outer-training part
#     (they never contribute to it), exactly like a real test-time lookup;
#   - outer-training rows: inner KFold(5) out-of-fold, so each row's value comes from
#     multipliers fitted on the other 4/5 of the outer-training part.
# Rows with no prev_month_spending keep NaN; CatBoost handles missing values natively.

def expected_spend(fit, apply):
    """prev × multiplier for the `apply` rows, multipliers fitted on the `fit` rows only."""
    mult = lookup(apply, [(keys, fit_multipliers(fit, keys)) for keys in LEVELS])
    return apply[PREV] * mult


def fold_expected_spend(fit, others):
    """expected_spend for one outer fold: inner-OOF for `fit` rows, full-fit for each frame in `others`."""
    inner = pd.Series(np.nan, index=fit.index)
    for itr, iva in KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED).split(fit):
        inner.iloc[iva] = expected_spend(fit.iloc[itr], fit.iloc[iva]).to_numpy()
    return inner, [expected_spend(fit, frame) for frame in others]


def add_features(x, raw, es):
    """Model features + the expected_spend family (all positional, aligned to `raw`)."""
    out = x.copy()
    out['expected_spend'] = es.to_numpy()
    # log1p versions do not change tree splits on their own, but CatBoost builds feature
    # combinations and borders per feature, so the log scale (where the prev -> y
    # relationship is linear) can still help; they are cheap to add.
    out['log_prev'] = np.log1p(raw[PREV]).to_numpy()
    out['log_expected_spend'] = np.log1p(es).to_numpy()
    # Last month relative to the long-run history: a recent spike or drop in spending.
    ratio = raw[PREV] / raw['historical_spending']
    out['prev_to_hist'] = ratio.replace([np.inf, -np.inf], np.nan).to_numpy()
    return out


# --- V3 target: log(y / prev) ---
# MAE is minimised by the conditional median, and the median commutes with any
# monotone transform: median(log(y / prev)) = log(median(y) / prev) for a given row.
# So exp() of an MAE prediction on the log scale is still a median of y in THB, which
# is what the L1-style metric wants. The earlier log1p + RMSE attempt (0.7716) fitted
# the conditional MEAN of log1p(y) on all rows; with 48% zeros that mean sits between
# log1p(0) = 0 and the spenders' level, and exp() of it is neither 0 nor the median.
#
# Missing prev (≈3k spenders): we predict log(y) instead (prev treated as 1) and add a
# `prev_missing` flag. Chosen over falling back to the V2 regressor because it keeps V3
# self-contained (`--variants v3` runs alone), trees separate the two sub-problems with
# one split on the flag, and both parts stay MAE-on-log, so the median argument holds.

def prev_or_one(raw):
    return raw[PREV].fillna(1.0).to_numpy()


def log_ratio(y, raw):
    return np.log(y) - np.log(prev_or_one(raw))


# --- Combine the two stages ---
# Under L1 the best constant for a row is its median. If p(spend) < 0.5 the median of
# the row's distribution is 0; if p > 0.5 it is a positive amount. So t = 0.5 is the
# theoretical rule (strictly, the median for p > 0.5 is the (1 - 0.5/p) quantile of the
# amount distribution, which we approximate with the conditional median). The OOF-tuned
# t is chosen and scored on the same OOF rows, so its score is optimistic.

def combine(p, amount, t):
    return np.where(p > t, np.maximum(0.0, amount), 0.0)


def tune_threshold(y, p, amount):
    scores = [_score(y, combine(p, amount, t)) for t in GRID]
    return float(GRID[int(np.argmax(scores))])


# --- Output helpers ---

def fmt4(v):
    return f'{v:.4f}'


def current_branch():
    try:
        return subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return '?'


LOG_HEADER = """# Task 3 experiment log

One row per experiment. Fold scores use `KFold(3, shuffle=True, random_state=342)` (same as the
baseline); metric is `_score` from `pis/tabular.py`. LB = Kaggle leaderboard (— = not submitted).

| date | branch | change | fold scores | OOF | LB | note |
|---|---|---|---|---|---|---|
| 2026-11-11 | main | log1p target + RMSE (CatBoost) | — | 0.7716 | — | from docs/KAGGLE_FINDINGS_TH.md; worse than baseline |
| 2026-11-13 | main | Baseline: CatBoost MAE, 1000 iters, depth 6, lr 0.05, ES 80 | 0.7873 / 0.7587 / 0.7807 | 0.7753 | — | baseline/task3_oof.csv; LB only as part of the full submission |
| ? | ? | Huber loss | ? | ? | — | TODO: numbers not recorded in the repo, fill in |
| 2026-11-15 | task3/oracle | Oracle B: true zeros + baseline amount | 0.8406 / 0.8135 / 0.8335 | 0.8290 | — | upper bound, NOT submittable |
| 2026-11-15 | task3/oracle | Oracle C: true zeros + prev × segment ratio | 0.7591 / 0.7556 / 0.7650 | 0.7599 | — | upper bound, NOT submittable |
| 2026-11-15 | task3/oracle | Oracle D: true zeros + prev × segment×VIP ratio | 0.7906 / 0.7855 / 0.7939 | 0.7900 | — | upper bound, NOT submittable |
| 2026-11-15 | task3/oracle | Oracle E: baseline zeros + true amount | 0.9255 / 0.9184 / 0.9250 | 0.9229 | — | upper bound, NOT submittable |
"""


def upsert_log(rows):
    """Add rows to LOG.md; a row with the same (branch, change) replaces the old one, so re-runs don't duplicate."""
    lines = (LOG_FILE.read_text(encoding='utf-8') if LOG_FILE.exists() else LOG_HEADER).rstrip('\n').split('\n')
    for row in rows:
        new = '| ' + ' | '.join(row) + ' |'
        key = (row[1], row[2])
        hit = [i for i, line in enumerate(lines)
               if line.startswith('| ') and tuple(c.strip() for c in line.strip('|').split('|')[1:3]) == key]
        if hit:
            lines[hit[0]] = new
        else:
            lines.append(new)
    LOG_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variants', default=','.join(VARIANTS), help='comma-separated subset of v1,v2,v3')
    parser.add_argument('--quick', action='store_true', help='250 iterations instead of 1000 (smoke test)')
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    if not variants or set(variants) - set(VARIANTS):
        parser.error(f'--variants must be a subset of {VARIANTS}')
    iterations = 250 if args.quick else 1000
    two_stage = 'v2' in variants or 'v3' in variants

    # --- Load data and check it lines up with the baseline OOF ---
    # Same row filter and preprocessing as pis/tabular.py. The baseline OOF has no fold
    # column, so we re-score it on our folds; if it does not reproduce the published
    # fold scores, the folds or row order are off and no comparison would be valid.
    train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)
    x, xt, categorical = _prepare(train, test, TARGET)
    y = train[TARGET].to_numpy(dtype=float)
    oof_base = pd.read_csv(ROOT / 'baseline' / 'task3_oof.csv')
    assert (train['player_id'].to_numpy() == oof_base['player_id'].to_numpy()).all(), 'row order differs from baseline OOF'
    base = oof_base['prediction'].to_numpy(dtype=float)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(x))
    base_folds = [_score(y[va], base[va]) for _, va in folds]
    if abs(_score(y, base) - EXPECTED_OOF) > 1e-4 or any(abs(a - b) > 1e-4 for a, b in zip(base_folds, EXPECTED_FOLDS)):
        sys.exit(f'Baseline OOF does not reproduce the published fold scores: {base_folds}')

    # --- Containers: OOF predictions and fold-averaged test predictions per stage ---
    n, nt = len(train), len(test)
    oof = {k: np.zeros(n) for k in ['v1', 'p', 'v2_amount', 'v3_amount']}
    tst = {k: np.zeros(nt) for k in oof}
    best_iter = {k: [] for k in ['v1', 'clf', 'v2_reg', 'v3_reg']}
    spend = y > 0

    # --- Outer CV ---
    # Early stopping uses the validation fold as eval_set with use_best_model=True, exactly
    # like the baseline, so the scores stay comparable. Known issue: picking the best
    # iteration on the rows we then score slightly inflates OOF (for every variant alike).
    for fold, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        fit_raw, val_raw = train.iloc[tr], train.iloc[va]
        es_fit, (es_val, es_test) = fold_expected_spend(fit_raw, [val_raw, test])
        x_fit = add_features(x.iloc[tr], fit_raw, es_fit)
        x_val = add_features(x.iloc[va], val_raw, es_val)
        x_tst = add_features(xt, test, es_test)
        y_fit, y_val = y[tr], y[va]
        params = cb_params(iterations, fold)

        if 'v1' in variants:
            # V1: baseline single-stage MAE model on all rows, plus the new features.
            model = CatBoostRegressor(loss_function='MAE', **params)
            model.fit(x_fit, y_fit, cat_features=categorical, eval_set=(x_val, y_val), use_best_model=True)
            oof['v1'][va] = np.maximum(0.0, model.predict(x_val))
            tst['v1'] += np.maximum(0.0, model.predict(x_tst)) / FOLDS
            best_iter['v1'].append(model.get_best_iteration() + 1)

        if two_stage:
            # Stage 1 (shared by V2 and V3): will they spend? Logloss on all rows.
            clf = CatBoostClassifier(loss_function='Logloss', **params)
            clf.fit(x_fit, spend[tr].astype(int), cat_features=categorical,
                    eval_set=(x_val, spend[va].astype(int)), use_best_model=True)
            oof['p'][va] = clf.predict_proba(x_val)[:, 1]
            tst['p'] += clf.predict_proba(x_tst)[:, 1] / FOLDS
            best_iter['clf'].append(clf.get_best_iteration() + 1)

        # Stage 2 regressors learn only from spenders (the classifier handles zeros), and
        # early-stop on the validation fold's spenders. They predict for every row; the
        # combine step decides which rows get the amount.
        s_fit, s_val = spend[tr], spend[va]
        if 'v2' in variants:
            # V2: amount in THB, MAE -> conditional median of y given y > 0.
            reg = CatBoostRegressor(loss_function='MAE', **params)
            reg.fit(x_fit[s_fit], y_fit[s_fit], cat_features=categorical,
                    eval_set=(x_val[s_val], y_val[s_val]), use_best_model=True)
            oof['v2_amount'][va] = np.maximum(0.0, reg.predict(x_val))
            tst['v2_amount'] += np.maximum(0.0, reg.predict(x_tst)) / FOLDS
            best_iter['v2_reg'].append(reg.get_best_iteration() + 1)

        if 'v3' in variants:
            # V3: amount as log(y / prev) with MAE; back-transform amount = prev × exp(pred).
            flag = lambda xf, raw: xf.assign(prev_missing=raw[PREV].isna().astype(int).to_numpy())  # noqa: E731
            f3, v3, t3 = flag(x_fit, fit_raw), flag(x_val, val_raw), flag(x_tst, test)
            reg = CatBoostRegressor(loss_function='MAE', **params)
            reg.fit(f3[s_fit], log_ratio(y_fit[s_fit], fit_raw[s_fit]), cat_features=categorical,
                    eval_set=(v3[s_val], log_ratio(y_val[s_val], val_raw[s_val])), use_best_model=True)
            oof['v3_amount'][va] = prev_or_one(val_raw) * np.exp(reg.predict(v3))
            tst['v3_amount'] += prev_or_one(test) * np.exp(reg.predict(t3)) / FOLDS
            best_iter['v3_reg'].append(reg.get_best_iteration() + 1)
        print(f'fold {fold}/{FOLDS} done in {time.time() - t0:.0f}s', flush=True)

    # --- Final predictions per variant (OOF and test) ---
    # Two-stage test predictions average p and amount over the 3 fold models first, then
    # apply the threshold: averaging already-thresholded predictions would give values
    # like 2/3 × amount, which is no row's median.
    final = {}  # label -> (variant, OOF prediction, test prediction, test file name)
    if 'v1' in variants:
        final['V1 single-stage + features'] = ('v1', oof['v1'], tst['v1'], 'task3_predictions.csv')
    thresholds = {}
    for v in ['v2', 'v3']:
        if v not in variants:
            continue
        amount, amount_t = oof[f'{v}_amount'], tst[f'{v}_amount']
        thresholds[v] = tune_threshold(y, oof['p'], amount)
        name = 'V2 two-stage, THB amount' if v == 'v2' else 'V3 two-stage, log-ratio amount'
        final[f'{name}, t=0.5'] = (v, combine(oof['p'], amount, 0.5), combine(tst['p'], amount_t, 0.5),
                                   'task3_predictions_t050.csv')
        final[f'{name}, t={thresholds[v]:.3f} (OOF-tuned, optimistic)'] = (
            v, combine(oof['p'], amount, thresholds[v]), combine(tst['p'], amount_t, thresholds[v]),
            'task3_predictions.csv')

    # --- Summary table ---
    rows = [['Baseline (baseline/task3_oof.csv)', *base_folds, _score(y, base), 0.0, '—']]
    for label, (_, pred, _, _) in final.items():
        fs = [_score(y[va], pred[va]) for _, va in folds]
        oof_score = _score(y, pred)
        wins = all(a > b for a, b in zip(fs, base_folds))
        rows.append([label, *fs, oof_score, oof_score - rows[0][4], 'yes' if wins else 'no'])
    summary = pd.DataFrame(rows, columns=['variant', 'fold1', 'fold2', 'fold3', 'OOF', 'Δ vs baseline',
                                          'wins all folds?'])
    shown = summary.copy()
    for c in ['fold1', 'fold2', 'fold3', 'OOF']:
        shown[c] = shown[c].map(fmt4)
    shown['Δ vs baseline'] = shown['Δ vs baseline'].map('{:+.4f}'.format)

    # --- Error by bucket: baseline vs best variant ---
    best_label = summary.iloc[1:].sort_values('OOF', ascending=False)['variant'].iloc[0]
    buckets = {'Baseline': bucket_table(y, base), best_label: bucket_table(y, final[best_label][1])}

    # --- Stage diagnostics for V2/V3 ---
    # AUC: how well stage 1 ranks spenders. Spender-only score: the regressor's amount
    # on rows with y > 0, i.e. stage 2 judged as if stage 1 were perfect (cf. oracle B
    # uses the baseline here). The baseline's own score on the same rows is the reference.
    diag = []
    if two_stage:
        diag.append(['stage 1 classifier', 'OOF AUC (y > 0)', fmt4(roc_auc_score(spend, oof['p']))])
        diag.append(['baseline on spenders', 'OOF score, y > 0 rows', fmt4(_score(y[spend], base[spend]))])
        for v in ['v2', 'v3']:
            if v in variants:
                diag.append([f'{v.upper()} regressor', 'OOF score, y > 0 rows',
                             fmt4(_score(y[spend], oof[f'{v}_amount'][spend]))])
    diag = pd.DataFrame(diag, columns=['model', 'measure', 'value'])
    iters = pd.DataFrame([[k, ' / '.join(map(str, v))] for k, v in best_iter.items() if v],
                         columns=['model', 'best iteration per fold'])

    print(f'\nTASK 3 AMOUNT MODEL ({iterations} iterations{", QUICK" if args.quick else ""})\n')
    print(shown.to_string(index=False))
    for label, table in buckets.items():
        print(f'\nError by bucket (OOF), {label}\n{fmt_bucket(table).to_string()}')
    if len(diag):
        print(f'\nStage diagnostics\n{diag.to_string(index=False)}')
    print(f'\nBest iterations (limit {iterations})\n{iters.to_string(index=False)}')

    # --- Test prediction files (outputs/ is gitignored) ---
    # V2/V3 main file uses the OOF-tuned threshold (tuning on train OOF is a legitimate
    # choice for the test file; only its OOF score is optimistic); t = 0.5 is saved next to it.
    for v, _, pred_t, name in final.values():
        folder = OUT / f'task3_{v}'
        folder.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({'player_id': test['player_id'], 'prediction': pred_t})
        assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
        frame.to_csv(folder / name, index=False)
        print(f'Wrote {(folder / name).relative_to(ROOT)}')

    # --- amount_model_results.md ---
    md = ['# Task 3 amount model',
          '',
          '> **CAUTION:** Early stopping uses the validation fold (as in the baseline), so every OOF score here '
          'is slightly optimistic; differences between rows are still comparable. Rows marked '
          '"OOF-tuned, optimistic" pick the threshold on the same OOF they are scored on.',
          '',
          f'Generated by `python experiments/task3/amount_model.py --variants {",".join(variants)}'
          f'{" --quick" if args.quick else ""}` ({iterations} iterations). Folds: `KFold({FOLDS}, shuffle=True, '
          f'random_state={SEED})`, same as the baseline. Metric: `_score` from `pis/tabular.py`. '
          f'`expected_spend` = `{PREV}` × median `y / {PREV}` among spenders per segment×VIP '
          '(fallback segment → global, groups < 30 rows skipped), fitted on the outer-training part only; '
          f'training rows get inner `KFold({INNER_FOLDS})` out-of-fold values. Threshold grid for V2/V3: '
          f'{GRID[0]:.3f}–{GRID[-1]:.3f} step 0.025.',
          '',
          '## Summary',
          '',
          to_markdown(shown),
          '',
          '## Error by bucket (OOF)',
          '']
    for label, table in buckets.items():
        md += [f'### {label}', '', to_markdown(fmt_bucket(table).reset_index()), '']
    if len(diag):
        md += ['## Stage diagnostics (V2/V3)', '', to_markdown(diag), '']
    md += [f'## Best iterations (limit {iterations})', '', to_markdown(iters), '']
    # Tables are regenerated; the Interpretation section is written by hand after
    # reading the numbers, so keep it across re-runs.
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')
    print(f'\nWrote {RESULTS.relative_to(ROOT)}')

    # --- LOG.md: one row per variant (upsert by branch + change) ---
    today, branch = datetime.date.today().isoformat(), current_branch()
    log_rows = []
    labels = {'v1': 'V1: baseline + expected_spend features',
              'v2': 'V2: two-stage (Logloss clf + MAE reg on spenders, THB)',
              'v3': 'V3: two-stage (Logloss clf + MAE reg on log(y/prev))'}
    for v in variants:
        rows_v = summary[summary['variant'].isin([k for k, f in final.items() if f[0] == v])]
        main_row = rows_v.iloc[0]  # V1: the only row; V2/V3: t = 0.5 (honest score)
        note = 'quick smoke test (250 iters)' if args.quick else ''
        if v in thresholds:
            tuned = rows_v.iloc[-1]
            note = '; '.join(filter(None, [note, f't=0.5; OOF-tuned t={thresholds[v]:.3f}: {tuned["OOF"]:.4f} '
                                                 '(optimistic)']))
        note = '; '.join(filter(None, [note, f'wins all folds: {main_row["wins all folds?"]}']))
        log_rows.append([today, branch, labels[v], ' / '.join(fmt4(main_row[f'fold{i}']) for i in (1, 2, 3)),
                         fmt4(main_row['OOF']), '—', note])
    upsert_log(log_rows)
    print(f'Updated {LOG_FILE.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
