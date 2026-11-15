"""Task 3 V4 refit: train the champion recipe (V4) on the FULL training set for Kaggle.

The competition says to combine train + validation for the final model. V4's test
predictions so far are the average of 3 fold models, each trained on 2/3 of the rows;
a refit trains on all of them. A refit leaves no held-out rows, so it has no CV score:
it is judged on Kaggle, and this script only checks that it is not broken.

Run from the repo root:
    python experiments/task3/refit/refit.py
Prints the iterations used, the runtime of every fit, the fold-1 check and the sanity
table; writes outputs/experiments/task3_v4_refit/task3_predictions.csv (gitignored) only
if every check passes, plus experiments/task3/refit/refit_results.md and one row in
experiments/task3/LOG.md. Nothing is promoted: the champion rule needs fold scores.
"""
import datetime
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold

# --- Make `pis` and ../amount_model/amount_model.py importable from anywhere ---
# Every piece of the V4 recipe (features, expected_spend leakage rule, target, quantile
# combine, CatBoost settings) is imported from amount_model.py instead of copied, so the
# refit cannot silently drift from the model that earned its CV score.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
from amount_model import (AMOUNTS, CHAMPION, CLASSIFIERS, COMPONENT_NAMES, MQ_LOSS, N_TEST, OUT,  # noqa: E402
                          RESULTS, TWO_STAGE, add_features, cap_predictions, cb_params, current_branch,
                          fold_expected_spend, log_ratio, prev_or_one, quantile_combine, upsert_log,
                          with_prev_flag)
from oracle import FOLDS, SEED, TARGET, to_markdown  # noqa: E402  (on sys.path via amount_model)
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402

# --- Settings ---
CLF, AMT = TWO_STAGE['v4']  # V4 = V3's classifier + the MultiQuantile log-ratio amount model
SEED_OFFSETS = [0, 1, 2]  # random_seed = SEED + offset = 342, 343, 344 (cb_params' scheme)
EXTRA = 1.2  # iterations × 1.2: the full train has 1.5× the rows of a CV training part
FOLD1_TOL = 0.002  # fold-1 refit vs V4's early-stopped fold-1 score
OUT_DIR = OUT / 'task3_v4_refit'
CV_PRED = OUT / 'task3_v4' / 'task3_predictions.csv'
RESULTS_MD = Path(__file__).with_name('refit_results.md')
LOG_CHANGE = 'V4 refit (full train, 3 seeds)'


# --- Iterations without early stopping ---
# With no validation rows there is nothing to early-stop on, so the iteration count is
# fixed in advance: the mean best iteration of V4's 3 CV fits, × 1.2 because the refit
# sees 50% more data (more data usually supports more trees before over-fitting). The
# CV best iterations are read from the "Best iterations" table that amount_model.py
# writes; a table from a --quick run (limit 250) is rejected.

def cv_best_iterations():
    text = RESULTS.read_text(encoding='utf-8')
    table = text[text.index('## Best iterations'):].split('\n## ')[0]
    found = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) == 3 and re.fullmatch(r'\d+( / \d+)*', cells[2]):
            found[cells[0]] = (int(cells[1]), [int(v) for v in cells[2].split(' / ')])
    out = {}
    for comp, full_limit in [(CLF, 1000), (AMT, AMOUNTS[AMT][1])]:
        name = COMPONENT_NAMES[comp]
        if name not in found:
            sys.exit(f'"{name}" missing from the Best iterations table in {RESULTS.relative_to(ROOT)}; '
                     'run amount_model.py --variants v4 first')
        limit, best = found[name]
        if limit != full_limit or len(best) != FOLDS:
            sys.exit(f'{RESULTS.relative_to(ROOT)} has limit {limit} for "{name}" (expected {full_limit}): '
                     'a --quick run overwrote it; re-run amount_model.py without --quick')
        out[comp] = (best, round(np.mean(best) * EXTRA))
    return out


def params(iterations, offset, **overrides):
    """V4's CatBoost settings with a fixed iteration count and no early stopping."""
    p = cb_params(iterations, offset, **overrides)
    p.pop('early_stopping_rounds')
    return p


# --- One V4 fit: classifier on all rows, MultiQuantile amount model on spenders ---
# Same features, target and prev_missing flag as amount_model.py. Returns the two
# models and their fit times (seconds).

def fit_v4(x_fit, raw_fit, y_fit, categorical, offset, n_iter):
    spend = y_fit > 0
    t0 = time.time()
    clf = CatBoostClassifier(loss_function='Logloss', **params(n_iter[CLF], offset, **CLASSIFIERS[CLF]))
    clf.fit(x_fit, spend.astype(int), cat_features=categorical)
    t1 = time.time()
    reg = CatBoostRegressor(loss_function=MQ_LOSS, **params(n_iter[AMT], offset))
    reg.fit(with_prev_flag(x_fit, raw_fit)[spend], log_ratio(y_fit[spend], raw_fit[spend]),
            cat_features=categorical)
    return clf, reg, (t1 - t0, time.time() - t1)


def predict_v4(clf, reg, x, raw):
    """(p, log-ratio quantiles): combined later, after averaging, like the CV fold average."""
    return clf.predict_proba(x)[:, 1], reg.predict(with_prev_flag(x, raw))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    champion = json.loads((CHAMPION / 'champion.json').read_text(encoding='utf-8'))
    if champion['variant'] != 'v4':
        sys.exit(f'champion is {champion["variant"]}, this script refits v4')
    if not CV_PRED.exists():
        sys.exit(f'{CV_PRED.relative_to(ROOT)} not found; run amount_model.py --variants v4 first')
    iters = cv_best_iterations()
    n_iter = {comp: n for comp, (_, n) in iters.items()}
    for comp, (best, n) in iters.items():
        print(f'{COMPONENT_NAMES[comp]}: CV best {" / ".join(map(str, best))}, mean {np.mean(best):.1f} '
              f'× {EXTRA} -> {n} iterations')

    # --- Data: same filter and preprocessing as amount_model.py / pis/tabular.py ---
    train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)
    x, xt, categorical = _prepare(train, test, TARGET)
    y = train[TARGET].to_numpy(dtype=float)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(x))
    timings = []  # [fit, model, iterations, seconds]

    # --- Fold-1 check: does the fixed-iteration rule (no early stopping) hold up? ---
    # Refit with the same rule on fold 1's training part only, score on fold 1's
    # validation, and compare with V4's early-stopped fold-1 score. Same seed as CV fold 1.
    # Caveat: fold 1's own best iteration is part of the mean, so this is a check of the
    # rule, not a fully independent estimate.
    tr, va = folds[0]
    fit_raw, val_raw = train.iloc[tr], train.iloc[va]
    es_fit, (es_val,) = fold_expected_spend(fit_raw, [val_raw])
    clf, reg, (tc, tr_) = fit_v4(add_features(x.iloc[tr], fit_raw, es_fit), fit_raw, y[tr], categorical, 1, n_iter)
    timings += [['fold-1 check', 'classifier', n_iter[CLF], tc], ['fold-1 check', 'amount MultiQuantile', n_iter[AMT], tr_]]
    p_va, q_va = predict_v4(clf, reg, add_features(x.iloc[va], val_raw, es_val), val_raw)
    fold1 = _score(y[va], cap_predictions(quantile_combine(p_va, q_va, prev_or_one(val_raw)), y[tr]))
    cv_fold1 = champion['fold_scores'][0]
    fold1_ok = abs(fold1 - cv_fold1) <= FOLD1_TOL
    print(f'fold-1 check: refit rule {fold1:.4f} vs V4 CV {cv_fold1:.4f} (Δ {fold1 - cv_fold1:+.4f}, '
          f'tolerance ±{FOLD1_TOL}) -> {"OK" if fold1_ok else "FAIL"}', flush=True)

    # --- V4's CV test p, re-made for the sanity table ---
    # amount_model.py saves only the final predictions, not p. The CV classifiers are
    # deterministic, so re-fitting them exactly as there (cb_params with early stopping on
    # each validation fold, seed SEED + fold) gives the same averaged test p. Check: rows
    # with p > 0.5 must be exactly the non-zero rows of the saved V4 file.
    cv = pd.read_csv(CV_PRED, float_precision='round_trip')
    assert (cv['player_id'].to_numpy() == test['player_id'].to_numpy()).all(), 'V4 file not in test.csv order'
    p_cv = np.zeros(len(test))
    for fold, (ftr, fva) in enumerate(folds, 1):
        f_raw, v_raw = train.iloc[ftr], train.iloc[fva]
        es_f, (es_v, es_t) = fold_expected_spend(f_raw, [v_raw, test])
        spend = y > 0
        c = CatBoostClassifier(loss_function='Logloss', **cb_params(1000, fold, **CLASSIFIERS[CLF]))
        c.fit(add_features(x.iloc[ftr], f_raw, es_f), spend[ftr].astype(int), cat_features=categorical,
              eval_set=(add_features(x.iloc[fva], v_raw, es_v), spend[fva].astype(int)), use_best_model=True)
        p_cv += c.predict_proba(add_features(xt, test, es_t))[:, 1] / FOLDS
    p_cv_ok = ((p_cv > 0.5) == (cv['prediction'].to_numpy() > 0)).all()
    print(f'V4 CV test p re-made: matches the saved file\'s zero pattern: {p_cv_ok}', flush=True)

    # --- Full refit, 3 seeds ---
    # expected_spend: inner KFold(5) out-of-fold on the full train for the training rows,
    # multipliers fitted on the full train for test rows: the same leakage rule as in CV,
    # with the full train playing the outer-training part. Seeds are averaged on p and on
    # the log-ratio quantiles BEFORE combining (as the CV fold average does): averaging
    # combined predictions would mix zeros and amounts into values that are no row's median.
    es_full, (es_test,) = fold_expected_spend(train, [test])
    x_full, x_test = add_features(x, train, es_full), add_features(xt, test, es_test)
    p_ref, q_ref = np.zeros(len(test)), 0.0
    for offset in SEED_OFFSETS:
        clf, reg, (tc, tr_) = fit_v4(x_full, train, y, categorical, offset, n_iter)
        timings += [[f'full, seed {SEED + offset}', 'classifier', n_iter[CLF], tc],
                    [f'full, seed {SEED + offset}', 'amount MultiQuantile', n_iter[AMT], tr_]]
        p, q = predict_v4(clf, reg, x_test, test)
        p_ref += p / len(SEED_OFFSETS)
        q_ref = q_ref + q / len(SEED_OFFSETS)
        print(f'seed {SEED + offset}: classifier {tc:.0f}s, amount {tr_:.0f}s', flush=True)
    # Post-processing cap (amount_model.cap_predictions): clip to [0, full-train max target].
    raw_pred = quantile_combine(p_ref, q_ref, prev_or_one(test))
    pred = cap_predictions(raw_pred, y)
    n_capped = int((pred != raw_pred).sum())
    print(f'cap at {y.max():,.0f}: {n_capped} test row(s) changed', flush=True)

    # --- Sanity table: refit vs V4's CV-averaged test predictions ---
    # No CV score is possible, so we check that the refit tells the same story as the CV
    # models: same ranking (Spearman), same share of zeros, similar totals and mean p.
    # Medians are shown for non-zero rows too, because ~50% zeros put the overall median at 0.
    old = cv['prediction'].to_numpy(dtype=float)
    spearman = pd.Series(pred).corr(pd.Series(old), method='spearman')
    stats = lambda v, p: [(v == 0).mean(), v.sum(), np.median(v), np.median(v[v > 0]), p.mean()]  # noqa: E731
    a, b = stats(old, p_cv), stats(pred, p_ref)
    checks = {'Spearman > 0.97': spearman > 0.97,
              'zero share within 0.01': abs(b[0] - a[0]) <= 0.01,
              'sum within 5%': abs(b[1] / a[1] - 1) <= 0.05,
              'median of non-zero within 5%': abs(b[3] / a[3] - 1) <= 0.05,
              'mean p within 0.01': abs(b[4] - a[4]) <= 0.01,
              f'fold-1 check within ±{FOLD1_TOL}': fold1_ok,
              'CV p re-made exactly': p_cv_ok}
    fmt = ['{:.4f}', '{:,.0f}', '{:,.2f}', '{:,.2f}', '{:.4f}']
    sanity = pd.DataFrame({'measure': ['zero share', 'sum (THB)', 'median', 'median of non-zero', 'mean p'],
                           'V4 CV average (3 fold models)': [f.format(v) for f, v in zip(fmt, a)],
                           'V4 refit (full train, 3 seeds)': [f.format(v) for f, v in zip(fmt, b)],
                           'refit / CV': [f'{d:+.4f}' if i in (0, 4) else f'{d:+.2%}' for i, d in
                                          enumerate([b[0] - a[0], b[1] / a[1] - 1, 0.0 if a[2] == 0 else b[2] / a[2] - 1,
                                                     b[3] / a[3] - 1, b[4] - a[4]])]})
    sanity.loc[len(sanity)] = ['Spearman (refit vs CV)', '—', f'{spearman:.4f}', '—']
    check_table = pd.DataFrame({'check': list(checks), 'result': ['pass' if v else 'FAIL' for v in checks.values()]})
    times = pd.DataFrame(timings, columns=['fit', 'model', 'iterations', 'seconds'])
    times['seconds'] = times['seconds'].map('{:.0f}'.format)
    ok = all(checks.values())

    print(f'\nSanity: refit vs V4 CV-averaged test predictions\n{sanity.to_string(index=False)}')
    print(f'\nChecks\n{check_table.to_string(index=False)}')
    print(f'\nRuntimes\n{times.to_string(index=False)}')

    # --- Output: only a refit that passes every check gets a predictions file ---
    if ok:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({'player_id': test['player_id'], 'prediction': pred})
        assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
        frame.to_csv(OUT_DIR / 'task3_predictions.csv', index=False)
        print(f'\nWrote {(OUT_DIR / "task3_predictions.csv").relative_to(ROOT)}')
    else:
        print('\nA check FAILED: no predictions file written; investigate before building a submission.')

    # --- refit_results.md and ../LOG.md ---
    md = ['# Task 3 V4 refit (full train)', '',
          '> **CAUTION:** A refit has no held-out rows, so there is no CV score. It is judged on Kaggle; '
          'these tables only check that it agrees with the CV models.', '',
          'Generated by `python experiments/task3/refit/refit.py`. Recipe: V4 from '
          '`experiments/task3/amount_model/amount_model.py` (classifier + `MultiQuantile` amount model on '
          '`log(y / prev)`, `quantile_combine` with s = 1, t = 0.5), trained on all training rows without '
          f'early stopping; seeds {", ".join(str(SEED + o) for o in SEED_OFFSETS)} averaged on p and log-quantiles '
          'before combining. `expected_spend`: inner `KFold(5)` OOF on the full train for training rows, '
          'full-train multipliers for test rows.', '',
          '## Iterations', '',
          to_markdown(pd.DataFrame([[COMPONENT_NAMES[c], ' / '.join(map(str, best)), f'{np.mean(best):.1f}', n]
                                    for c, (best, n) in iters.items()],
                                   columns=['model', 'CV best iterations', 'mean', f'used (mean × {EXTRA})'])), '',
          '## Fold-1 check', '',
          f'Same recipe and iteration rule, trained on fold 1\'s training part, scored on fold 1\'s validation: '
          f'**{fold1:.4f}** vs V4\'s early-stopped fold-1 score {cv_fold1:.4f} (Δ {fold1 - cv_fold1:+.4f}).', '',
          '## Sanity: refit vs V4 CV-averaged test predictions', '', to_markdown(sanity), '',
          to_markdown(check_table), '', '## Runtimes', '', to_markdown(times), '']
    RESULTS_MD.write_text('\n'.join(md), encoding='utf-8')
    note = (f'no CV possible, judged on Kaggle; iterations clf {n_iter[CLF]}, amount {n_iter[AMT]} '
            f'(CV mean × {EXTRA}); fold-1 check {fold1:.4f} vs V4 CV {cv_fold1:.4f} ({fold1 - cv_fold1:+.4f}); '
            f'vs V4 CV test preds: Spearman {spearman:.4f}, zero share {b[0]:.4f} vs {a[0]:.4f}, '
            f'sum {b[1] / a[1] - 1:+.2%}, median non-zero {b[3]:,.0f} vs {a[3]:,.0f}, mean p {b[4]:.4f} vs {a[4]:.4f}; '
            f'runtime per seed clf {np.mean([t[3] for t in timings[2:] if t[1] == "classifier"]):.0f}s + amount '
            f'{np.mean([t[3] for t in timings[2:] if t[1] != "classifier"]):.0f}s; checks: {"all pass" if ok else "FAILED"}; '
            '`refit/`, not promoted (champion rule needs fold scores)')
    upsert_log([[datetime.date.today().isoformat(), current_branch(), LOG_CHANGE, '—', '— (not measurable)', '—', note]])
    print(f'Wrote {RESULTS_MD.relative_to(ROOT)}; updated experiments/task3/LOG.md')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
