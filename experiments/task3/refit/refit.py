"""Task 3 refit: train a two-stage recipe (V4 or V8) on the FULL training set for Kaggle.

The competition says to combine train + validation for the final model. CV test
predictions are the average of 3 fold models, each trained on 2/3 of the rows; a refit
trains on all of them. A refit leaves no held-out rows, so it has no CV score: it is
judged on Kaggle, and this script only checks that it is not broken.

Variants (same classifier + MultiQuantile amount recipe, quantile combine s = 1, t = 0.5):
  - v4: V4's settings; iterations from V4's val-es CV run (amount_model_results.md);
  - v8: V4 with the Optuna-tuned amount settings of experiments/task3/tuning/v8.json;
        iterations from its honest-es CV run (classifier and amount best iterations).
Both: iterations = mean CV best iteration × 1.2, seeds 342-344, and final predictions
clipped with amount_model.cap_predictions to [0, full-train max target] (300,000).

Run from the repo root:
    python experiments/task3/refit/refit.py [--variant v4|v8] [--threads N]
--threads sets CatBoost's thread_count (default -1 = all cores). On a fresh clone, v8 first
needs its CV outputs: python experiments/task3/tuning/v8_cv.py (rebuilds them from v8.json).
Prints the elapsed time of every step, the iterations used, the runtime of every fit, the fold-1 check and the sanity
table; writes outputs/experiments/task3_<variant>_refit/task3_predictions.csv (capped) and
task3_predictions_uncapped.csv (gitignored) only if every check passes, plus the results
file (refit_results.md for v4, refit_v8_results.md for v8) and one row in
experiments/task3/LOG.md. Nothing is promoted: the champion rule needs fold scores.
"""
import argparse
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
# Every piece of the recipe (features, expected_spend leakage rule, target, quantile
# combine, CatBoost settings, cap) is imported from amount_model.py instead of copied, so
# the refit cannot silently drift from the model that earned its CV score.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
from amount_model import (AMOUNTS, CHAMPION, CLASSIFIERS, COMPONENT_NAMES, MQ_LOSS, N_TEST, OUT,  # noqa: E402
                          RESULTS, TWO_STAGE, add_features, cap_predictions, cb_params, current_branch,
                          fold_expected_spend, log_ratio, prev_or_one, quantile_combine, set_threads,
                          upsert_log, with_prev_flag)
from oracle import FOLDS, SEED, TARGET, to_markdown  # noqa: E402  (on sys.path via amount_model)
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402

# --- Settings ---
CLF, AMT = TWO_STAGE['v4']  # V4 = V3's classifier + the MultiQuantile log-ratio amount model
SEED_OFFSETS = [0, 1, 2]  # random_seed = SEED + offset = 342, 343, 344 (cb_params' scheme)
EXTRA = 1.2  # iterations × 1.2: the full train has more rows than a CV training part
FOLD1_TOL = 0.002  # fold-1 refit vs the variant's CV fold-1 score
TASK3 = Path(__file__).resolve().parents[1]
V8_JSON = TASK3 / 'tuning' / 'v8.json'
VARIANTS = {
    'v4': dict(name='V4', cv_pred=OUT / 'task3_v4' / 'task3_predictions.csv',
               results=Path(__file__).with_name('refit_results.md'),
               log='V4 refit + cap (full train, 3 seeds, predictions clipped to [0, 300,000])',
               recipe='V4 from `experiments/task3/amount_model/amount_model.py` (val-es protocol)'),
    'v8': dict(name='V8', cv_pred=OUT / 'task3_v8' / 'task3_predictions.csv',
               results=Path(__file__).with_name('refit_v8_results.md'),
               log='V8 refit (full train, 3 seeds, cap)',
               recipe='V8 = V4 with the Optuna-tuned amount settings of `experiments/task3/tuning/v8.json` '
                      '(honest-es protocol)')}


# --- Iterations without early stopping ---
# With no validation rows there is nothing to early-stop on, so the iteration count is
# fixed in advance: the mean best iteration of the variant's 3 CV fits, × 1.2 because the
# refit sees more data (more data usually supports more trees before over-fitting).
# V4: read from the "Best iterations" table that amount_model.py writes (a table from a
# --quick run, limit 250, is rejected). V8: from v8.json (honest-es best iterations).

def v4_best_iterations():
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
        out[comp] = best
    return out


def setup(variant, champion):
    """(best CV iterations per component, amount overrides, CV fold-1 score, config text) of a variant."""
    if variant == 'v4':
        # V4's val-es scores: champion.json keeps them under previous_protocol once the
        # champion is scored under honest-es.
        old = champion if champion.get('protocol', 'val-es') == 'val-es' else champion.get('previous_protocol') or {}
        if old.get('variant') != 'v4':
            sys.exit('V4 val-es fold scores not found in champion.json')
        return v4_best_iterations(), {}, old['fold_scores'][0], 'V4 CV (val-es)'
    if not V8_JSON.exists():
        sys.exit(f'{V8_JSON.relative_to(ROOT)} not found: V8 was not promoted by tuning.py')
    v8 = json.loads(V8_JSON.read_text(encoding='utf-8'))
    best = {CLF: v8['classifier_best_iterations'], AMT: v8['amount_best_iterations']}
    return best, v8['amount_params'], v8['fold_scores'][0], 'V8 CV (honest-es)'


def params(iterations, offset, **overrides):
    """V4's CatBoost settings (+ overrides) with a fixed iteration count and no early stopping."""
    p = cb_params(iterations, offset, **overrides)
    p.pop('early_stopping_rounds')
    return p


# --- One fit: classifier on all rows, MultiQuantile amount model on spenders ---
# Same features, target and prev_missing flag as amount_model.py. Returns the two
# models and their fit times (seconds).

def fit_two_stage(x_fit, raw_fit, y_fit, categorical, offset, n_iter, amount_params):
    spend = y_fit > 0
    t0 = time.time()
    clf = CatBoostClassifier(loss_function='Logloss', **params(n_iter[CLF], offset, **CLASSIFIERS[CLF]))
    clf.fit(x_fit, spend.astype(int), cat_features=categorical)
    t1 = time.time()
    reg = CatBoostRegressor(loss_function=MQ_LOSS, **params(n_iter[AMT], offset, **amount_params))
    reg.fit(with_prev_flag(x_fit, raw_fit)[spend], log_ratio(y_fit[spend], raw_fit[spend]),
            cat_features=categorical)
    return clf, reg, (t1 - t0, time.time() - t1)


def predict(clf, reg, x, raw):
    """(p, log-ratio quantiles): combined later, after averaging, like the CV fold average."""
    return clf.predict_proba(x)[:, 1], reg.predict(with_prev_flag(x, raw))


def cv_test_p(variant, train, test, x, xt, categorical, y, folds, cv):
    """Averaged CV test p of the variant's classifier, and whether it matches the saved zero pattern.

    V4 (val-es): CV classifiers are deterministic, so re-fitting them exactly as in
    amount_model.py (early stopping on each validation fold, seed SEED + fold) gives the
    same averaged test p. V8 (honest-es): tuning.py keeps the classifier p fixed, saved by
    honest_es.py in outputs/experiments/task3_v4_honest/classifier_p.npz.
    Check: rows with p > 0.5 must be exactly the non-zero rows of the saved CV file.
    """
    if variant == 'v8':
        path = OUT / 'task3_v4_honest' / 'classifier_p.npz'
        if not path.exists():
            sys.exit(f'{path.relative_to(ROOT)} not found; run experiments/task3/tuning/v8_cv.py')
        p_cv = np.load(path)['p_test'].mean(axis=0)
    else:
        p_cv = np.zeros(len(test))
        for fold, (ftr, fva) in enumerate(folds, 1):
            f_raw, v_raw = train.iloc[ftr], train.iloc[fva]
            es_f, (es_v, es_t) = fold_expected_spend(f_raw, [v_raw, test])
            spend = y > 0
            c = CatBoostClassifier(loss_function='Logloss', **cb_params(1000, fold, **CLASSIFIERS[CLF]))
            c.fit(add_features(x.iloc[ftr], f_raw, es_f), spend[ftr].astype(int), cat_features=categorical,
                  eval_set=(add_features(x.iloc[fva], v_raw, es_v), spend[fva].astype(int)), use_best_model=True)
            p_cv += c.predict_proba(add_features(xt, test, es_t))[:, 1] / FOLDS
    return p_cv, bool(((p_cv > 0.5) == (cv['prediction'].to_numpy() > 0)).all())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='v4')
    parser.add_argument('--threads', type=int, default=-1, help='CatBoost thread_count (default -1 = all cores)')
    args = parser.parse_args()
    set_threads(args.threads)
    cfg = VARIANTS[args.variant]
    t_start = t_step = time.time()
    steps = []  # [step, seconds]

    def step_done(name):
        nonlocal t_step
        now = time.time()
        steps.append([name, now - t_step])
        print(f'[{now - t_step:6.0f}s] {name}', flush=True)
        t_step = now

    out_dir = OUT / f'task3_{args.variant}_refit'
    champion = json.loads((CHAMPION / 'champion.json').read_text(encoding='utf-8'))
    if not cfg['cv_pred'].exists():
        how = ('python experiments/task3/tuning/v8_cv.py' if args.variant == 'v8'
               else 'python experiments/task3/amount_model/amount_model.py --variants v4')
        sys.exit(f'{cfg["cv_pred"].relative_to(ROOT)} not found; run the {cfg["name"]} CV first: {how}')
    best_cv, amount_params, cv_fold1, cv_name = setup(args.variant, champion)
    n_iter = {comp: round(np.mean(best) * EXTRA) for comp, best in best_cv.items()}
    names = {**COMPONENT_NAMES, **({AMT: 'amount log-ratio MultiQuantile, V8 settings (limit 5000)'}
                                   if args.variant == 'v8' else {})}
    for comp, best in best_cv.items():
        print(f'{names[comp]}: CV best {" / ".join(map(str, best))}, mean {np.mean(best):.1f} '
              f'× {EXTRA} -> {n_iter[comp]} iterations')
    if amount_params:
        print(f'amount settings: {amount_params}')
    print(f'CatBoost threads: {"all cores" if args.threads == -1 else args.threads}')

    # --- Data: same filter and preprocessing as amount_model.py / pis/tabular.py ---
    train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)
    x, xt, categorical = _prepare(train, test, TARGET)
    y = train[TARGET].to_numpy(dtype=float)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(x))
    timings = []  # [fit, model, iterations, seconds]
    step_done('data + folds')

    # --- Fold-1 check: does the fixed-iteration rule (no early stopping) hold up? ---
    # Refit with the same rule on fold 1's training part only, score on fold 1's
    # validation, and compare with the variant's early-stopped fold-1 score. Same seed as
    # CV fold 1. Caveat: fold 1's own best iteration is part of the mean, so this is a
    # check of the rule, not a fully independent estimate. For V8 the CV models trained on
    # 85% of the training part (honest-es), the check on all of it.
    tr, va = folds[0]
    fit_raw, val_raw = train.iloc[tr], train.iloc[va]
    es_fit, (es_val,) = fold_expected_spend(fit_raw, [val_raw])
    clf, reg, (tc, tr_) = fit_two_stage(add_features(x.iloc[tr], fit_raw, es_fit), fit_raw, y[tr], categorical, 1,
                                        n_iter, amount_params)
    timings += [['fold-1 check', 'classifier', n_iter[CLF], tc], ['fold-1 check', 'amount MultiQuantile', n_iter[AMT], tr_]]
    p_va, q_va = predict(clf, reg, add_features(x.iloc[va], val_raw, es_val), val_raw)
    fold1 = _score(y[va], cap_predictions(quantile_combine(p_va, q_va, prev_or_one(val_raw)), y[tr]))
    fold1_ok = abs(fold1 - cv_fold1) <= FOLD1_TOL
    print(f'fold-1 check: refit rule {fold1:.4f} vs {cv_name} {cv_fold1:.4f} (Δ {fold1 - cv_fold1:+.4f}, '
          f'tolerance ±{FOLD1_TOL}) -> {"OK" if fold1_ok else "FAIL"}', flush=True)
    step_done('fold-1 check')

    # --- The variant's CV test p, for the sanity table ---
    cv = pd.read_csv(cfg['cv_pred'], float_precision='round_trip')
    assert (cv['player_id'].to_numpy() == test['player_id'].to_numpy()).all(), 'CV file not in test.csv order'
    p_cv, p_cv_ok = cv_test_p(args.variant, train, test, x, xt, categorical, y, folds, cv)
    print(f'{cfg["name"]} CV test p: matches the saved file\'s zero pattern: {p_cv_ok}', flush=True)
    step_done(f'{cfg["name"]} CV test p')

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
        clf, reg, (tc, tr_) = fit_two_stage(x_full, train, y, categorical, offset, n_iter, amount_params)
        timings += [[f'full, seed {SEED + offset}', 'classifier', n_iter[CLF], tc],
                    [f'full, seed {SEED + offset}', 'amount MultiQuantile', n_iter[AMT], tr_]]
        p, q = predict(clf, reg, x_test, test)
        p_ref += p / len(SEED_OFFSETS)
        q_ref = q_ref + q / len(SEED_OFFSETS)
        print(f'seed {SEED + offset}: classifier {tc:.0f}s, amount {tr_:.0f}s', flush=True)
        step_done(f'full refit, seed {SEED + offset}')
    # Post-processing cap (amount_model.cap_predictions): clip to [0, full-train max target].
    raw_pred = quantile_combine(p_ref, q_ref, prev_or_one(test))
    pred = cap_predictions(raw_pred, y)
    n_capped = int((pred != raw_pred).sum())
    print(f'cap at {y.max():,.0f}: {n_capped} test row(s) changed', flush=True)
    # Determinism: an earlier refit's uncapped file (or, before the cap existed, its main
    # file) must be reproduced exactly by the same recipe.
    previous = next((p for p in [out_dir / 'task3_predictions_uncapped.csv', out_dir / 'task3_predictions.csv']
                     if p.exists()), None)
    repro = None
    if previous is not None:
        old_pred = pd.read_csv(previous, float_precision='round_trip')['prediction'].to_numpy(dtype=float)
        if len(old_pred) == len(raw_pred):
            repro = float(np.max(np.abs(old_pred - raw_pred)))
            print(f'uncapped refit vs the earlier file {previous.name}: max |Δ| {repro:.6g}', flush=True)

    # --- Sanity table: refit vs the variant's CV-averaged test predictions ---
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
              'CV p matches the CV file': p_cv_ok}
    fmt = ['{:.4f}', '{:,.0f}', '{:,.2f}', '{:,.2f}', '{:.4f}']
    name = cfg['name']
    sanity = pd.DataFrame({'measure': ['zero share', 'sum (THB)', 'median', 'median of non-zero', 'mean p'],
                           f'{name} CV average (3 fold models)': [f.format(v) for f, v in zip(fmt, a)],
                           f'{name} refit (full train, 3 seeds, cap)': [f.format(v) for f, v in zip(fmt, b)],
                           'refit / CV': [f'{d:+.4f}' if i in (0, 4) else f'{d:+.2%}' for i, d in
                                          enumerate([b[0] - a[0], b[1] / a[1] - 1, 0.0 if a[2] == 0 else b[2] / a[2] - 1,
                                                     b[3] / a[3] - 1, b[4] - a[4]])]})
    sanity.loc[len(sanity)] = ['Spearman (refit vs CV)', '—', f'{spearman:.4f}', '—']
    check_table = pd.DataFrame({'check': list(checks), 'result': ['pass' if v else 'FAIL' for v in checks.values()]})
    times = pd.DataFrame(timings, columns=['fit', 'model', 'iterations', 'seconds'])
    times['seconds'] = times['seconds'].map('{:.0f}'.format)
    ok = all(checks.values())

    print(f'\nSanity: refit vs {name} CV-averaged test predictions\n{sanity.to_string(index=False)}')
    print(f'\nChecks\n{check_table.to_string(index=False)}')
    print(f'\nRuntimes\n{times.to_string(index=False)}')

    # --- Output: only a refit that passes every check gets a predictions file ---
    if ok:
        out_dir.mkdir(parents=True, exist_ok=True)
        for file, values in [('task3_predictions.csv', pred), ('task3_predictions_uncapped.csv', raw_pred)]:
            frame = pd.DataFrame({'player_id': test['player_id'], 'prediction': values})
            assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
            frame.to_csv(out_dir / file, index=False)
        print(f'\nWrote {(out_dir / "task3_predictions.csv").relative_to(ROOT)} (capped) and the uncapped file')
    else:
        print('\nA check FAILED: no predictions file written; investigate before building a submission.')

    # --- Results file and ../LOG.md ---
    amount_text = (f' Amount settings: `{json.dumps(amount_params)}`.' if amount_params else '')
    md = [f'# Task 3 {name} refit (full train)', '',
          '> **CAUTION:** A refit has no held-out rows, so there is no CV score. It is judged on Kaggle; '
          'these tables only check that it agrees with the CV models.', '',
          f'Generated by `python experiments/task3/refit/refit.py --variant {args.variant}`. Recipe: {cfg["recipe"]} '
          '(classifier + `MultiQuantile` amount model on `log(y / prev)`, `quantile_combine` with s = 1, t = 0.5), '
          f'trained on all training rows without early stopping; seeds {", ".join(str(SEED + o) for o in SEED_OFFSETS)} '
          'averaged on p and log-quantiles before combining; final predictions clipped to [0, full-train max target] '
          f'= [0, {y.max():,.0f}] (`cap_predictions`; {n_capped} test row(s) changed by the cap). `expected_spend`: inner '
          f'`KFold(5)` OOF on the full train for training rows, full-train multipliers for test rows.{amount_text}', '',
          '## Iterations', '',
          to_markdown(pd.DataFrame([[names[c], ' / '.join(map(str, best)), f'{np.mean(best):.1f}', n_iter[c]]
                                    for c, best in best_cv.items()],
                                   columns=['model', f'CV best iterations ({cv_name})', 'mean',
                                            f'used (mean × {EXTRA})'])), '',
          '## Fold-1 check', '',
          f'Same recipe and iteration rule, trained on fold 1\'s training part, scored on fold 1\'s validation: '
          f'**{fold1:.4f}** vs {cv_name} fold-1 score {cv_fold1:.4f} (Δ {fold1 - cv_fold1:+.4f}).', '']
    if repro is not None:
        md += ['## Determinism', '',
               f'Uncapped refit vs the earlier refit file `{previous.name}`: max |Δ| = {repro:.6g} THB.', '']
    md += [f'## Sanity: refit vs {name} CV-averaged test predictions', '', to_markdown(sanity), '',
           to_markdown(check_table), '', '## Runtimes', '', to_markdown(times), '']
    cfg['results'].write_text('\n'.join(md), encoding='utf-8')
    note = (f'no CV possible, judged on Kaggle; iterations clf {n_iter[CLF]}, amount {n_iter[AMT]} '
            f'(CV mean × {EXTRA}); cap: {n_capped} test row(s) changed; fold-1 check {fold1:.4f} vs {cv_name} '
            f'{cv_fold1:.4f} ({fold1 - cv_fold1:+.4f}); vs {name} CV test preds: Spearman {spearman:.4f}, zero share '
            f'{b[0]:.4f} vs {a[0]:.4f}, sum {b[1] / a[1] - 1:+.2%}, median non-zero {b[3]:,.0f} vs {a[3]:,.0f}, '
            f'mean p {b[4]:.4f} vs {a[4]:.4f}; '
            + (f'uncapped = earlier refit file (max |Δ| {repro:.3g}); ' if repro is not None else '')
            + f'runtime per seed clf {np.mean([t[3] for t in timings[2:] if t[1] == "classifier"]):.0f}s + amount '
            f'{np.mean([t[3] for t in timings[2:] if t[1] != "classifier"]):.0f}s; checks: {"all pass" if ok else "FAILED"}; '
            '`refit/`, not promoted (champion rule needs fold scores)')
    upsert_log([[datetime.date.today().isoformat(), current_branch(), cfg['log'], '—', '— (not measurable)', '—', note]])
    print(f'Wrote {cfg["results"].relative_to(ROOT)}; updated experiments/task3/LOG.md')
    step_done('sanity checks + write files')
    print('\nElapsed per step\n' + '\n'.join(f'  {name:<32} {sec:7.0f}s' for name, sec in steps)
          + f'\n  {"total":<32} {time.time() - t_start:7.0f}s')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
