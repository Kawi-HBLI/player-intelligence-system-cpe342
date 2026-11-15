"""Task 3 V6: V4 with an ensembled stage-2 amount model (CatBoost + LightGBM quantiles).

The oracle test and the classifier feature screen both point at the amount stage ("how
much?") as the part with room left. V6 keeps everything of V4 (classifier, features,
folds, expected_spend leakage rule, log(y / prev) target with the prev_missing flag,
quantile_combine with s = 1 and t = 0.5) and changes only where the amount quantiles
come from: the average, on the log-ratio scale and per alpha, of V4's CatBoost
MultiQuantile model and one LightGBM quantile model per alpha. Two model families
usually make different mistakes, so the average should cut amount error.

  - V6:  w = 0.5 (fixed before the run; honest, the only promotion candidate);
  - V6w: w chosen on OOF from 0.0–1.0 step 0.1 (optimistic, reported only);
  - LightGBM alone (w = 0): reference row.

Run from the repo root:
    python experiments/task3/ensemble/ensemble.py [--quick] [--promote]
Prints the summary, bucket table, quantile coverage and runtimes; writes
experiments/task3/ensemble/ensemble_results.md (the hand-written Interpretation section
is kept), rows in experiments/task3/LOG.md and OOF/test predictions in
outputs/experiments/task3_<variant>/ (gitignored). --quick (250 trees per model) only
prints. --promote makes V6 the champion if it beats V4 on all 3 folds.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold

# --- Make `pis`, ../amount_model and ../oracle importable from anywhere ---
# Every V4 piece is imported, not copied, so V6 differs from V4 only in the amount model.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
from amount_model import (ALPHAS, AMOUNTS, CLASSIFIERS, COMPONENT_NAMES, MQ_LOSS, N_TEST, OUT,  # noqa: E402
                          TWO_STAGE, add_features, cb_params, current_branch, fold_expected_spend,
                          load_champion, log_ratio, prev_or_one, promote, quantile_combine, upsert_log,
                          with_prev_flag)
from oracle import (BUCKETS, EXPECTED_FOLDS, EXPECTED_OOF, FOLDS, SEED, TARGET,  # noqa: E402
                    bucket_table, to_markdown)
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402

# --- Settings ---
CLF, AMT = TWO_STAGE['v4']  # V4's classifier ('clf') and MultiQuantile amount model ('mq')
W_FIXED = 0.5  # V6 blend weight on CatBoost, fixed before the run
W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)  # V6w scan; w = 0 is LightGBM alone, w = 1 is V4
LGB_CAP, LGB_ES = 3000, 100
LGB_NAME = 'LightGBM quantile α={:.2f}'  # one model per alpha; refit.py reads these names
RESULTS = Path(__file__).with_name('ensemble_results.md')
INTERPRETATION = '## Interpretation'
NAMES = {'v4': 'V4 (re-run here)',
         'v6_lgbm': 'LightGBM alone (w=0, reference)',
         'v6': f'V6 = V4 + LightGBM amount blend, w={W_FIXED}',
         'v6w': 'V6w = V6 with w tuned on OOF'}
LOG_LABELS = {'v6_lgbm': 'V6 reference: V4 classifier + LightGBM quantile amount alone (w=0)',
              'v6': 'V6: V4 + amount quantiles blended 0.5 CatBoost MultiQuantile + 0.5 LightGBM quantile (log-ratio scale)',
              'v6w': 'V6w: V6 with blend weight w tuned on OOF'}


# --- LightGBM quantile models ---
# Same rows (spenders), target (log(y / prev), prev treated as 1 when missing) and
# features (V4's + prev_missing) as V4's MultiQuantile model, one model per alpha.
# Categoricals: `_prepare` gives them as strings ('__MISSING__' for NaN, as CatBoost
# sees them); LightGBM gets them as pandas `category` with one fixed category list per
# column (from train + test, no target involved), so codes mean the same in every frame.

def lgb_categories(x, xt, categorical):
    return {c: sorted(set(x[c]) | set(xt[c])) for c in categorical}


def to_lgb(frame, cats):
    out = frame.copy()
    for c, values in cats.items():
        out[c] = pd.Categorical(out[c], categories=values)
    return out


def lgb_params(alpha, seed, n_estimators=LGB_CAP):
    return dict(objective='quantile', alpha=alpha, num_leaves=63, learning_rate=0.05, n_estimators=n_estimators,
                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=seed, n_jobs=-1, verbose=-1)


# --- Blend ---
# Per alpha on the log-ratio scale: log_q = w · CatBoost + (1 − w) · LightGBM. The
# quantiles of the two models are averaged before back-transforming, like the fold and
# seed averages; quantile_combine then sorts each row (crossing alphas) and picks the
# (1 − 0.5/p) quantile exactly as in V4.

def blend(cb, lg, w):
    return w * cb + (1.0 - w) * lg


def coverage(truth, log_q):
    """Share of spenders whose true log-ratio is ≤ each (row-sorted) predicted quantile."""
    return (truth[:, None] <= np.sort(log_q, axis=1)).mean(axis=0)


def fmt4(v):
    return f'{v:.4f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--quick', action='store_true', help='250 trees per model; prints only (smoke test)')
    parser.add_argument('--promote', action='store_true', help='make V6 the champion if it beats V4 on all folds')
    args = parser.parse_args()
    if args.promote and args.quick:
        parser.error('--promote needs a full run, not --quick')
    cap = lambda full: 250 if args.quick else full  # noqa: E731
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
    cats = lgb_categories(x, xt, categorical)

    n, nt, k = len(train), len(test), len(ALPHAS)
    p_oof, p_tst = np.zeros(n), np.zeros(nt)
    cb_oof, cb_tst = np.zeros((n, k)), np.zeros((nt, k))
    lg_oof, lg_tst = np.zeros((n, k)), np.zeros((nt, k))
    best = {CLF: [], AMT: [], **{a: [] for a in ALPHAS}}
    seconds = {'classifier': [], 'CatBoost MultiQuantile': [], 'LightGBM, 10 alphas': [], 'features': []}

    # --- Outer CV: V4's two models exactly as in amount_model.py, then LightGBM ---
    for fold, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        fit_raw, val_raw = train.iloc[tr], train.iloc[va]
        es_fit, (es_val, es_test) = fold_expected_spend(fit_raw, [val_raw, test])
        x_fit = add_features(x.iloc[tr], fit_raw, es_fit)
        x_val = add_features(x.iloc[va], val_raw, es_val)
        x_tst = add_features(xt, test, es_test)
        seconds['features'].append(time.time() - t0)

        t0 = time.time()
        clf = CatBoostClassifier(loss_function='Logloss', **cb_params(cap(1000), fold, **CLASSIFIERS[CLF]))
        clf.fit(x_fit, spend[tr].astype(int), cat_features=categorical,
                eval_set=(x_val, spend[va].astype(int)), use_best_model=True)
        p_oof[va] = clf.predict_proba(x_val)[:, 1]
        p_tst += clf.predict_proba(x_tst)[:, 1] / FOLDS
        best[CLF].append(clf.get_best_iteration() + 1)
        seconds['classifier'].append(time.time() - t0)

        s_fit, s_val = spend[tr], spend[va]
        f3, v3, t3 = with_prev_flag(x_fit, fit_raw), with_prev_flag(x_val, val_raw), with_prev_flag(x_tst, test)
        y3_fit = log_ratio(y[tr][s_fit], fit_raw[s_fit])
        y3_val = log_ratio(y[va][s_val], val_raw[s_val])

        t0 = time.time()
        reg = CatBoostRegressor(loss_function=MQ_LOSS, **cb_params(cap(AMOUNTS[AMT][1]), fold))
        reg.fit(f3[s_fit], y3_fit, cat_features=categorical, eval_set=(v3[s_val], y3_val), use_best_model=True)
        cb_oof[va] = reg.predict(v3)
        cb_tst += reg.predict(t3) / FOLDS
        best[AMT].append(reg.get_best_iteration() + 1)
        seconds['CatBoost MultiQuantile'].append(time.time() - t0)

        # LightGBM: early stopping on the validation fold's spenders with the same quantile
        # loss (as CatBoost does; same known optimism, applied to both alike).
        t0 = time.time()
        lf, lv, lt = to_lgb(f3, cats), to_lgb(v3, cats), to_lgb(t3, cats)
        for j, a in enumerate(ALPHAS):
            m = lgb.LGBMRegressor(**lgb_params(float(a), SEED + fold, cap(LGB_CAP)))
            m.fit(lf[s_fit], y3_fit, eval_set=[(lv[s_val], y3_val)], eval_metric='quantile',
                  callbacks=[lgb.early_stopping(LGB_ES, verbose=False)])
            lg_oof[va, j] = m.predict(lv)
            lg_tst[:, j] += m.predict(lt) / FOLDS
            best[a].append(m.best_iteration_ or cap(LGB_CAP))
        seconds['LightGBM, 10 alphas'].append(time.time() - t0)
        print(f'fold {fold}/{FOLDS}: classifier {seconds["classifier"][-1]:.0f}s, CatBoost MQ '
              f'{seconds["CatBoost MultiQuantile"][-1]:.0f}s, LightGBM {seconds["LightGBM, 10 alphas"][-1]:.0f}s',
              flush=True)

    prev_tr, prev_te = prev_or_one(train), prev_or_one(test)

    # --- V4 re-run must reproduce the champion (champion OOF is rounded to satang) ---
    v4_oof = quantile_combine(p_oof, cb_oof, prev_tr)
    v4_folds = [_score(y[va], v4_oof[va]) for _, va in folds]
    v4_ok = all(abs(a - b) <= 5e-5 for a, b in zip(v4_folds, champ_folds))
    print(f'V4 re-run: {" / ".join(map(fmt4, v4_folds))} vs champion {" / ".join(map(fmt4, champ_folds))} '
          f'-> {"reproduced" if v4_ok else "NOT reproduced"}', flush=True)
    if not v4_ok and not args.quick:
        sys.exit('V4 re-run does not reproduce the champion; comparisons would not be valid')

    # --- Variants ---
    w_scan = pd.Series({w: _score(y, quantile_combine(p_oof, blend(cb_oof, lg_oof, w), prev_tr)) for w in W_GRID})
    best_w = float(w_scan.idxmax())
    final = {}
    for v, w in [('v6_lgbm', 0.0), ('v6', W_FIXED), ('v6w', best_w)]:
        label = NAMES[v] + (f', w={best_w:.1f} (OOF-tuned, optimistic)' if v == 'v6w' else '')
        final[v] = dict(label=label, w=w, oof=quantile_combine(p_oof, blend(cb_oof, lg_oof, w), prev_tr),
                        test=quantile_combine(p_tst, blend(cb_tst, lg_tst, w), prev_te))

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
    v6_wins = bool(stats['v6']['wins all folds vs champion?'])

    # --- Error by bucket: V6 vs V4 (LightGBM alone alongside) ---
    e4 = bucket_table(y, champ)['error_sum']
    el = bucket_table(y, final['v6_lgbm']['oof'])['error_sum']
    e6 = bucket_table(y, final['v6']['oof'])['error_sum']
    tot = lambda e: [*e, e.sum()]  # noqa: E731
    bucket_cmp = pd.DataFrame({'bucket': [*BUCKETS, 'total'],
                               'count': [*bucket_table(y, base)['count'].map('{:,}'.format), f'{n:,}'],
                               'V4 error (M THB)': [f'{v / 1e6:.2f}' for v in tot(e4)],
                               'LightGBM alone (M THB)': [f'{v / 1e6:.2f}' for v in tot(el)],
                               'V6 error (M THB)': [f'{v / 1e6:.2f}' for v in tot(e6)],
                               'V6 − V4 (M THB)': [f'{(b - a) / 1e6:+.2f}' for a, b in zip(tot(e4), tot(e6))],
                               'V6 − V4 %': [f'{(b - a) / a:+.1%}' for a, b in zip(tot(e4), tot(e6))]})

    # --- Amount-model diagnostics: coverage, crossing, spenders-only score of α=0.5 ---
    truth = log_ratio(y[spend], train[spend])
    med = list(ALPHAS).index(0.5)
    cb_bl = blend(cb_oof, lg_oof, W_FIXED)
    cov = pd.DataFrame({'alpha': [f'{a:.2f}' for a in ALPHAS],
                        'CatBoost MQ (V4)': [f'{v:.3f}' for v in coverage(truth, cb_oof[spend])],
                        'LightGBM': [f'{v:.3f}' for v in coverage(truth, lg_oof[spend])],
                        f'blend w={W_FIXED} (V6)': [f'{v:.3f}' for v in coverage(truth, cb_bl[spend])]})
    spender_score = lambda lq: _score(y[spend], (prev_tr * np.exp(lq[:, med]))[spend])  # noqa: E731
    crossing = lambda lq: (np.diff(lq, axis=1) < 0).any(axis=1).mean()  # noqa: E731
    amount_models = {'CatBoost MultiQuantile (V4)': cb_oof, 'LightGBM quantile': lg_oof,
                     f'blend w={W_FIXED} (V6)': cb_bl}
    sp = {name: spender_score(lq) for name, lq in amount_models.items()}
    diag = pd.DataFrame([['baseline (single-stage, all of y)', fmt4(_score(y[spend], base[spend])), '—']]
                        + [[name, fmt4(sp[name]), f'{crossing(lq):.2%}'] for name, lq in amount_models.items()],
                        columns=['amount model', 'OOF score on y > 0 rows, α=0.5 column', 'rows with crossing quantiles'])
    scan = pd.DataFrame({'w (CatBoost share)': [f'{w:.1f}' for w in W_GRID], 'OOF': [fmt4(v) for v in w_scan],
                         'Δ vs V4 (w=1)': [f'{v - w_scan[1.0]:+.4f}' for v in w_scan]})
    iters = pd.DataFrame([[COMPONENT_NAMES[CLF], cap(1000), ' / '.join(map(str, best[CLF]))],
                          [COMPONENT_NAMES[AMT], cap(AMOUNTS[AMT][1]), ' / '.join(map(str, best[AMT]))]]
                         + [[LGB_NAME.format(a), cap(LGB_CAP), ' / '.join(map(str, best[a]))] for a in ALPHAS],
                         columns=['model', 'limit', 'best iteration per fold'])
    times = pd.DataFrame([[comp, *[f'{s:.0f}' for s in secs], f'{sum(secs):.0f}'] for comp, secs in seconds.items()],
                         columns=['step', 'fold1 (s)', 'fold2 (s)', 'fold3 (s)', 'total (s)'])
    total = time.time() - t_start

    print(f'\nTASK 3 V6 AMOUNT ENSEMBLE{" (QUICK: 250 trees)" if args.quick else ""}\n{shown.to_string(index=False)}')
    print(f'\nError by bucket, V6 vs V4\n{bucket_cmp.to_string(index=False)}')
    print(f'\nAmount models\n{diag.to_string(index=False)}')
    print(f'\nQuantile coverage, y > 0 rows (≈ alpha if calibrated)\n{cov.to_string(index=False)}')
    print(f'\nBlend weight scan (OOF)\n{scan.to_string(index=False)}')
    print(f'\nBest iterations\n{iters.to_string(index=False)}')
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

    # --- Promotion: V6 only (V6w is tuned on the OOF it is scored on) ---
    promoted = False
    if v6_wins and args.promote:
        r = stats['v6']
        promote('v6', final['v6']['label'], oof_frames['v6'], [r['fold1'], r['fold2'], r['fold3']], r['OOF'],
                champ_meta, script='experiments/task3/ensemble/ensemble.py')
        promoted = True
        decision = f'PROMOTED: V6 beats V4 on all 3 folds (OOF {r["OOF"]:.4f}, {r["Δ vs champion"]:+.4f}).'
    elif v6_wins:
        decision = 'Candidate: V6 beats V4 on all 3 folds; re-run with --promote to promote it.'
    else:
        decision = (f'STOP: V6 does not beat V4 on all 3 folds (Δ per fold {stats["v6"]["Δ per fold vs champion"]}); '
                    'champion stays V4, no refit, no Kaggle file.')
    print(f'\n{decision}')

    # --- ensemble_results.md (Interpretation kept) ---
    md = ['# Task 3 V6: ensembled amount model', '',
          '> **CAUTION:** Early stopping uses the validation fold (as in the baseline and V4) for both amount '
          'models, so every OOF score here is slightly optimistic; differences between rows are still comparable. '
          'V6w picks w on the same OOF it is scored on.', '',
          f'Generated by `python experiments/task3/ensemble/ensemble.py{" --promote" if args.promote else ""}`. '
          f'Folds `KFold({FOLDS}, shuffle=True, random_state={SEED})`; metric `_score` from `pis/tabular.py`. '
          'Classifier, features, `expected_spend` rule, target `log(y / prev)` + `prev_missing` and '
          '`quantile_combine` (s = 1, t = 0.5) are imported from `amount_model.py` unchanged (V4 re-run here: '
          f'{" / ".join(map(fmt4, v4_folds))}, reproduces the champion). New: one `LGBMRegressor('
          "objective='quantile', alpha=a)` per alpha in "
          f'{ALPHAS[0]:.2f}…{ALPHAS[-1]:.2f} on the same spender rows (num_leaves 63, lr 0.05, ≤ {LGB_CAP} trees, '
          f'early stopping {LGB_ES} on the validation fold\'s spenders, min_child_samples 50, subsample 0.8 / '
          'freq 1, colsample_bytree 0.8, seed 342 + fold; categoricals as pandas `category`). Blend per alpha on '
          'the log-ratio scale: `log_q = w · CatBoost + (1 − w) · LightGBM`, then sorted per row and combined '
          f'as in V4. V6: w = {W_FIXED} fixed in advance; V6w: best w on the OOF grid 0.0–1.0 step 0.1.', '',
          f'**Decision: {decision}**', '',
          '## Summary', '', to_markdown(shown), '',
          '## Error by bucket: V6 vs V4', '', to_markdown(bucket_cmp), '',
          '## Amount models', '',
          'Spenders-only score: each model\'s α=0.5 column (`prev × exp(...)`) on the y > 0 rows, i.e. stage 2 '
          'judged as if stage 1 were perfect.', '', to_markdown(diag), '',
          '### Quantile coverage', '',
          'Share of spenders whose true `log(y / prev)` is ≤ the predicted (row-sorted) quantile; ≈ alpha if '
          'calibrated.', '', to_markdown(cov), '',
          '### Blend weight scan (OOF-tuned, optimistic; w = 1 is V4, w = 0 is LightGBM alone)', '',
          to_markdown(scan), '',
          '## Best iterations', '', to_markdown(iters), '',
          '## Runtimes', '', to_markdown(times), '', f'Total {total:.0f}s.', '']
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')

    # --- ../LOG.md ---
    today, branch = datetime.date.today().isoformat(), current_branch()
    log_rows = []
    for v in ['v6_lgbm', 'v6', 'v6w']:
        r = stats[v]
        notes = []
        if v == 'v6_lgbm':
            notes.append('reference row, not a candidate')
        if v == 'v6w':
            notes.append(f'best w={best_w:.1f} on the grid 0.0–1.0 (OOF-tuned, optimistic, not a candidate)')
        notes.append(f'wins all folds vs baseline: {"yes" if r["wins all folds vs baseline?"] else "no"}')
        notes.append(f'vs champion (V4): {r["Δ vs champion"]:+.4f} ({r["Δ per fold vs champion"]}), wins all folds: '
                     f'{"yes" if r["wins all folds vs champion?"] else "no"}')
        if v == 'v6':
            cbs, lgs, bls = sp.values()
            notes.append(f'spenders-only α=0.5: blend {bls:.4f} vs CatBoost {cbs:.4f} vs LightGBM {lgs:.4f}')
            notes.append('**PROMOTED to champion**' if promoted else decision.rstrip('.'))
            notes.append(f'`ensemble/`; runtime {total:.0f}s')
        log_rows.append([today, branch, LOG_LABELS[v], ' / '.join(fmt4(r[f'fold{i}']) for i in (1, 2, 3)),
                         fmt4(r['OOF']), '—', '; '.join(notes)])
    upsert_log(log_rows)
    print(f'Wrote {RESULTS.relative_to(ROOT)}; updated experiments/task3/LOG.md')


if __name__ == '__main__':
    main()
