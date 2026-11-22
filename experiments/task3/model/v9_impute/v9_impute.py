"""Task 3 V9: label-free imputation of the spend-history features (protocol honest-es).

Why (experiments/task3/eda/missing_eda.ipynb): rows with prev_month_spending missing score
0.758 vs 0.794 for the rest, with the same target distribution, so the gap comes from the
model. V8 predicts log(y / prev); with prev missing it sets prev = 1 and has to learn log(y)
directly from ~6% of the training rows. prev can be imputed from the other features with
R² 0.995 (log1p), mostly from historical_spending, which those rows almost always have.
After imputation those rows use the same ratio target as everyone else.

Variants (everything else is V8: same folds, honest-es fit/es split, V4 classifier recipe,
V8's amount settings from tuning/v8.json, quantile combine, cap):
  - v9a: impute prev_month_spending only;
  - v9b: impute prev_month_spending, historical_spending and avg_transaction_value.

Imputer rules (no target, no test rows):
  - one HistGradientBoostingRegressor per imputed feature, on log1p(feature), from the other
    raw features (as in missing_eda.ipynb section 5); spending_30d is never an input;
  - fitted inside each outer fold on the outer-training rows only (never on validation or
    test rows), then applied to the outer-training, outer-validation and test rows;
  - only missing values are replaced; real values are untouched, so rows with a real prev
    keep exactly V8's ratio target;
  - the original missing flags (prev_missing, ...) stay as features of both stages.

Evaluation: V8 is re-run here with both seed schemes (amount random_seed 342 + fold and
1342 + fold) and checked against v8.json. A V9 variant is eligible only if it beats V8 on
all 3 folds in BOTH seed runs. --promote makes the eligible variant the champion (the
simpler V9a if both are eligible and V9b's OOF gain over V9a is <= 0.0003).

Run from the repo root:
    python experiments/task3/model/v9_impute/v9_impute.py --threads 12 [--promote]
--threads 12 matters: V8 was recorded with 12 CatBoost threads (see LOG.md).
Writes v9_impute_results.md next to this file (the hand-written Interpretation section is
kept), four rows in experiments/task3/LOG.md, and OOF/test predictions + classifier p to
outputs/experiments/task3_v9a/ and task3_v9b/ (gitignored). With --promote and an eligible
variant: champion/ and v9.json (the settings refit.py --variant v9 reads).
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# --- Make tuning/ (honest_es, v8_cv), ../amount_model and ../oracle importable ---
# The whole V8 pipeline is imported, not copied: V9 only swaps the per-fold inputs
# (Data.inputs) and reuses fit_classifier / run_pipeline as they are, so any difference
# to V8 comes from the imputation alone.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tuning'))
from honest_es import (CLF_ITERS, OUT, PROTOCOL, ROOT, SEED, Data, classifier_cache,  # noqa: E402
                       fit_classifier, honest_split, read_champion, run_pipeline, save_predictions,
                       write_champion)
from v8_cv import Steps  # noqa: E402
from amount_model import (INTERPRETATION, N_TEST, add_features, current_branch, current_commit,  # noqa: E402
                          fold_expected_spend, log_ratio, prev_or_one, set_threads, upsert_log)
from oracle import BUCKETS, FOLDS, PREV, TARGET, bucket_table, to_markdown  # noqa: E402
from pis.tabular import _score  # noqa: E402

# --- Settings ---
V8_JSON = HERE.parent / 'tuning' / 'v8.json'
V9_JSON = HERE / 'v9.json'
RESULTS = HERE / 'v9_impute_results.md'
SPEND_HISTORY = ['prev_month_spending', 'historical_spending', 'avg_transaction_value']
VARIANTS = {'v9a': SPEND_HISTORY[:1], 'v9b': SPEND_HISTORY}
NAMES = {'v9a': 'V9a: V8 + label-free imputation of prev_month_spending (in-fold, outer-train rows only)',
         'v9b': 'V9b: V8 + label-free imputation of prev / historical / avg_transaction_value (in-fold)'}
SEED_SCHEMES = [SEED, 1342]  # amount model random_seed = scheme + fold, as V8's two runs
TOL = 1e-4  # V8 re-run vs v8.json
SIMPLER_MARGIN = 0.0003  # V9b must beat V9a by more than this OOF to be chosen over it
RULE = ('promoted only if it beats the current champion on all 3 folds, within the same protocol, in both amount '
        'seed runs (342 + fold and 1342 + fold)')
FLAG = {PREV: 'prev_missing'}  # V8's flag name; the other features get '<feature>_missing'


# --- The imputer ---
# One model per imputed feature, fitted on rows where that feature is present. Inputs are
# the ORIGINAL raw features (NaN kept, HistGradientBoosting handles it natively), so in V9b
# the historical_spending imputer never sees an imputed prev: each imputer only learns from
# real values. Predictions are clipped to the range of real values seen in the fit rows;
# the lower clip also keeps log(prev) finite for the ratio target.

def raw_features(frame):
    return [c for c in frame.columns if c not in ('player_id', TARGET)]


def fit_imputers(fit_raw, cols):
    """{feature: (input columns, model, (lo, hi))}, fitted on `fit_raw` only (never the target)."""
    imputers = {}
    for c in cols:
        inputs = [f for f in raw_features(fit_raw) if f != c]
        assert TARGET not in inputs
        present = fit_raw[c].notna()
        model = HistGradientBoostingRegressor(random_state=SEED)
        model.fit(fit_raw.loc[present, inputs], np.log1p(fit_raw.loc[present, c]))
        imputers[c] = (inputs, model, (fit_raw.loc[present, c].min(), fit_raw.loc[present, c].max()))
    return imputers


def apply_imputers(imputers, raw):
    """Copy of `raw` with the missing values of each imputed feature filled; real values untouched."""
    out = raw.copy()
    for c, (inputs, model, (lo, hi)) in imputers.items():
        miss = raw[c].isna().to_numpy()
        if miss.any():
            out.loc[miss, c] = np.clip(np.expm1(model.predict(raw.loc[miss, inputs])), lo, hi)
    return out


def imputer_r2(imputers, raw):
    """Held-out R² (log1p) of each imputer on the rows of `raw` where the feature is present."""
    out = {}
    for c, (inputs, model, _) in imputers.items():
        present = raw[c].notna()
        out[c] = r2_score(np.log1p(raw.loc[present, c]), model.predict(raw.loc[present, inputs]))
    return out


def with_flags(xf, raw, cols):
    """Model features + one missing flag per imputed feature, from the ORIGINAL raw frame."""
    return xf.assign(**{FLAG.get(c, f'{c}_missing'): raw[c].isna().astype(int).to_numpy() for c in cols})


def model_x(x, raw, raw0, es, cols):
    """V9 model matrix (both stages): imputed values in `x`, V8's add_features on the imputed
    frame `raw`, then the missing flags from the original frame `raw0`. refit.py uses it too."""
    x = x.copy()
    x[cols] = raw[cols].to_numpy()  # same column names in the model matrix and the raw frame
    return with_flags(add_features(x, raw, es), raw0, cols)


# --- Per-fold inputs: V8's fold_inputs on imputed frames ---
# Same steps and same order as honest_es.fold_inputs; the differences are marked V9.
# Imputation is a preprocessing step: every downstream piece (model features, the
# expected_spend family, the log-ratio target, the prev used to turn quantiles back into
# THB) reads the imputed frame, and both stages get the missing flags.

def fold_inputs_v9(d, fold, tr, va, cols):
    tr_raw0, va_raw0 = d.train.iloc[tr], d.train.iloc[va]
    imputers = fit_imputers(tr_raw0, cols)  # V9: outer-training rows only
    tr_raw, va_raw, te_raw = (apply_imputers(imputers, f) for f in (tr_raw0, va_raw0, d.test))
    r2 = imputer_r2(imputers, va_raw0)  # held out: the validation fold's real values
    counts = {c: [int(f[c].isna().sum()) for f in (tr_raw0, va_raw0, d.test)] for c in cols}

    es_tr, (es_va, es_te) = fold_expected_spend(tr_raw, [va_raw, te_raw])
    x_tr = model_x(d.x.iloc[tr], tr_raw, tr_raw0, es_tr, cols)
    x_va = model_x(d.x.iloc[va], va_raw, va_raw0, es_va, cols)
    x_te = model_x(d.xt, te_raw, d.test, es_te, cols)
    y_tr = d.y[tr]
    spend_tr = y_tr > 0
    fit, es = honest_split(len(tr), fold, stratify=spend_tr)  # identical split to V8
    # Amount model: same log-ratio target, but the denominator is the imputed prev where prev
    # was missing (prev_or_one only falls back to 1 if a value is still missing).
    target = np.full(len(tr), np.nan)
    target[spend_tr] = log_ratio(y_tr[spend_tr], tr_raw[spend_tr])
    a_fit, a_es = fit[spend_tr[fit]], es[spend_tr[es]]
    return dict(fold=fold, tr=tr, va=va, fit=fit, es=es, x_tr=x_tr, x_va=x_va, x_te=x_te, spend_tr=spend_tr,
                a_fit_x=x_tr.iloc[a_fit], a_fit_y=target[a_fit], a_es_x=x_tr.iloc[a_es], a_es_y=target[a_es],
                a_va_x=x_va, a_te_x=x_te, y_tr_max=y_tr.max(), prev_va=prev_or_one(va_raw),
                prev_te=prev_or_one(te_raw), r2=r2, counts=counts)


class ImputedData(Data):
    """honest_es.Data whose per-fold inputs are imputed; run_pipeline uses it unchanged.

    Test rows get a different imputed prev from each fold's imputer. run_pipeline averages
    the fold models' log-quantiles and multiplies by one prev, so the test prev is the
    geometric mean of the three (the average on the same log scale as the quantiles).
    """

    def __init__(self, cols):
        super().__init__()
        self.cols = cols

    def inputs(self):
        if self._inputs is None:
            self._inputs = [fold_inputs_v9(self, fold, tr, va, self.cols) for fold, (tr, va) in enumerate(self.folds, 1)]
            self.prev_te = np.exp(np.mean([np.log(fi['prev_te']) for fi in self._inputs], axis=0))
        return self._inputs


def fit_classifiers(d):
    """V4's honest-es classifier (honest_es.fit_classifier) on the imputed inputs, seed SEED + fold."""
    p_oof, p_test, best, seconds = np.zeros(len(d.y)), np.zeros((FOLDS, N_TEST)), [], []
    for fi in d.inputs():
        p_va, p_te, b, s = fit_classifier(d, fi, CLF_ITERS)
        p_oof[fi['va']], p_test[fi['fold'] - 1] = p_va, p_te
        best.append(b)
        seconds.append(s)
    return dict(p_oof=p_oof, p_test=p_test, best=best, seconds=seconds)


# --- Report helpers ---

def fmt4(v):
    return f'{v:.4f}'


def per_fold(a, b):
    return ' / '.join(f'{x - y:+.4f}' for x, y in zip(a, b))


def wins(r, ref):
    return all(a > b for a, b in zip(r['folds'], ref['folds']))


def group_masks(train):
    """Row groups of missing_eda.ipynb: prev missing / present, 0-1 vs 2+ missing features."""
    n_missing = train[raw_features(train)].isna().sum(axis=1).to_numpy()
    prev_missing = train[PREV].isna().to_numpy()
    return {'all rows': np.ones(len(train), bool), 'prev missing': prev_missing, 'prev present': ~prev_missing,
            '0-1 missing features': n_missing <= 1, '2+ missing features': n_missing >= 2}


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--threads', type=int, default=-1,
                        help='CatBoost thread_count (default -1 = all cores); 12 reproduces V8 exactly')
    parser.add_argument('--promote', action='store_true', help='make the eligible V9 variant the champion')
    args = parser.parse_args()
    set_threads(args.threads)
    v8 = json.loads(V8_JSON.read_text(encoding='utf-8'))
    champ = read_champion()
    # V9 is always compared with V8 (re-run below). The champion is V8 before the promotion
    # and a V9 variant after it (a reproduction run): anything else means another model
    # took over and this comparison is no longer the right one.
    if champ.get('protocol') != PROTOCOL or champ.get('variant') not in ('v8', *VARIANTS):
        sys.exit(f'champion is {champ.get("variant")} on protocol {champ.get("protocol")}; V9 is compared with V8 (honest-es)')
    steps = Steps()

    # --- V8 reference: the same pipeline, both seed schemes, checked against v8.json ---
    # Re-running (instead of reading v8.json) puts V8 and V9 in the same environment
    # (CatBoost version, thread count); the check makes sure that environment reproduces V8.
    d8 = Data()
    clf8 = classifier_cache(d8)  # honest_es's cached V8 classifier p (built with 12 threads)
    if clf8['best'] != v8['classifier_best_iterations']:
        sys.exit(f'cached V8 classifier best iterations {clf8["best"]} != v8.json {v8["classifier_best_iterations"]}: '
                 'delete outputs/experiments/task3_v4_honest/classifier_p.npz and re-run with --threads 12')
    steps.done('V8: data, inputs, cached classifier p')
    res = {}
    for seed in SEED_SCHEMES:
        res[('v8', seed)] = run_pipeline(d8, clf8, v8['amount_params'], seed, v8['amount_iterations_limit'])
        steps.done(f'V8 amount model, seed {seed} + fold')
    for seed, key in [(SEED, 'fold_scores'), (1342, 'fold_scores_seed1342')]:
        deltas = [a - b for a, b in zip(res[('v8', seed)]['folds'], v8[key])]
        print(f'V8 seed {seed}: {" / ".join(f"{s:.6f}" for s in res[("v8", seed)]["folds"])} vs v8.json '
              f'{" / ".join(f"{s:.6f}" for s in v8[key])}', flush=True)
        if any(abs(x) > TOL for x in deltas):
            sys.exit(f'V8 seed {seed} does not reproduce v8.json within {TOL:g}: run with --threads 12')

    # --- V9a / V9b: imputed inputs, own classifier, V8's amount settings, both seeds ---
    data, clf = {}, {}
    for v, cols in VARIANTS.items():
        data[v] = ImputedData(cols)
        data[v].inputs()
        steps.done(f'{v}: data + imputers + fold inputs')
        clf[v] = fit_classifiers(data[v])
        steps.done(f'{v}: classifier (best iterations {"/".join(map(str, clf[v]["best"]))})')
        for seed in SEED_SCHEMES:
            res[(v, seed)] = run_pipeline(data[v], clf[v], v8['amount_params'], seed, v8['amount_iterations_limit'])
            r = res[(v, seed)]
            steps.done(f'{v} amount model, seed {seed} + fold: {" / ".join(map(fmt4, r["folds"]))}')

    # --- Eligibility: win all 3 folds vs V8 in BOTH seed runs ---
    eligible = [v for v in VARIANTS if all(wins(res[(v, s)], res[('v8', s)]) for s in SEED_SCHEMES)]
    chosen = None
    if eligible:
        chosen = eligible[0]
        if 'v9b' in eligible and 'v9a' in eligible and \
                res[('v9b', SEED)]['score'] - res[('v9a', SEED)]['score'] > SIMPLER_MARGIN:
            chosen = 'v9b'
    y = d8.y

    # --- Tables ---
    summary_rows = []
    for seed in SEED_SCHEMES:
        ref = res[('v8', seed)]
        for v in ['v8', *VARIANTS]:
            r = res[(v, seed)]
            is_ref = v == 'v8'
            summary_rows.append([v.upper(), f'{seed} + fold', *map(fmt4, r['folds']), fmt4(r['score']),
                                 '—' if is_ref else f'{r["score"] - ref["score"]:+.4f}',
                                 '—' if is_ref else per_fold(r['folds'], ref['folds']),
                                 '—' if is_ref else ('yes' if wins(r, ref) else 'no')])
    summary = pd.DataFrame(summary_rows, columns=['model', 'amount seed', 'fold1', 'fold2', 'fold3', 'OOF',
                                                  'Δ OOF vs V8', 'Δ per fold vs V8', 'wins all folds?'])
    groups = group_masks(d8.train)
    group_rows = []
    for name, m in groups.items():
        s = {v: _score(y[m], res[(v, SEED)]['oof'][m]) for v in ['v8', *VARIANTS]}
        group_rows.append([name, f'{m.sum():,}', *[fmt4(s[v]) for v in ['v8', *VARIANTS]],
                           *[f'{s[v] - s["v8"]:+.4f}' for v in VARIANTS]])
    group_table = pd.DataFrame(group_rows, columns=['group', 'rows', 'V8', 'V9a', 'V9b', 'Δ V9a', 'Δ V9b'])
    side = pd.DataFrame({'bucket': [*BUCKETS, 'total'],
                         'count': [*bucket_table(y, res[('v8', SEED)]['oof'])['count'].map('{:,}'.format), f'{len(y):,}']})
    e = {v: bucket_table(y, res[(v, SEED)]['oof'])['error_sum'] for v in ['v8', *VARIANTS]}
    for v in ['v8', *VARIANTS]:
        side[f'{v.upper()} error (M THB)'] = [f'{x / 1e6:.2f}' for x in [*e[v], e[v].sum()]]
    for v in VARIANTS:
        side[f'Δ {v.upper()} − V8'] = [f'{(b - a) / 1e6:+.2f}' for a, b in
                                       zip([*e['v8'], e['v8'].sum()], [*e[v], e[v].sum()])]
    imp_rows = []
    for v in VARIANTS:
        for fi in data[v].inputs():
            for c in VARIANTS[v]:
                n_tr, n_va, n_te = fi['counts'][c]
                imp_rows.append([v.upper(), fi['fold'], c, fmt4(fi['r2'][c]), f'{n_tr:,}', f'{n_va:,}', f'{n_te:,}'])
    imp_table = pd.DataFrame(imp_rows, columns=['variant', 'fold', 'feature', 'R² (log1p, val rows present)',
                                                'imputed: outer-train', 'imputed: validation', 'imputed: test'])
    iters = pd.DataFrame([[v.upper(), '/'.join(map(str, (clf8 if v == 'v8' else clf[v])['best'])),
                           *['/'.join(map(str, res[(v, s)]['best'])) for s in SEED_SCHEMES]]
                          for v in ['v8', *VARIANTS]],
                         columns=['model', 'classifier best iterations', 'amount best (seed 342)', 'amount best (seed 1342)'])
    time_table = pd.DataFrame(steps.rows, columns=['step', 'seconds'])
    time_table['seconds'] = time_table['seconds'].map('{:.0f}'.format)

    gains = '; '.join(f'{v.upper()} seed {s}: {per_fold(res[(v, s)]["folds"], res[("v8", s)]["folds"])}'
                      for v in VARIANTS for s in SEED_SCHEMES)
    if chosen:
        decision = (f'{chosen.upper()} beats V8 on all 3 folds in both seed runs (eligible: '
                    f'{", ".join(v.upper() for v in eligible)}; chosen: {chosen.upper()}). Per fold vs V8: {gains}.')
    else:
        decision = (f'STOP: neither V9a nor V9b beats V8 on all 3 folds in both seed runs; champion stays V8, '
                    f'no refit, no Kaggle file. Per fold vs V8: {gains}.')
    print(f'\nSummary\n{summary.to_string(index=False)}\n\nScore inside groups (seed 342 OOF)\n'
          f'{group_table.to_string(index=False)}\n\nError by bucket (seed 342)\n{side.to_string(index=False)}\n\n'
          f'Imputer R² per fold\n{imp_table.to_string(index=False)}\n\nBest iterations\n{iters.to_string(index=False)}'
          f'\n\n{decision}', flush=True)

    # --- Outputs (gitignored): predictions + classifier p of both variants, for refit.py ---
    for v in VARIANTS:
        folder = OUT / f'task3_{v}'
        save_predictions(data[v], folder, res[(v, SEED)])
        np.savez(folder / 'classifier_p.npz', p_oof=clf[v]['p_oof'], p_test=clf[v]['p_test'],
                 best=np.array(clf[v]['best']), seconds=np.array(clf[v]['seconds']))

    # --- Promotion (protocol honest-es) ---
    promoted = False
    if chosen and args.promote and champ['variant'] in VARIANTS:
        print(f'champion is already {champ["variant"].upper()}; not re-promoted')
    elif chosen and args.promote:
        r, r2 = res[(chosen, SEED)], res[(chosen, 1342)]
        meta = {'name': NAMES[chosen], 'variant': chosen, 'protocol': PROTOCOL, 'protocol_note': champ['protocol_note'],
                'script': 'experiments/task3/model/v9_impute/v9_impute.py', 'branch': current_branch(),
                'commit': current_commit(), 'commit_note': 'HEAD when the run started; the promotion itself lands in the next commit',
                'fold_scores': [round(s, 6) for s in r['folds']], 'oof': round(r['score'], 6),
                'fold_scores_seed1342': [round(s, 6) for s in r2['folds']], 'oof_seed1342': round(r2['score'], 6),
                'date': datetime.date.today().isoformat(), 'previous_champion': champ['name'],
                'previous_champion_fold_scores': champ['fold_scores'], 'previous_champion_oof': champ['oof'],
                'rule': RULE, 'previous_protocol': champ.get('previous_protocol'),
                'lb_history': champ.get('lb_history', []),
                **({'submission': champ['submission']} if 'submission' in champ else {})}
        write_champion(meta, pd.DataFrame({'player_id': d8.train['player_id'], 'target': y, 'prediction': r['oof']}))
        V9_JSON.write_text(json.dumps({
            'variant': chosen, 'imputed_features': VARIANTS[chosen], 'protocol': PROTOCOL,
            'imputer': 'HistGradientBoostingRegressor(random_state=342) on log1p(feature) from the other raw features, '
                       'fitted on the training rows only; values clipped to the observed range',
            'amount_params_from': 'experiments/task3/model/tuning/v8.json', 'amount_params': v8['amount_params'],
            'amount_iterations_limit': v8['amount_iterations_limit'], 'classifier_best_iterations': clf[chosen]['best'],
            'amount_best_iterations': r['best'], 'amount_best_iterations_seed1342': r2['best'],
            'fold_scores': [round(s, 6) for s in r['folds']], 'oof': round(r['score'], 6),
            'fold_scores_seed1342': [round(s, 6) for s in r2['folds']], 'oof_seed1342': round(r2['score'], 6),
            'threads': args.threads, 'date': datetime.date.today().isoformat()}, indent=2) + '\n', encoding='utf-8')
        promoted = True
        print(f'PROMOTED: {chosen.upper()} is the champion; wrote champion/ and {V9_JSON.relative_to(ROOT)}')
    elif chosen:
        print(f'{chosen.upper()} is eligible; re-run with --promote to promote it.')

    # --- v9_impute_results.md (Interpretation kept) and ../../LOG.md ---
    md = ['# Task 3 V9: label-free imputation of the spend-history features', '',
          f'Generated by `python experiments/task3/model/v9_impute/v9_impute.py --threads {args.threads}'
          f'{" --promote" if args.promote else ""}`. Protocol {PROTOCOL}, same folds, fit/es split, classifier recipe, '
          'amount settings (`tuning/v8.json`), quantile combine and cap as V8. Imputer: one '
          '`HistGradientBoostingRegressor(random_state=342)` per feature on `log1p(feature)` from the other raw features '
          '(never `spending_30d`), fitted inside each outer fold on the outer-training rows only and applied to the '
          'outer-training, validation and test rows; only missing values are replaced (clipped to the observed range). '
          'Every downstream piece reads the imputed frame: model features, the `expected_spend` family, the log-ratio '
          'target (so the imputed prev is the denominator only where prev was missing) and the prev that turns '
          'quantiles back into THB (test: geometric mean of the 3 fold imputers). Both stages get the original missing '
          'flags (`prev_missing`, and in V9b `historical_spending_missing`, `avg_transaction_value_missing`); the '
          'classifier is refitted on these inputs with V4\'s recipe. V8 is re-run in the same script and matches '
          f'`v8.json` within {TOL:g} in both seed runs.', '',
          '> **NOTE:** Rule fixed before the run: a variant is eligible only if it beats V8 on all 3 folds with BOTH '
          f'amount seed schemes (342 + fold and 1342 + fold); if both are eligible, V9b is chosen only if its OOF beats '
          f'V9a by more than {SIMPLER_MARGIN}.', '',
          f'**Decision: {decision}**' + (' **Promoted.**' if promoted else ''), '',
          '## Summary', '', to_markdown(summary), '',
          '## Score inside row groups (seed 342 OOF)', '',
          'Groups as in `experiments/task3/eda/missing_eda.ipynb` (missing = in the original train).', '',
          to_markdown(group_table), '',
          '## Error by bucket (seed 342 OOF, million THB)', '', to_markdown(side), '',
          '## Imputer R² per fold', '',
          'R² on `log1p` of the feature, on the validation fold\'s rows where it is present (the imputer never saw them); '
          'counts = values imputed in each part of the fold.', '', to_markdown(imp_table), '',
          '## Best iterations', '', to_markdown(iters), '',
          '## Runtimes', '', f'CatBoost threads: {"all cores" if args.threads == -1 else args.threads}.', '',
          to_markdown(time_table), '', f'Total {time.time() - steps.t0:.0f}s.', '']
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')

    today, branch = datetime.date.today().isoformat(), current_branch()
    rows = []
    for v in VARIANTS:
        for seed in SEED_SCHEMES:
            r, ref = res[(v, seed)], res[('v8', seed)]
            g = groups['prev missing']
            note = (f'protocol {PROTOCOL}; amount seed {seed} + fold; vs V8 same seed: {per_fold(r["folds"], ref["folds"])} '
                    f'(OOF {r["score"] - ref["score"]:+.4f}, wins all folds: {"yes" if wins(r, ref) else "no"}); '
                    f'prev-missing rows {_score(y[g], r["oof"][g]):.4f} vs V8 {_score(y[g], ref["oof"][g]):.4f}; '
                    f'imputer R² {min(fi["r2"][PREV] for fi in data[v].inputs()):.3f}+ (prev); '
                    f'{"eligible" if v in eligible else "not eligible"}'
                    + ('; **promoted**' if promoted and v == chosen and seed == SEED else '')
                    + '; `model/v9_impute/`')
            rows.append([today, branch, f'{NAMES[v]}, amount seed {seed} + fold', ' / '.join(map(fmt4, r['folds'])),
                         fmt4(r['score']), '—', note])
    upsert_log(rows)
    print(f'Wrote {RESULTS.relative_to(ROOT)}; updated experiments/task3/LOG.md')
    print(steps.summary())


if __name__ == '__main__':
    main()
