"""Task 3 honest early stopping ("honest-es"): V4 and the baseline re-scored without peeking.

Until now every model early-stopped on the outer validation fold that is also scored
(`eval_set=(x_val, y_val), use_best_model=True`, as in pis/tabular.py). The best iteration
is then picked on the scored rows, so every OOF score is slightly optimistic, and tuning
hyper-parameters on such scores would partly fit that noise. Protocol "honest-es":

  - in each outer fold, the outer-training part is split ONCE into fit (85%) and es (15%)
    with seed SEED + fold (stratified on y > 0 for two-stage models, plain for the
    single-stage baseline);
  - models train on fit and early-stop on es; the outer validation fold is only scored;
  - expected_spend for fit/es rows is the inner-OOF value computed on the FULL
    outer-training part (amount_model.fold_expected_spend, unchanged); validation and
    test rows use multipliers fitted on the full outer-training part, as before;
  - final predictions are clipped with amount_model.cap_predictions (the outer-training
    part's max target, 300,000).

Run from the repo root:
    python experiments/task3/model/tuning/honest_es.py [--quick] [--record]
Re-runs V4 ("V4-honest") and the baseline single-stage CatBoost ("baseline-honest") under
honest-es, prints them next to their old (val-ES) scores, writes the honest-es section of
experiments/task3/model/tuning/tuning_results.md, upserts two rows in experiments/task3/LOG.md and
saves OOF/test predictions plus the per-fold classifier p (reused, fixed, by tuning.py) to
outputs/experiments/task3_v4_honest/ (gitignored). --record makes V4-honest the champion
reference of the new protocol (champion.json gets "protocol": "honest-es" and keeps the old
numbers in "previous_protocol"); from then on, compare only within the same protocol.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold, train_test_split

# --- Make `pis`, ../amount_model/amount_model.py and ../oracle/oracle.py importable ---
# Every piece of the V4 recipe (features, expected_spend rule, target, quantile combine,
# CatBoost settings, cap) is imported, not copied, so V4-honest differs from V4 only in
# where early stopping looks.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
from amount_model import (AMOUNTS, CHAMPION, CLASSIFIERS, MQ_LOSS, N_TEST, OUT, TWO_STAGE,  # noqa: E402
                          add_features, cap_predictions, cb_params, current_branch, current_commit,
                          fold_expected_spend, log_ratio, prev_or_one, quantile_combine, upsert_log,
                          with_prev_flag)
from oracle import (BUCKETS, EXPECTED_FOLDS, EXPECTED_OOF, FOLDS, SEED, TARGET,  # noqa: E402
                    bucket_table, to_markdown)
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402

# --- Settings ---
PROTOCOL = 'honest-es'
OLD_PROTOCOL = 'val-es'  # early stopping on the scored validation fold (pis/tabular.py and steps 1-13)
ES_SHARE = 0.15  # share of the outer-training part held out for early stopping
CLF, AMT = TWO_STAGE['v4']
CLF_ITERS, AMT_ITERS = 1000, AMOUNTS[AMT][1]  # V4's limits (classifier 1000, amount 3000)
TUNING = Path(__file__).resolve().parent
RESULTS_MD = TUNING / 'tuning_results.md'
OUT_DIR = OUT / 'task3_v4_honest'
INTERPRETATION = '## Interpretation'
PROTOCOL_NOTE = (f'early stopping on a fixed {1 - ES_SHARE:.0%}/{ES_SHARE:.0%} fit/es split of each outer-training '
                 f'part (seed {SEED} + fold, stratified on y > 0 for two-stage models); the outer validation '
                 'fold is only scored; predictions capped at the outer-training max target')
LABELS = {'v4_honest': 'V4-honest: V4 recipe under honest-es (fit/es 85/15 split of the outer-train part, '
                       'stratified on y > 0), cap',
          'baseline_honest': 'Baseline-honest: baseline CatBoost MAE under honest-es (plain 85/15 fit/es split), cap'}


# --- Data and folds (same filter, preprocessing, folds and checks as amount_model.py) ---

class Data:
    """Train/test frames, model matrices, target, outer folds and the baseline OOF."""

    def __init__(self):
        self.train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
        self.test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
        self.train = self.train.loc[self.train[TARGET].notna()].reset_index(drop=True)
        self.x, self.xt, self.categorical = _prepare(self.train, self.test, TARGET)
        self.y = self.train[TARGET].to_numpy(dtype=float)
        self.folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(self.x))
        oof_base = pd.read_csv(ROOT / 'baseline' / 'task3_oof.csv')
        assert (self.train['player_id'].to_numpy() == oof_base['player_id'].to_numpy()).all(), 'row order differs'
        self.base = oof_base['prediction'].to_numpy(dtype=float)
        base_folds = self.fold_scores(self.base)
        if abs(_score(self.y, self.base) - EXPECTED_OOF) > 1e-4 or \
                any(abs(a - b) > 1e-4 for a, b in zip(base_folds, EXPECTED_FOLDS)):
            sys.exit(f'Baseline OOF does not reproduce the published fold scores: {base_folds}')
        self.prev_tr, self.prev_te = prev_or_one(self.train), prev_or_one(self.test)
        self._inputs = None

    def fold_scores(self, pred):
        return [_score(self.y[va], pred[va]) for _, va in self.folds]

    def inputs(self):
        """Per-fold V4 model inputs (built once; reused by every trial in tuning.py)."""
        if self._inputs is None:
            self._inputs = [fold_inputs(self, fold, tr, va) for fold, (tr, va) in enumerate(self.folds, 1)]
        return self._inputs


def honest_split(n, fold, stratify=None):
    """(fit, es) positions within an outer-training part of n rows: one fixed split per fold."""
    return train_test_split(np.arange(n), test_size=ES_SHARE, random_state=SEED + fold, stratify=stratify)


def fold_inputs(d, fold, tr, va):
    """Features of one outer fold for the two-stage models, with the fit/es split.

    expected_spend: inner-OOF on the FULL outer-training part (fit and es rows alike),
    multipliers fitted on the full outer-training part for validation and test rows.
    """
    tr_raw, va_raw = d.train.iloc[tr], d.train.iloc[va]
    es_tr, (es_va, es_te) = fold_expected_spend(tr_raw, [va_raw, d.test])
    x_tr = add_features(d.x.iloc[tr], tr_raw, es_tr)
    x_va = add_features(d.x.iloc[va], va_raw, es_va)
    x_te = add_features(d.xt, d.test, es_te)
    y_tr = d.y[tr]
    spend_tr = y_tr > 0
    fit, es = honest_split(len(tr), fold, stratify=spend_tr)
    # Amount model (stage 2): spenders only, target log(y / prev), + prev_missing flag.
    a_tr = with_prev_flag(x_tr, tr_raw)
    target = np.full(len(tr), np.nan)
    target[spend_tr] = log_ratio(y_tr[spend_tr], tr_raw[spend_tr])
    a_fit, a_es = fit[spend_tr[fit]], es[spend_tr[es]]
    return dict(fold=fold, tr=tr, va=va, fit=fit, es=es, x_tr=x_tr, x_va=x_va, x_te=x_te, spend_tr=spend_tr,
                a_fit_x=a_tr.iloc[a_fit], a_fit_y=target[a_fit], a_es_x=a_tr.iloc[a_es], a_es_y=target[a_es],
                a_va_x=with_prev_flag(x_va, va_raw), a_te_x=with_prev_flag(x_te, d.test), y_tr_max=y_tr.max(),
                prev_va=d.prev_tr[va])


# --- Stage 1: V4's classifier under honest-es ---

def fit_classifier(d, fi, iterations=CLF_ITERS):
    """V4 classifier on fit rows, early-stopped on es rows -> (p on va, p on test, best iteration, seconds)."""
    t0 = time.time()
    fit, es, spend = fi['fit'], fi['es'], fi['spend_tr'].astype(int)
    clf = CatBoostClassifier(loss_function='Logloss', **cb_params(iterations, fi['fold'], **CLASSIFIERS[CLF]))
    clf.fit(fi['x_tr'].iloc[fit], spend[fit], cat_features=d.categorical,
            eval_set=(fi['x_tr'].iloc[es], spend[es]), use_best_model=True)
    return (clf.predict_proba(fi['x_va'])[:, 1], clf.predict_proba(fi['x_te'])[:, 1],
            clf.get_best_iteration() + 1, time.time() - t0)


def classifier_cache(d, quick=False):
    """Per-fold honest-es classifier p (OOF and test), computed once and kept fixed.

    tuning.py tunes only the amount model; the classifier was shown not to benefit from
    tuning (V3b, PR #8), so every trial reuses exactly these p.
    """
    path = (OUT_DIR / 'quick' if quick else OUT_DIR) / 'classifier_p.npz'
    if path.exists():
        z = np.load(path)
        return dict(p_oof=z['p_oof'], p_test=z['p_test'], best=z['best'].tolist(), seconds=z['seconds'].tolist())
    p_oof, p_test, best, seconds = np.zeros(len(d.y)), np.zeros((FOLDS, N_TEST)), [], []
    for fi in d.inputs():
        p_va, p_te, b, s = fit_classifier(d, fi, 250 if quick else CLF_ITERS)
        p_oof[fi['va']], p_test[fi['fold'] - 1] = p_va, p_te
        best.append(b)
        seconds.append(s)
        print(f'classifier fold {fi["fold"]}: best iteration {b}, {s:.0f}s', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, p_oof=p_oof, p_test=p_test, best=np.array(best), seconds=np.array(seconds))
    return dict(p_oof=p_oof, p_test=p_test, best=best, seconds=seconds)


# --- Stage 2: the MultiQuantile amount model, and the full V4 pipeline ---

def fit_amount(d, fi, params=None, seed=None, iterations=AMT_ITERS):
    """MultiQuantile log-ratio model on fit spenders, early-stopped on es spenders.

    `params` override V4's CatBoost settings (cb_params); `seed` overrides random_seed
    (default SEED + fold). Returns (log-quantiles on va, on test, best iteration, seconds).
    """
    t0 = time.time()
    p = cb_params(iterations, fi['fold'], **(params or {}))
    if seed is not None:
        p['random_seed'] = seed
    reg = CatBoostRegressor(loss_function=MQ_LOSS, **p)
    reg.fit(fi['a_fit_x'], fi['a_fit_y'], cat_features=d.categorical,
            eval_set=(fi['a_es_x'], fi['a_es_y']), use_best_model=True)
    return reg.predict(fi['a_va_x']), reg.predict(fi['a_te_x']), reg.get_best_iteration() + 1, time.time() - t0


def run_pipeline(d, clf, params=None, seed_base=SEED, iterations=AMT_ITERS, on_fold=None):
    """Full V4 pipeline (fixed classifier p + amount model with `params`) under honest-es.

    OOF: per fold, cap(quantile_combine(p, q, prev), outer-train max). Test: p and the
    log-quantiles averaged over the 3 fold models first, then combined and capped at the
    full-train max (as amount_model.py does). `on_fold(fold, score)` is called after each
    fold (tuning.py uses it to prune). Returns a dict with predictions, scores, iterations.
    """
    oof, oof_raw, q_test = np.zeros(len(d.y)), np.zeros(len(d.y)), 0.0
    best, seconds, scores = [], [], []
    for fi in d.inputs():
        q_va, q_te, b, s = fit_amount(d, fi, params, seed_base + fi['fold'], iterations)
        va = fi['va']
        oof_raw[va] = quantile_combine(clf['p_oof'][va], q_va, fi['prev_va'])
        oof[va] = cap_predictions(oof_raw[va], d.y[fi['tr']])
        q_test = q_test + q_te / FOLDS
        best.append(b)
        seconds.append(s)
        scores.append(_score(d.y[va], oof[va]))
        if on_fold is not None:
            on_fold(fi['fold'], scores[-1])
    test = cap_predictions(quantile_combine(clf['p_test'].mean(axis=0), q_test, d.prev_te), d.y)
    return dict(oof=oof, oof_raw=oof_raw, test=test, folds=scores, score=_score(d.y, oof), best=best,
                seconds=seconds, n_capped_oof=int((oof != oof_raw).sum()))


def run_baseline(d, quick=False):
    """pis/tabular.py's CatBoost MAE (same params, same features) under honest-es, capped."""
    oof, test, best, seconds = np.zeros(len(d.y)), np.zeros(N_TEST), [], []
    for fold, (tr, va) in enumerate(d.folds, 1):
        t0 = time.time()
        fit, es = (tr[pos] for pos in honest_split(len(tr), fold))
        model = CatBoostRegressor(loss_function='MAE', **cb_params(250 if quick else 1000, fold))
        model.fit(d.x.iloc[fit], d.y[fit], cat_features=d.categorical, eval_set=(d.x.iloc[es], d.y[es]),
                  use_best_model=True)
        oof[va] = cap_predictions(model.predict(d.x.iloc[va]), d.y[tr])
        test += model.predict(d.xt) / FOLDS
        best.append(model.get_best_iteration() + 1)
        seconds.append(time.time() - t0)
        print(f'baseline fold {fold}: best iteration {best[-1]}, {seconds[-1]:.0f}s', flush=True)
    return dict(oof=oof, test=cap_predictions(test, d.y), folds=d.fold_scores(oof), score=_score(d.y, oof),
                best=best, seconds=seconds)


# --- Champion under the new protocol ---

def read_champion():
    return json.loads((CHAMPION / 'champion.json').read_text(encoding='utf-8'))


def champion_protocol(meta):
    return meta.get('protocol', OLD_PROTOCOL)


def write_champion(meta, oof_frame):
    """champion/task3_oof.csv (rounded to satang, as amount_model.promote) + champion.json."""
    oof_frame.round({'target': 2, 'prediction': 2}).to_csv(CHAMPION / 'task3_oof.csv', index=False)
    (CHAMPION / 'champion.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def record_reference(old, result, script):
    """Make V4-honest the reference of the honest-es protocol; the val-es numbers move to previous_protocol."""
    keep = ['name', 'variant', 'script', 'branch', 'commit', 'fold_scores', 'oof', 'date']
    previous = {'protocol': OLD_PROTOCOL,
                'protocol_note': 'early stopping on the scored outer validation fold (slightly optimistic OOF)',
                **{k: old[k] for k in keep if k in old}}
    return {'name': 'V4-honest: V4 recipe under honest early stopping', 'variant': 'v4_honest',
            'protocol': PROTOCOL, 'protocol_note': PROTOCOL_NOTE, 'script': script,
            'branch': current_branch(), 'commit': current_commit(),
            'commit_note': 'HEAD when the run started; the protocol switch lands in the next commit',
            'fold_scores': [round(s, 6) for s in result['folds']], 'oof': round(result['score'], 6),
            'date': datetime.date.today().isoformat(),
            'previous_champion': f'{old["name"]} (same recipe, {OLD_PROTOCOL} protocol)',
            'rule': 'promoted only if it beats the current champion on all 3 folds, within the same protocol',
            'previous_protocol': previous, 'lb_history': old.get('lb_history', []),
            **({'submission': old['submission']} if 'submission' in old else {})}


def save_predictions(d, folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({'player_id': d.train['player_id'], 'target': d.y, 'prediction': result['oof']})
    oof.to_csv(folder / 'task3_oof.csv', index=False)
    frame = pd.DataFrame({'player_id': d.test['player_id'], 'prediction': result['test']})
    assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
    frame.to_csv(folder / 'task3_predictions.csv', index=False)
    return oof


# --- tuning_results.md: each script owns one marked section; the interpretation is kept ---

def write_section(name, lines):
    start, end = f'<!-- {name}:start -->', f'<!-- {name}:end -->'
    text = RESULTS_MD.read_text(encoding='utf-8') if RESULTS_MD.exists() else '\n'.join([
        '# Task 3 tuning: honest early stopping + Optuna on the amount model', '',
        'Generated sections are rewritten by `honest_es.py` and `tuning.py`; the interpretation at the end '
        'is written by hand and kept across re-runs.', '',
        '<!-- honest-es:start -->', '<!-- honest-es:end -->', '', '<!-- optuna:start -->', '<!-- optuna:end -->',
        '', INTERPRETATION, '', '_TODO_', ''])
    head, rest = text.split(start)
    _, tail = rest.split(end)
    RESULTS_MD.write_text(head + start + '\n' + '\n'.join(lines).strip('\n') + '\n' + end + tail, encoding='utf-8')


def fmt4(v):
    return f'{v:.4f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--quick', action='store_true', help='250 iterations for every model (smoke test)')
    parser.add_argument('--record', action='store_true',
                        help='make V4-honest the champion reference of the honest-es protocol')
    args = parser.parse_args()
    if args.record and args.quick:
        parser.error('--record needs a full run, not --quick')
    t_start = time.time()
    d = Data()
    champ = read_champion()
    old = champ if champion_protocol(champ) == OLD_PROTOCOL else champ['previous_protocol']
    if old['variant'] != 'v4':
        sys.exit(f'the {OLD_PROTOCOL} reference is {old["variant"]}; this script re-runs v4')

    # --- V4-honest: fixed classifier p + V4's amount model, both early-stopped on es ---
    clf = classifier_cache(d, args.quick)
    v4 = run_pipeline(d, clf, iterations=250 if args.quick else AMT_ITERS)
    print(f'V4-honest: {" / ".join(map(fmt4, v4["folds"]))}, OOF {v4["score"]:.4f} '
          f'(uncapped OOF {_score(d.y, v4["oof_raw"]):.4f}, {v4["n_capped_oof"]} OOF rows capped)', flush=True)
    base = run_baseline(d, args.quick)
    print(f'Baseline-honest: {" / ".join(map(fmt4, base["folds"]))}, OOF {base["score"]:.4f}', flush=True)

    # --- Tables: honest vs old protocol, and the honest baseline -> V4 gain ---
    old_v4, old_base = (old['fold_scores'], old['oof']), (EXPECTED_FOLDS, EXPECTED_OOF)
    old_base_exact = d.fold_scores(d.base), _score(d.y, d.base)
    rows = []
    for name, (of, oo), new in [('Baseline (single-stage CatBoost MAE)', old_base_exact, base),
                                ('V4 (quantile combine)', old_v4, v4)]:
        rows.append([name, OLD_PROTOCOL, *map(fmt4, of), fmt4(oo), '—', '—'])
        rows.append([name, PROTOCOL, *map(fmt4, new['folds']), fmt4(new['score']),
                     f'{new["score"] - oo:+.4f}', ' / '.join(f'{a - b:+.5f}' for a, b in zip(new['folds'], of))])
    compare = pd.DataFrame(rows, columns=['model', 'protocol', 'fold1', 'fold2', 'fold3', 'OOF',
                                          'Δ OOF vs val-es', 'Δ per fold vs val-es'])
    gain_rows = pd.DataFrame([[PROTOCOL, f'{v4["score"] - base["score"]:+.4f}',
                               ' / '.join(f'{a - b:+.4f}' for a, b in zip(v4['folds'], base['folds']))],
                              [OLD_PROTOCOL, f'{old_v4[1] - old_base_exact[1]:+.4f}',
                               ' / '.join(f'{a - b:+.4f}' for a, b in zip(old_v4[0], old_base_exact[0]))]],
                             columns=['protocol', 'V4 − baseline (OOF)', 'per fold'])
    iters = pd.DataFrame([['classifier (V4)', CLF_ITERS, ' / '.join(map(str, clf['best']))],
                          ['amount MultiQuantile (V4)', AMT_ITERS, ' / '.join(map(str, v4['best']))],
                          ['baseline CatBoost MAE', 1000, ' / '.join(map(str, base['best']))]],
                         columns=['model', 'limit', f'best iteration per fold ({PROTOCOL})'])
    side = pd.DataFrame({'bucket': BUCKETS, 'count': bucket_table(d.y, d.base)['count'].map('{:,}'.format).to_numpy()})
    champ_oof = pd.read_csv(CHAMPION / 'task3_oof.csv')['prediction'].to_numpy(dtype=float)
    for label, pred in [('baseline (val-es)', d.base), ('baseline-honest', base['oof']),
                        ('V4 (val-es)', champ_oof if champion_protocol(champ) == OLD_PROTOCOL else None),
                        ('V4-honest', v4['oof'])]:
        if pred is not None:
            side[label] = (bucket_table(d.y, pred)['error_sum'] / 1e6).map('{:.1f}'.format).to_numpy()
    times = pd.DataFrame([['classifier (V4)', *[f'{s:.0f}' for s in clf['seconds']]],
                          ['amount MultiQuantile (V4)', *[f'{s:.0f}' for s in v4['seconds']]],
                          ['baseline CatBoost MAE', *[f'{s:.0f}' for s in base['seconds']]]],
                         columns=['model', 'fold1 s', 'fold2 s', 'fold3 s'])
    print(f'\n{compare.to_string(index=False)}\n\n{gain_rows.to_string(index=False)}\n\n{iters.to_string(index=False)}'
          f'\n\nError by bucket (million THB)\n{side.to_string(index=False)}\n\n{times.to_string(index=False)}')

    if args.quick:
        print('\nquick run: nothing written (no LOG, no results section, no champion change)')
        return
    save_predictions(d, OUT_DIR, v4)
    save_predictions(d, OUT / 'task3_baseline_honest', base)

    # --- Champion: V4-honest becomes the reference of the honest-es protocol ---
    recorded = False
    if args.record:
        if champion_protocol(champ) != OLD_PROTOCOL:
            print(f'champion.json is already on protocol {champion_protocol(champ)}; not re-recorded')
        else:
            oof_frame = pd.DataFrame({'player_id': d.train['player_id'], 'target': d.y, 'prediction': v4['oof']})
            write_champion(record_reference(champ, v4, 'experiments/task3/model/tuning/honest_es.py'), oof_frame)
            recorded = True
            print('Recorded V4-honest as the champion reference of protocol honest-es; commit champion/ with LOG.md.')

    runtime = time.time() - t_start
    write_section('honest-es', [
        '## Part 1: honest early stopping (honest-es)', '',
        f'Generated by `python experiments/task3/model/tuning/honest_es.py{" --record" if args.record else ""}`. '
        f'Protocol: {PROTOCOL_NOTE}. `expected_spend` for fit/es rows is the inner `KFold(5)` OOF value computed on the '
        'full outer-training part (unchanged), so only early stopping moves. The old protocol (`val-es`) early-stopped '
        'on the scored validation fold.', '',
        '### V4 and baseline: old protocol vs honest-es', '', to_markdown(compare), '',
        f'V4-honest with the cap: {v4["n_capped_oof"]} OOF rows capped; OOF uncapped '
        f'{_score(d.y, v4["oof_raw"]):.4f} vs capped {v4["score"]:.4f}.', '',
        '### Baseline → V4 gain, within each protocol', '', to_markdown(gain_rows), '',
        '### Best iterations (honest-es)', '', to_markdown(iters), '',
        '### Error by bucket (OOF, million THB)', '', to_markdown(side), '',
        '### Runtimes', '', to_markdown(times), '', f'Total run: {runtime:.0f}s.', ''])
    print(f'Wrote the honest-es section of {RESULTS_MD.relative_to(ROOT)}')

    today, branch = datetime.date.today().isoformat(), current_branch()
    note_v4 = (f'protocol {PROTOCOL}; old V4 ({OLD_PROTOCOL}) {" / ".join(map(fmt4, old_v4[0]))}, OOF {old_v4[1]:.4f}; '
               f'Δ {v4["score"] - old_v4[1]:+.4f} ({" / ".join(f"{a - b:+.4f}" for a, b in zip(v4["folds"], old_v4[0]))}); '
               f'best iterations clf {"/".join(map(str, clf["best"]))}, amount {"/".join(map(str, v4["best"]))}; '
               f'cap: {v4["n_capped_oof"]} OOF rows ({_score(d.y, v4["oof_raw"]):.4f} uncapped)'
               + ('; **recorded in champion.json as the reference of protocol honest-es** (compare only within one '
                  'protocol from here on)' if recorded else ''))
    note_base = (f'protocol {PROTOCOL}; old baseline ({OLD_PROTOCOL}) {old_base_exact[1]:.4f}, Δ '
                 f'{base["score"] - old_base_exact[1]:+.4f}; honest baseline → V4-honest gain '
                 f'{v4["score"] - base["score"]:+.4f} ({" / ".join(f"{a - b:+.4f}" for a, b in zip(v4["folds"], base["folds"]))}) '
                 f'vs {old_v4[1] - old_base_exact[1]:+.4f} under {OLD_PROTOCOL}; best iterations {"/".join(map(str, base["best"]))}; '
                 f'`tuning/honest_es.py`, run {runtime:.0f}s')
    upsert_log([[today, branch, LABELS['v4_honest'], ' / '.join(map(fmt4, v4['folds'])), fmt4(v4['score']), '—', note_v4],
                [today, branch, LABELS['baseline_honest'], ' / '.join(map(fmt4, base['folds'])), fmt4(base['score']),
                 '—', note_base]])
    print('Updated experiments/task3/LOG.md')


if __name__ == '__main__':
    main()
