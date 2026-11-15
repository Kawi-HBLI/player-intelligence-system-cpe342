"""Task 3 amount model: better "how much?" stage (V1 features, V2/V3 two-stage, V3 tuning).

The oracle test showed the amount stage is the main bottleneck (perfect amount +0.148
vs perfect zeros +0.054), and the EDA showed spenders' y ≈ prev_month_spending × a
segment×VIP multiplier. This script turns that into a leak-free feature and tries it
in a single-stage model (V1) and in two-stage models (V2: amount in THB, V3: amount
as a log-ratio to prev_month_spending). V3a/V3b tune V3 one change at a time.

Run from the repo root:
    python experiments/task3/amount_model/amount_model.py [--variants v1,v2,v3,v3a,v3b] [--quick] [--promote]
Prints the summary, error-by-bucket and stage diagnostics; rewrites the tables in
experiments/task3/amount_model/amount_model_results.md (the hand-written interpretation
is kept); upserts one row per new variant in experiments/task3/LOG.md; writes OOF and test
predictions to outputs/experiments/task3_<variant>/ (gitignored). Every variant is compared
fold by fold against the baseline AND the current champion (experiments/task3/champion/);
--promote makes the best variant that beats the champion on all folds the new champion.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# --- Make `pis` and ../oracle/oracle.py importable no matter where the script is launched from ---
# Reuse the project's data path, preprocessing and metric, and the oracle's multiplier
# code, so the expected_spend feature is exactly the rule scored as scenario D there.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare, _score  # noqa: E402
from oracle import (BUCKETS, EXPECTED_FOLDS, EXPECTED_OOF, FOLDS, PREV, SEED, TARGET,  # noqa: E402
                    bucket_table, fit_multipliers, fmt_bucket, lookup, to_markdown)

# --- Settings ---
# Outer folds, seed and CatBoost settings match pis/tabular.py so every score is
# directly comparable with the baseline fold by fold. INNER_FOLDS only splits the
# outer-training part to build expected_spend for the training rows (see below).
INNER_FOLDS = 5
LEVELS = [['segment', 'vip_status'], ['segment'], []]  # fallback order: segment×VIP -> segment -> global
GRID = np.round(np.arange(0.30, 0.70 + 1e-9, 0.025), 3)  # thresholds tried for p(spend)
N_TEST = 25_889
TASK3 = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).with_name('amount_model_results.md')
LOG_FILE = TASK3 / 'LOG.md'  # one log for all Task 3 experiments
CHAMPION = TASK3 / 'champion'  # task3_oof.csv + champion.json of the current best model
OUT = ROOT / 'outputs' / 'experiments'
INTERPRETATION = '## Interpretation'

# --- Variants ---
# A two-stage variant = (stage-1 classifier, stage-2 amount model). Components are trained
# once per fold and shared, so a variant that changes one thing reuses everything else:
# V3a only raises the amount model's cap (V3's hit 972-1000 of 1000), and V3b keeps V3a's
# amount model exactly and changes only the classifier. Overrides are on top of cb_params().
VARIANTS = ['v1', 'v2', 'v3', 'v3a', 'v3b']
CLASSIFIERS = {'clf': {}, 'clf_deep': dict(depth=8, learning_rate=0.03)}
AMOUNTS = {'thb': ('thb', 1000), 'logratio': ('logratio', 1000), 'logratio_3k': ('logratio', 3000)}
TWO_STAGE = {'v2': ('clf', 'thb'), 'v3': ('clf', 'logratio'),
             'v3a': ('clf', 'logratio_3k'), 'v3b': ('clf_deep', 'logratio_3k')}
NAMES = {'v1': 'V1 single-stage + features',
         'v2': 'V2 two-stage, THB amount',
         'v3': 'V3 two-stage, log-ratio amount',
         'v3a': 'V3a = V3, amount model 3000 iters',
         'v3b': 'V3b = V3a, classifier depth 8 lr 0.03'}
LOG_LABELS = {'v1': 'V1: baseline + expected_spend features',
              'v2': 'V2: two-stage (Logloss clf + MAE reg on spenders, THB)',
              'v3': 'V3: two-stage (Logloss clf + MAE reg on log(y/prev))',
              'v3a': 'V3a: V3 with amount regressor iterations 3000',
              'v3b': 'V3b: V3a + classifier depth 8, lr 0.03'}
COMPONENT_NAMES = {'clf': 'classifier depth 6, lr 0.05', 'clf_deep': 'classifier depth 8, lr 0.03',
                   'thb': 'amount THB, 1000 iters', 'logratio': 'amount log-ratio, 1000 iters',
                   'logratio_3k': 'amount log-ratio, 3000 iters'}


def cb_params(iterations, fold, **overrides):
    """Baseline CatBoost settings (pis/tabular.py), seed shifted per fold like the baseline."""
    params = dict(iterations=iterations, depth=6, learning_rate=0.05, random_seed=SEED + fold,
                  thread_count=-1, verbose=False, allow_writing_files=False, early_stopping_rounds=80)
    return {**params, **overrides}


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



def current_commit():
    """Short HEAD hash, with '+dirty' if the working tree has uncommitted changes."""
    try:
        run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout.strip()
        return run('rev-parse', '--short', 'HEAD') + ('+dirty' if run('status', '--porcelain') else '')
    except (OSError, subprocess.CalledProcessError):
        return '?'


LOG_HEADER = """# Task 3 experiment log

| step | date | branch | change | fold scores | OOF | LB | note |
|---|---|---|---|---|---|---|---|
"""


def upsert_log(rows):
    """Add rows to ../LOG.md, one per variant, keyed by the `change` text.

    - same change already logged on THIS branch: replaced in place (re-runs don't duplicate);
    - same change logged on ANOTHER branch: skipped. That variant was introduced earlier and
      is only re-run here as a reference, its row stays where it was first logged;
    - otherwise appended. `step` is the order experiments were run in: rows from one run
      share a step, a re-run keeps this branch's step, a new branch gets the next number.
    """
    lines = (LOG_FILE.read_text(encoding='utf-8') if LOG_FILE.exists() else LOG_HEADER).rstrip('\n').split('\n')
    cells = {i: [c.strip() for c in line.strip().strip('|').split('|')]
             for i, line in enumerate(lines) if line.startswith('| ') and not line.startswith('| step')}
    steps = [int(c[0]) for c in cells.values() if c[0].isdigit()]
    same_branch = [int(c[0]) for c in cells.values() if c[0].isdigit() and c[2] == rows[0][1]]
    step = str(same_branch[0] if same_branch else max(steps, default=0) + 1)
    written = []
    for row in rows:
        hit = [(i, c[2]) for i, c in cells.items() if c[3] == row[2]]
        if hit and hit[0][1] != row[1]:
            continue
        new = '| ' + ' | '.join([step, *row]) + ' |'
        if hit:
            lines[hit[0][0]] = new
        else:
            lines.append(new)
        written.append(row[2])
    LOG_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return written


# --- Champion ---
# The champion is the best model so far, stored as its OOF predictions so any later
# experiment can be scored against it fold by fold without re-training it. Promotion
# rule: a variant replaces the champion only if it beats it on ALL 3 folds (an OOF gain
# driven by one fold is within fold-to-fold noise). Only honest rows are candidates:
# V1 and the t = 0.5 rows, never the OOF-tuned threshold rows (optimistic by design).
# Without a champion yet, the baseline plays that role.

def load_champion(train, base):
    """(meta, OOF prediction) of the current champion; the baseline if none exists yet."""
    if not (CHAMPION / 'task3_oof.csv').exists():
        return {'name': 'Baseline (no champion yet)'}, base
    meta = json.loads((CHAMPION / 'champion.json').read_text(encoding='utf-8'))
    oof = pd.read_csv(CHAMPION / 'task3_oof.csv')
    assert (oof['player_id'].to_numpy() == train['player_id'].to_numpy()).all(), 'champion OOF row order differs'
    assert np.allclose(oof['target'].to_numpy(), train[TARGET].to_numpy()), 'champion OOF targets differ'
    return meta, oof['prediction'].to_numpy(dtype=float)


def promote(variant, label, oof_frame, fold_scores, oof_score, previous):
    """Write experiments/task3/champion/ for the new champion (commit it with LOG.md)."""
    CHAMPION.mkdir(exist_ok=True)
    # Rounded to satang (2 decimals): the file is tracked in git, and full float precision
    # nearly doubles its size while changing fold scores by far less than 1e-4. The target
    # is already given to 2 decimals, so rounding it is lossless.
    oof_frame.round({'target': 2, 'prediction': 2}).to_csv(CHAMPION / 'task3_oof.csv', index=False)
    meta = {'name': label, 'variant': variant, 'script': 'experiments/task3/amount_model/amount_model.py',
            'branch': current_branch(), 'commit': current_commit(),
            'commit_note': 'HEAD when the run started; the promotion itself lands in the next commit',
            'fold_scores': [round(s, 6) for s in fold_scores], 'oof': round(oof_score, 6),
            'date': datetime.date.today().isoformat(), 'previous_champion': previous,
            'rule': 'promoted only if it beats the current champion on all 3 folds'}
    (CHAMPION / 'champion.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--variants', default=','.join(VARIANTS), help=f'comma-separated subset of {VARIANTS}')
    parser.add_argument('--quick', action='store_true', help='250 iterations for every model (smoke test)')
    parser.add_argument('--promote', action='store_true',
                        help='make the best variant that beats the champion on all folds the new champion')
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    if not variants or set(variants) - set(VARIANTS):
        parser.error(f'--variants must be a subset of {VARIANTS}')
    if args.promote and args.quick:
        parser.error('--promote needs a full run, not --quick')
    iters = lambda full: 250 if args.quick else full  # noqa: E731
    two_stage = [v for v in variants if v in TWO_STAGE]
    classifiers = sorted({TWO_STAGE[v][0] for v in two_stage})
    amounts = sorted({TWO_STAGE[v][1] for v in two_stage})

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
    champ_meta, champ = load_champion(train, base)
    champ_folds = [_score(y[va], champ[va]) for _, va in folds]

    # --- Containers: OOF predictions and fold-averaged test predictions per component ---
    keys = (['v1'] if 'v1' in variants else []) + classifiers + amounts
    n, nt = len(train), len(test)
    oof = {k: np.zeros(n) for k in keys}
    tst = {k: np.zeros(nt) for k in keys}
    best_iter = {k: [] for k in keys}
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

        if 'v1' in variants:
            # V1: baseline single-stage MAE model on all rows, plus the new features.
            model = CatBoostRegressor(loss_function='MAE', **cb_params(iters(1000), fold))
            model.fit(x_fit, y_fit, cat_features=categorical, eval_set=(x_val, y_val), use_best_model=True)
            oof['v1'][va] = np.maximum(0.0, model.predict(x_val))
            tst['v1'] += np.maximum(0.0, model.predict(x_tst)) / FOLDS
            best_iter['v1'].append(model.get_best_iteration() + 1)

        for c in classifiers:
            # Stage 1: will they spend? Logloss on all rows. No class weighting: the
            # threshold is fixed at 0.5 by the median rule, so we want calibrated p.
            clf = CatBoostClassifier(loss_function='Logloss', **cb_params(iters(1000), fold, **CLASSIFIERS[c]))
            clf.fit(x_fit, spend[tr].astype(int), cat_features=categorical,
                    eval_set=(x_val, spend[va].astype(int)), use_best_model=True)
            oof[c][va] = clf.predict_proba(x_val)[:, 1]
            tst[c] += clf.predict_proba(x_tst)[:, 1] / FOLDS
            best_iter[c].append(clf.get_best_iteration() + 1)

        # Stage 2 regressors learn only from spenders (the classifier handles zeros), and
        # early-stop on the validation fold's spenders. They predict for every row; the
        # combine step decides which rows get the amount.
        s_fit, s_val = spend[tr], spend[va]
        for a in amounts:
            kind, cap = AMOUNTS[a]
            reg = CatBoostRegressor(loss_function='MAE', **cb_params(iters(cap), fold))
            if kind == 'thb':
                # V2: amount in THB, MAE -> conditional median of y given y > 0.
                reg.fit(x_fit[s_fit], y_fit[s_fit], cat_features=categorical,
                        eval_set=(x_val[s_val], y_val[s_val]), use_best_model=True)
                oof[a][va] = np.maximum(0.0, reg.predict(x_val))
                tst[a] += np.maximum(0.0, reg.predict(x_tst)) / FOLDS
            else:
                # V3 family: amount as log(y / prev) with MAE; amount = prev × exp(pred).
                flag = lambda xf, raw: xf.assign(prev_missing=raw[PREV].isna().astype(int).to_numpy())  # noqa: E731
                f3, v3, t3 = flag(x_fit, fit_raw), flag(x_val, val_raw), flag(x_tst, test)
                reg.fit(f3[s_fit], log_ratio(y_fit[s_fit], fit_raw[s_fit]), cat_features=categorical,
                        eval_set=(v3[s_val], log_ratio(y_val[s_val], val_raw[s_val])), use_best_model=True)
                oof[a][va] = prev_or_one(val_raw) * np.exp(reg.predict(v3))
                tst[a] += prev_or_one(test) * np.exp(reg.predict(t3)) / FOLDS
            best_iter[a].append(reg.get_best_iteration() + 1)
        print(f'fold {fold}/{FOLDS} done in {time.time() - t0:.0f}s', flush=True)

    # --- Final predictions per variant (OOF and test) ---
    # Two-stage test predictions average p and amount over the 3 fold models first, then
    # apply the threshold: averaging already-thresholded predictions would give values
    # like 2/3 × amount, which is no row's median. The main files use t = 0.5 (the median
    # rule, honest score); the OOF-tuned threshold is saved next to them as *_tuned.csv.
    final = {}  # label -> dict(variant, oof, test, suffix, honest)
    if 'v1' in variants:
        final[NAMES['v1']] = dict(variant='v1', oof=oof['v1'], test=tst['v1'], suffix='', honest=True)
    thresholds = {}
    for v in two_stage:
        c, a = TWO_STAGE[v]
        thresholds[v] = tune_threshold(y, oof[c], oof[a])
        for t, suffix, tag in [(0.5, '', 't=0.5'), (thresholds[v], '_tuned',
                                                     f't={thresholds[v]:.3f} (OOF-tuned, optimistic)')]:
            final[f'{NAMES[v]}, {tag}'] = dict(variant=v, oof=combine(oof[c], oof[a], t),
                                               test=combine(tst[c], tst[a], t), suffix=suffix, honest=not suffix)

    # --- Summary table: every variant vs baseline AND vs champion, fold by fold ---
    def row(label, pred):
        fs = [_score(y[va], pred[va]) for _, va in folds]
        s = _score(y, pred)
        return [label, *fs, s, s - _score(y, base), all(a > b for a, b in zip(fs, base_folds)),
                s - _score(y, champ), all(a > b for a, b in zip(fs, champ_folds))]

    rows = [row('Baseline (baseline/task3_oof.csv)', base)]
    if 'variant' in champ_meta:
        rows.append(row(f'Champion: {champ_meta["name"]}', champ))
    rows += [row(label, f['oof']) for label, f in final.items()]
    cols = ['variant', 'fold1', 'fold2', 'fold3', 'OOF', 'Δ vs baseline', 'wins all folds vs baseline?',
            'Δ vs champion', 'wins all folds vs champion?']
    summary = pd.DataFrame(rows, columns=cols).set_index('variant', drop=False)
    shown = summary.copy()
    for c in ['fold1', 'fold2', 'fold3', 'OOF']:
        shown[c] = shown[c].map(fmt4)
    for c in ['Δ vs baseline', 'Δ vs champion']:
        shown[c] = shown[c].map('{:+.4f}'.format)
    for c in ['wins all folds vs baseline?', 'wins all folds vs champion?']:
        shown[c] = shown[c].map({True: 'yes', False: 'no'})
    # A reference row is not compared with itself: baseline vs baseline, champion vs champion
    # (with no champion yet, the baseline is the champion).
    shown.iloc[0, [6] if 'variant' in champ_meta else [6, 8]] = '—'
    if 'variant' in champ_meta:
        shown.iloc[1, 8] = '—'

    # --- Champion candidate: best honest row that beats the champion on all folds ---
    honest = [label for label, f in final.items() if f['honest']]
    winners = summary.loc[honest][summary.loc[honest, 'wins all folds vs champion?']]
    candidate = winners['OOF'].idxmax() if len(winners) else None

    # --- Error by bucket: baseline, champion, best variant, then all honest variants side by side ---
    best_label = summary.loc[honest, 'OOF'].idxmax()
    buckets = {'Baseline': bucket_table(y, base)}
    if 'variant' in champ_meta:
        buckets[f'Champion: {champ_meta["name"]}'] = bucket_table(y, champ)
    buckets[best_label] = bucket_table(y, final[best_label]['oof'])
    side = pd.DataFrame({'bucket': BUCKETS, 'count': bucket_table(y, base)['count'].map('{:,}'.format).to_numpy()})
    for label, pred in [('baseline', base)] + [(final[h]['variant'].upper(), final[h]['oof']) for h in honest]:
        side[label] = (bucket_table(y, pred)['error_sum'] / 1e6).map('{:.1f}'.format).to_numpy()

    # --- Stage diagnostics ---
    # AUC: how well stage 1 ranks spenders. Spender-only score: the amount model on rows
    # with y > 0, i.e. stage 2 judged as if stage 1 were perfect. The baseline's own score
    # on the same rows is the reference.
    used_by = lambda comp: '/'.join(v.upper() for v in two_stage if comp in TWO_STAGE[v])  # noqa: E731
    diag = [[f'{COMPONENT_NAMES[c]} ({used_by(c)})', 'OOF AUC (y > 0)', fmt4(roc_auc_score(spend, oof[c]))]
            for c in classifiers]
    if two_stage:
        diag.append(['baseline on spenders', 'OOF score, y > 0 rows', fmt4(_score(y[spend], base[spend]))])
    diag += [[f'{COMPONENT_NAMES[a]} ({used_by(a)})', 'OOF score, y > 0 rows',
              fmt4(_score(y[spend], oof[a][spend]))] for a in amounts]
    diag = pd.DataFrame(diag, columns=['model', 'measure', 'value'])
    caps = {'v1': iters(1000), **{c: iters(1000) for c in CLASSIFIERS}, **{a: iters(AMOUNTS[a][1]) for a in AMOUNTS}}
    iters_table = pd.DataFrame([[COMPONENT_NAMES.get(k, 'V1 regressor'), caps[k], ' / '.join(map(str, v))]
                                for k, v in best_iter.items()],
                               columns=['model', 'limit', 'best iteration per fold'])

    print(f'\nTASK 3 AMOUNT MODEL{" (QUICK: 250 iterations)" if args.quick else ""}')
    print(f'Champion: {champ_meta["name"]} (OOF {_score(y, champ):.4f})\n')
    print(shown.to_string(index=False))
    for label, table in buckets.items():
        print(f'\nError by bucket (OOF), {label}\n{fmt_bucket(table).to_string()}')
    print(f'\nError by bucket, all variants (million THB)\n{side.to_string(index=False)}')
    if len(diag):
        print(f'\nStage diagnostics\n{diag.to_string(index=False)}')
    print(f'\nBest iterations\n{iters_table.to_string(index=False)}')

    # --- OOF and test prediction files (outputs/ is gitignored) ---
    oof_frames = {}
    for label, f in final.items():
        folder = OUT / f'task3_{f["variant"]}'
        folder.mkdir(parents=True, exist_ok=True)
        oof_frames[label] = pd.DataFrame({'player_id': train['player_id'], 'target': y, 'prediction': f['oof']})
        oof_frames[label].to_csv(folder / f'task3_oof{f["suffix"]}.csv', index=False)
        frame = pd.DataFrame({'player_id': test['player_id'], 'prediction': f['test']})
        assert len(frame) == N_TEST and frame['prediction'].notna().all() and (frame['prediction'] >= 0).all()
        frame.to_csv(folder / f'task3_predictions{f["suffix"]}.csv', index=False)
    print(f'\nWrote OOF + test predictions to {OUT.relative_to(ROOT)}/task3_<variant>/')

    # --- Promotion ---
    promoted = None
    if candidate is None:
        print(f'Champion unchanged: no variant beats {champ_meta["name"]} on all folds.')
    elif args.promote:
        c = summary.loc[candidate]
        promote(final[candidate]['variant'], candidate, oof_frames[candidate],
                [c['fold1'], c['fold2'], c['fold3']], c['OOF'], champ_meta['name'])
        promoted = candidate
        print(f'PROMOTED to champion: {candidate} (OOF {c["OOF"]:.4f}); commit champion/ with LOG.md.')
    else:
        print(f'Candidate: {candidate} beats the champion on all folds; re-run with --promote to promote it.')

    # --- amount_model_results.md ---
    md = ['# Task 3 amount model',
          '',
          '> **CAUTION:** Early stopping uses the validation fold (as in the baseline), so every OOF score here '
          'is slightly optimistic; differences between rows are still comparable. Rows marked '
          '"OOF-tuned, optimistic" pick the threshold on the same OOF they are scored on.',
          '',
          f'Generated by `python experiments/task3/amount_model/amount_model.py --variants {",".join(variants)}'
          f'{" --quick" if args.quick else ""}{" --promote" if args.promote else ""}`. Folds: `KFold({FOLDS}, '
          f'shuffle=True, random_state={SEED})`, same as the baseline. Metric: `_score` from `pis/tabular.py`. '
          f'`expected_spend` = `{PREV}` × median `y / {PREV}` among spenders per segment×VIP '
          '(fallback segment → global, groups < 30 rows skipped), fitted on the outer-training part only; '
          f'training rows get inner `KFold({INNER_FOLDS})` out-of-fold values. Threshold grid for two-stage '
          f'variants: {GRID[0]:.3f}–{GRID[-1]:.3f} step 0.025.',
          '',
          f'Champion at the start of this run: **{champ_meta["name"]}** (`experiments/task3/champion/`). '
          'A variant is promoted only if it beats the champion on all 3 folds (t = 0.5 rows only). '
          + (f'**Promoted in this run: {promoted}.**' if promoted else 'Champion unchanged by this run.'),
          '',
          '## Summary',
          '',
          to_markdown(shown),
          '',
          '## Error by bucket (OOF)',
          '',
          '### All variants (million THB)',
          '',
          to_markdown(side),
          '']
    for label, table in buckets.items():
        md += [f'### {label}', '', to_markdown(fmt_bucket(table).reset_index()), '']
    if len(diag):
        md += ['## Stage diagnostics', '', to_markdown(diag), '']
    md += ['## Best iterations', '', to_markdown(iters_table), '']
    # Tables are regenerated; the Interpretation section is written by hand after
    # reading the numbers, so keep it across re-runs.
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')
    print(f'Wrote {RESULTS.relative_to(ROOT)}')

    # --- ../LOG.md: one row per variant introduced on this branch ---
    today, branch = datetime.date.today().isoformat(), current_branch()
    log_rows = []
    for v in variants:
        label = NAMES[v] if v == 'v1' else f'{NAMES[v]}, t=0.5'
        r = summary.loc[label]
        notes = ['quick smoke test (250 iters)'] if args.quick else []
        if v in thresholds:
            tuned = summary.loc[[k for k, f in final.items() if f['variant'] == v and not f['honest']][0]]
            notes.append(f't=0.5; OOF-tuned t={thresholds[v]:.3f}: {tuned["OOF"]:.4f} (optimistic)')
        notes.append(f'wins all folds vs baseline: {"yes" if r["wins all folds vs baseline?"] else "no"}')
        notes.append(f'vs champion ({champ_meta["name"]}): {r["Δ vs champion"]:+.4f}, wins all folds: '
                     f'{"yes" if r["wins all folds vs champion?"] else "no"}')
        if label == promoted:
            notes.append('**PROMOTED to champion**')
        log_rows.append([today, branch, LOG_LABELS[v], ' / '.join(fmt4(r[f'fold{i}']) for i in (1, 2, 3)),
                         fmt4(r['OOF']), '—', '; '.join(notes)])
    written = upsert_log(log_rows)
    print(f'Updated {LOG_FILE.relative_to(ROOT)}: {len(written)} row(s) '
          f'({len(log_rows) - len(written)} already logged on an earlier branch, kept as is)')


if __name__ == '__main__':
    main()
