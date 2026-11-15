"""Task 3 oracle test: upper bounds for each stage of a two-stage model.

ANALYSIS ONLY. Scenarios B-E read the true target, so they are upper bounds and must
never be used for a submission.

Run from the repo root:  python experiments/task3/oracle.py
Prints the summary and error-by-bucket tables and rewrites the tables in
experiments/task3/oracle_results.md (the hand-written interpretation is kept).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# --- Make `pis` importable no matter where the script is launched from ---
# Reuse the project's data path and metric instead of reimplementing them, so the
# numbers here are directly comparable with the baseline.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _score  # noqa: E402

# --- Settings ---
# Folds and seed must match pis/tabular.py, otherwise the baseline OOF predictions
# would be scored against different validation splits. EXPECTED_* is the baseline
# result that scenario A has to reproduce before any other number can be trusted.
TARGET, PREV = 'spending_30d', 'prev_month_spending'
FOLDS, SEED, MIN_GROUP = 3, 342, 30  # same folds as the baseline (pis/tabular.py)
EXPECTED_OOF, EXPECTED_FOLDS = 0.7753, [0.7873, 0.7587, 0.7807]
BINS, BUCKETS = [-1, 0, 1000, 20000, np.inf], ['0', '>0-1k', '1k-20k', '>20k']  # as in eda.ipynb
RESULTS = Path(__file__).with_name('oracle_results.md')
INTERPRETATION = '## Interpretation'


# --- Simple "how much?" rule used by scenarios C and D ---
# Each fit_* function only ever receives the training part of a fold, and lookup()
# applies the result to that fold's validation part, so no validation target leaks
# into the multipliers. MIN_GROUP = 30 drops groups too small for a stable median
# (e.g. rare segment × VIP combinations); their rows fall back to the next level.

def fit_multipliers(train, keys):
    """Median of y / prev per group, from spenders with a known prev; drop groups < MIN_GROUP rows."""
    spent = train[(train[TARGET] > 0) & train[PREV].notna()]
    ratio = spent[TARGET] / spent[PREV]
    if not keys:
        return float(ratio.median())
    stats = ratio.groupby([spent[k] for k in keys]).agg(['median', 'size'])  # NaN keys are dropped
    return stats.loc[stats['size'] >= MIN_GROUP, 'median']


def fit_level(train, keys):
    """Median spend per group, used only for spenders whose prev_month_spending is missing."""
    spent = train[train[TARGET] > 0]
    if not keys:
        return float(spent[TARGET].median())
    stats = spent[TARGET].groupby([spent[k] for k in keys]).agg(['median', 'size'])
    return stats.loc[stats['size'] >= MIN_GROUP, 'median']


def lookup(valid, hierarchy):
    """Apply the first group level in `hierarchy` that has a fitted value; the last level is global."""
    out = pd.Series(np.nan, index=valid.index)
    # Levels go from most to least specific (segment×VIP -> segment -> global). Each
    # level only fills rows still NaN: a missing segment/VIP value or a group dropped
    # for having < MIN_GROUP rows. The global level always fills the rest.
    for keys, table in hierarchy:
        if not keys:
            out = out.fillna(table)
            break
        idx = pd.MultiIndex.from_frame(valid[keys]) if len(keys) > 1 else pd.Index(valid[keys[0]])
        out = out.fillna(pd.Series(table.reindex(idx).to_numpy(), index=valid.index))
    return out


def rule_prediction(train, valid, levels):
    """Oracle-spend rule: 0 where y == 0, else prev × in-fold group multiplier.

    Spenders with no prev_month_spending get the in-fold group median spend instead
    (same fallback order), because prev × multiplier is undefined for them.
    """
    mult = lookup(valid, [(k, fit_multipliers(train, k)) for k in levels])
    level = lookup(valid, [(k, fit_level(train, k)) for k in levels])
    # NaN prev gives NaN here; filling with the group's typical spend keeps those
    # ~3k spenders in the score with a model-free guess instead of dropping them.
    pred = (valid[PREV] * mult).fillna(level)
    return np.where(valid[TARGET].to_numpy() == 0, 0.0, pred.to_numpy())


# --- Error by spend bucket (same buckets as the EDA) ---
# Shows which kind of player the remaining error comes from after each scenario.

def bucket_table(y, pred):
    frame = pd.DataFrame({'bucket': pd.cut(y, BINS, labels=BUCKETS), 'err': np.abs(y - pred)})
    table = frame.groupby('bucket', observed=False)['err'].agg(count='size', error_sum='sum')
    table['error_share'] = table['error_sum'] / table['error_sum'].sum()
    return table


def fmt_bucket(table):
    out = table.copy()
    out['count'] = out['count'].map('{:,}'.format)
    out['error_sum'] = out['error_sum'].map('{:,.0f}'.format)
    out['error_share'] = out['error_share'].map('{:.1%}'.format)
    return out


def to_markdown(frame):
    """Markdown table without the optional `tabulate` dependency."""
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '---|' * len(frame.columns)]
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ

    # --- Load data and check it lines up with the baseline OOF ---
    # The OOF file has no fold column, so rows are matched to folds by position;
    # that only works if train.csv is in the same order as when the baseline ran.
    train =pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)  # as pis/tabular.py
    oof = pd.read_csv(ROOT / 'baseline' / 'task3_oof.csv')
    assert len(train) == len(oof) and (train['player_id'].to_numpy() == oof['player_id'].to_numpy()).all(), \
        'player_id order differs between train.csv and baseline/task3_oof.csv'
    assert np.allclose(train[TARGET].to_numpy(), oof['target'].to_numpy()), 'targets differ from OOF file'

    # --- Build the 5 scenario predictions (A-E) ---
    # Each scenario makes one stage perfect (using the true target) and keeps the
    # other as is, so the score gain shows how much that stage is worth.
    y = train[TARGET].to_numpy(dtype=float)
    base = oof['prediction'].to_numpy(dtype=float)
    zero = y == 0
    names = {'A': 'A. Baseline',
             'B': 'B. Oracle zeros + baseline amount',
             'C': 'C. Oracle zeros + prev × segment ratio',
             'D': 'D. Oracle zeros + prev × segment×VIP ratio',
             'E': 'E. Baseline zeros + oracle amount'}
    preds = {k: np.zeros(len(y)) for k in names}
    preds['A'] = base  # reference point: the current single-stage model
    preds['B'] = np.where(zero, 0.0, base)  # perfect "will they spend?", baseline "how much?"
    preds['E'] = np.where(zero, base, y)  # perfect "how much?", baseline zero problem left in
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(train))
    for tr, va in folds:
        fit, val = train.iloc[tr], train.iloc[va]
        # C, D: perfect "will they spend?" + a simple ratio rule for "how much?",
        # i.e. is a rule from the EDA enough for the amount stage?
        # Fallback order, `[]` = global. Next level if key is NaN or group < MIN_GROUP.
        preds['C'][va] = rule_prediction(fit, val, [['segment'], []])  # segment -> global
        # segment×VIP -> segment -> global
        preds['D'][va] = rule_prediction(fit, val, [['segment', 'vip_status'], ['segment'], []])
    preds = {k: np.maximum(0.0, p) for k, p in preds.items()}

    # --- Score per fold and overall; stop if A != baseline ---
    # If A does not match the published baseline, the folds or row order are off and
    # every other scenario would be scored on the wrong basis, so exit instead.
    rows = []
    for k, name in names.items():
        fold_scores = [_score(y[va], preds[k][va]) for _, va in folds]
        rows.append([name, *fold_scores, _score(y, preds[k])])
    summary = pd.DataFrame(rows, columns=['scenario', 'fold1', 'fold2', 'fold3', 'OOF'])
    summary['Δ vs baseline'] = summary['OOF'] - summary.loc[0, 'OOF']

    a = summary.iloc[0]
    if abs(a['OOF'] - EXPECTED_OOF) > 1e-4 or any(abs(a[f'fold{i + 1}'] - s) > 1e-4
                                                     for i, s in enumerate(EXPECTED_FOLDS)):
        sys.exit(f'Scenario A does not reproduce the baseline: {a.to_dict()}')

    # --- Print tables and write oracle_results.md ---
    shown = summary.copy()
    for c in ['fold1', 'fold2', 'fold3', 'OOF']:
        shown[c] = shown[c].map('{:.4f}'.format)
    shown['Δ vs baseline'] = shown['Δ vs baseline'].map('{:+.4f}'.format)
    buckets = {k: fmt_bucket(bucket_table(y, preds[k])) for k in ['A', 'B', 'C', 'E']}
    no_prev = int((~zero & train[PREV].isna()).sum())

    print('ORACLE TEST (upper bounds, NOT submittable)\n')
    print(shown.to_string(index=False))
    for k, table in buckets.items():
        print(f'\nError by bucket (OOF), {names[k]}\n{table.to_string()}')
    print(f'\nC/D: {no_prev:,} spenders have no {PREV}; they get the in-fold group median spend.')

    md = ['# Task 3 oracle test',
          '',
          '> **CAUTION:** Oracle = upper bound, not a submittable result. Scenarios B–E use the '
          'true target (`spending_30d`) to decide who spends and/or how much.',
          '',
          f'Generated by `python experiments/task3/oracle.py`. Folds: `KFold({FOLDS}, shuffle=True, '
          f'random_state={SEED})`, same as the baseline. Metric: `_score` from `pis/tabular.py`. '
          'Multipliers in C/D are fitted on each fold\'s training part only (median of '
          f'`y / {PREV}` among spenders; groups with < {MIN_GROUP} rows fall back to segment, then global). '
          f'{no_prev:,} spenders have no `{PREV}`; C/D give them the in-fold group median spend.',
          '',
          '## Summary',
          '',
          to_markdown(shown),
          '',
          '## Error by bucket (OOF)',
          '']
    for k, table in buckets.items():
        md += [f'### {names[k]}', '', to_markdown(table.reset_index()), '']
    # The tables are regenerated, but the Interpretation section is written by hand
    # after reading the numbers; keep it so a re-run does not wipe it out.
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')
    print(f'\nWrote {RESULTS.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
