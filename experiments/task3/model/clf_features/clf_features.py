"""Task 3 classifier feature screen: do new features lift stage 1 ("will they spend?")?

V4's classifier is stuck at OOF AUC 0.7841 and a bigger model did not help (V3b: 0.7842),
so this tests new INPUTS. Only the classifier is trained (no amount model, no combine):
the V4 classifier recipe (Logloss, cb_params, early stopping on the validation fold, same
folds and seeds) with V4's features + one candidate group at a time, then all groups.

Rule (decided before running): a group is kept only if it raises AUC on EVERY fold vs the
V4 classifier; if the best combination of kept groups adds < +0.002 OOF AUC, stop and
write up the negative result (no V5 run).

Run from the repo root:
    python experiments/task3/model/clf_features/clf_features.py [--threads 12]
Writes experiments/task3/model/clf_features/clf_features_results.md (the hand-written
Interpretation section is kept) and one summary row in experiments/task3/LOG.md.

Known issue: REF_AUC (0.7841) was recorded with 12 CatBoost threads. CatBoost on CPU builds
slightly different trees with another thread count (all 16 cores give 0.7845), so without
--threads 12 the reference check prints a warning and the screen goes on. The screen is a
negative result (no V5) and its conclusion does not depend on this.
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'amount_model'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amount_model import (CLASSIFIERS, TWO_STAGE, add_features, cb_params, current_branch,  # noqa: E402
                          fold_expected_spend, set_threads, upsert_log)
from features import GROUPS, build_groups, with_groups  # noqa: E402
from oracle import FOLDS, SEED, TARGET, to_markdown  # noqa: E402
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _prepare  # noqa: E402

CLF = TWO_STAGE['v4'][0]  # V4's classifier component ('clf'): depth 6, lr 0.05
REF_AUC = 0.7841  # V4 classifier OOF AUC from amount_model_results.md, recorded with 12 CatBoost threads
REF_THREADS = 12
MIN_GAIN = 0.002
RESULTS = Path(__file__).with_name('clf_features_results.md')
INTERPRETATION = '## Interpretation'
LOG_CHANGE = 'Classifier feature screen: V4 clf + timing / activity / spend_habit / zero_rate groups'


def fit_clf(x_fit, y_fit, x_val, y_val, categorical, fold):
    """V4's stage-1 classifier, exactly as in amount_model.py."""
    clf = CatBoostClassifier(loss_function='Logloss', **cb_params(1000, fold, **CLASSIFIERS[CLF]))
    clf.fit(x_fit, y_fit, cat_features=categorical, eval_set=(x_val, y_val), use_best_model=True)
    return clf


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--threads', type=int, default=-1,
                        help=f'CatBoost thread_count (default -1 = all cores); {REF_THREADS} reproduces REF_AUC exactly')
    args = parser.parse_args()
    set_threads(args.threads)
    sys.stdout.reconfigure(encoding='utf-8')
    t_start = time.time()
    train = pd.read_csv(DEFAULT_DATA / '03_regression' / 'train.csv')
    test = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv')
    train = train.loc[train[TARGET].notna()].reset_index(drop=True)
    x, _, categorical = _prepare(train, test, TARGET)
    y = train[TARGET].to_numpy(dtype=float)
    spend = (y > 0).astype(int)
    folds = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(x))

    # --- Per-fold inputs, built once: V4 features + every group's features ---
    per_fold = []
    for tr, va in folds:
        fit_raw, val_raw = train.iloc[tr], train.iloc[va]
        es_fit, (es_val,) = fold_expected_spend(fit_raw, [val_raw])
        fit_parts, (val_parts,) = build_groups(GROUPS, fit_raw, y[tr], [val_raw])
        per_fold.append(dict(x_fit=add_features(x.iloc[tr], fit_raw, es_fit),
                             x_val=add_features(x.iloc[va], val_raw, es_val),
                             fit=dict(zip(GROUPS, fit_parts)), val=dict(zip(GROUPS, val_parts))))
    print(f'features built in {time.time() - t_start:.0f}s', flush=True)

    results, importances = {}, {}

    def run(label, groups):
        t0 = time.time()
        oof, best = np.zeros(len(y)), []
        imp = None
        for fold, ((tr, va), d) in enumerate(zip(folds, per_fold), 1):
            xf = with_groups(d['x_fit'], [d['fit'][g] for g in groups])
            xv = with_groups(d['x_val'], [d['val'][g] for g in groups])
            clf = fit_clf(xf, spend[tr], xv, spend[va], categorical, fold)
            oof[va] = clf.predict_proba(xv)[:, 1]
            best.append(clf.get_best_iteration() + 1)
            fi = pd.Series(clf.get_feature_importance(), index=xf.columns)
            imp = fi / FOLDS if imp is None else imp + fi / FOLDS
        results[label] = dict(groups=groups, oof=oof, best=best, seconds=time.time() - t0,
                              fold_auc=[roc_auc_score(spend[va], oof[va]) for _, va in folds],
                              auc=roc_auc_score(spend, oof), logloss=log_loss(spend, oof))
        importances[label] = imp
        r = results[label]
        print(f'{label}: AUC {r["auc"]:.4f} ({" / ".join(f"{a:.4f}" for a in r["fold_auc"])}), '
              f'{r["seconds"]:.0f}s', flush=True)

    run('V4 classifier (reference)', [])
    ref = results['V4 classifier (reference)']
    if abs(ref['auc'] - REF_AUC) > 5e-5:
        # Known issue (see the docstring and LOG.md step 11): a warning, not a stop.
        print(f'WARNING: reference AUC {ref["auc"]:.5f} does not reproduce V4\'s {REF_AUC}. REF_AUC was recorded '
              f'with {REF_THREADS} CatBoost threads and this run uses --threads {args.threads}; CatBoost builds '
              f'slightly different trees with another thread count. Run with --threads {REF_THREADS} to reproduce it '
              'exactly. Continuing: the deltas below are measured against this run\'s own reference.',
              file=sys.stderr, flush=True)
    for g in GROUPS:
        run(f'+ {g}', [g])
    run('+ all 4 groups', list(GROUPS))

    # --- Keep rule: a group stays only if it beats the reference on every fold ---
    wins = lambda r: all(a > b for a, b in zip(r['fold_auc'], ref['fold_auc']))  # noqa: E731
    kept = [g for g in GROUPS if wins(results[f'+ {g}'])]
    if len(kept) >= 2 and kept != GROUPS:
        run(f'+ kept ({" + ".join(kept)})', kept)
    # Best combination = best OOF AUC among configs made only of kept groups (and winning every fold).
    kept_labels = [lbl for lbl, r in results.items() if r['groups'] and set(r['groups']) <= set(kept) and wins(r)]
    best_label = max(kept_labels, key=lambda lbl: results[lbl]['auc']) if kept_labels else None
    gain = results[best_label]['auc'] - ref['auc'] if best_label else None
    passed = gain is not None and gain >= MIN_GAIN
    top_any = max((lbl for lbl, r in results.items() if r['groups']), key=lambda lbl: results[lbl]['auc'])
    any_note = f' (best of all configs, kept or not: {top_any} {results[top_any]["auc"] - ref["auc"]:+.4f})'
    if best_label is None:
        decision = f'STOP: no group raises AUC on every fold, so no V5 run{any_note}.'
    elif not passed:
        decision = (f'STOP: kept groups: {", ".join(kept)}; best kept combination ({best_label}) adds '
                    f'{gain:+.4f} OOF AUC < +{MIN_GAIN}, so no V5 run{any_note}.')
    else:
        decision = f'PASS: {best_label} adds {gain:+.4f} OOF AUC (≥ +{MIN_GAIN}); kept groups: {", ".join(kept)}.'

    # --- Tables ---
    rows = []
    for label, r in results.items():
        is_ref = not r['groups']
        rows.append([label, *[f'{a:.4f}' for a in r['fold_auc']], f'{r["auc"]:.4f}',
                     '—' if is_ref else f'{r["auc"] - ref["auc"]:+.4f}',
                     '—' if is_ref else ' / '.join(f'{a - b:+.4f}' for a, b in zip(r['fold_auc'], ref['fold_auc'])),
                     '—' if is_ref else ('yes' if wins(r) else 'no'),
                     f'{r["logloss"]:.4f}', ' / '.join(map(str, r['best'])), f'{r["seconds"]:.0f}'])
    screen = pd.DataFrame(rows, columns=['classifier inputs', 'fold1 AUC', 'fold2 AUC', 'fold3 AUC', 'OOF AUC',
                                         'Δ OOF AUC', 'Δ per fold', 'wins every fold?', 'OOF logloss',
                                         'best iterations', 'seconds'])
    imp = importances['+ all 4 groups'].sort_values(ascending=False)
    ref_imp = importances['V4 classifier (reference)']
    new_cols = set().union(*[per_fold[0]['fit'][g].columns for g in GROUPS])
    group_of = {c: g for g in GROUPS for c in per_fold[0]['fit'][g].columns}
    top = pd.DataFrame({'rank': range(1, 16), 'feature': imp.index[:15],
                        'importance': [f'{v:.2f}' for v in imp.values[:15]],
                        'new?': [f'new ({group_of[c]})' if c in new_cols else 'V4' for c in imp.index[:15]],
                        # Same feature in the V4 reference model: shows whether a new feature brings new
                        # information or only takes over importance from the columns it is built from.
                        'importance in V4 reference': [f'{ref_imp[c]:.2f}' if c in ref_imp else '—'
                                                       for c in imp.index[:15]]})
    new_rank = pd.DataFrame({'feature': [c for c in imp.index if c in new_cols],
                             'group': [group_of[c] for c in imp.index if c in new_cols],
                             'rank (of {})'.format(len(imp)): [list(imp.index).index(c) + 1 for c in imp.index if c in new_cols],
                             'importance': [f'{imp[c]:.2f}' for c in imp.index if c in new_cols]})
    total = time.time() - t_start

    print(f'\nCLASSIFIER FEATURE SCREEN\n{screen.to_string(index=False)}')
    print(f'\nTop 15 importances, all-in model (PredictionValuesChange, mean of 3 folds)\n{top.to_string(index=False)}')
    print(f'\nNew features in the all-in model\n{new_rank.to_string(index=False)}')
    print(f'\n{decision}\nTotal runtime {total:.0f}s')

    # --- clf_features_results.md (Interpretation kept) ---
    md = ['# Task 3 classifier feature screen', '',
          'Generated by `python experiments/task3/model/clf_features/clf_features.py`. Stage-1 classifier only '
          '(V4 recipe: `CatBoostClassifier` Logloss, depth 6, lr 0.05, early stopping 80 on the validation fold, '
          f'seed {SEED} + fold), `KFold({FOLDS}, shuffle=True, random_state={SEED})` like every Task 3 experiment. '
          'Inputs = V4 features (incl. the `expected_spend` family) + the listed group(s). Feature definitions and '
          'the reason for each are in `features.py`. `group_zero_rate` uses the target: inner `KFold(5)` OOF '
          'for the training rows, full outer-train for validation rows, smoothed toward the global rate (m = 50).',
          '', '> **NOTE:** Rule fixed before the run: a group is kept only if it raises AUC on every fold; if the '
          f'best combination of kept groups adds < +{MIN_GAIN} OOF AUC, stop (no V5).', '',
          f'**Decision: {decision}**', '',
          '## Screen', '', to_markdown(screen), '',
          '## Top 15 feature importances (all-in model)', '',
          'CatBoost `PredictionValuesChange`, mean of the 3 fold models.', '', to_markdown(top), '',
          '### Where the new features rank (all-in model)', '', to_markdown(new_rank), '',
          '## Runtime', '',
          f'Total {total:.0f}s (features built once per fold; per-config seconds, 3 fold fits each, are in the '
          'screen table).', '']
    old = RESULTS.read_text(encoding='utf-8') if RESULTS.exists() else ''
    keep = old[old.index(INTERPRETATION):] if INTERPRETATION in old else f'{INTERPRETATION}\n\n_TODO_\n'
    RESULTS.write_text('\n'.join(md) + '\n' + keep, encoding='utf-8')

    all_in = results['+ all 4 groups']
    note = ('classifier only, metric = AUC (not the Task 3 score); reference V4 clf AUC '
            f'{ref["auc"]:.4f} ({" / ".join(f"{a:.4f}" for a in ref["fold_auc"])}); '
            + '; '.join(f'{lbl.lstrip("+ ")} {r["auc"] - ref["auc"]:+.4f} (wins every fold: {"yes" if wins(r) else "no"})'
                        for lbl, r in results.items() if r['groups'])
            + f'; {decision} `model/clf_features/`; runtime {total:.0f}s')
    upsert_log([[datetime.date.today().isoformat(), current_branch(), LOG_CHANGE,
                 'AUC ' + ' / '.join(f'{a:.4f}' for a in all_in['fold_auc']) + ' (all-in)',
                 f'AUC {all_in["auc"]:.4f} (all-in)', '—', note]])
    print(f'Wrote {RESULTS.relative_to(ROOT)}; updated experiments/task3/LOG.md')


if __name__ == '__main__':
    main()
