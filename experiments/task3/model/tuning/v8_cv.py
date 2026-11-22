"""Task 3 V8 CV rebuilt from v8.json, without Optuna (protocol honest-es).

refit.py --variant v8 needs V8's CV test predictions and the honest-es classifier p, both
in outputs/ (gitignored). tuning.py made them during the Optuna study, but re-running it
would redo the whole study. This script rebuilds them from the tracked settings alone:

  1. data, outer folds and the baseline check (honest_es.Data);
  2. per-fold V4 inputs: expected_spend, features, fit/es split (honest_es.fold_inputs);
  3. the honest-es classifier p (honest_es.classifier_cache): loaded from
     outputs/experiments/task3_v4_honest/classifier_p.npz, or built and saved there with
     the code honest_es.py runs (no LOG, no results section, no champion change);
  4. V8's amount model with the settings of v8.json (amount_params, iterations limit,
     random_seed 342 + fold), early-stopped on es, through honest_es.run_pipeline, the
     function tuning.py scored every trial with.
Check: the fold scores must match champion.json (V8) within 1e-4. Only then does it write
outputs/experiments/task3_v8/task3_predictions.csv (test, averaged over the 3 fold models)
and task3_oof.csv, where refit.py --variant v8 reads them. Nothing tracked is written.

Run from the repo root:
    python experiments/task3/model/tuning/v8_cv.py [--threads N]
--threads sets CatBoost's thread_count (default -1 = all cores). CatBoost on CPU builds
slightly different trees with a different thread count. V8 was recorded with 12 threads:
--threads 12 reproduces it exactly (classifier best iterations 98 / 182 / 240, OOF equal to
champion/task3_oof.csv), while 16 threads gave classifier best iterations 95 / 106 / 273 and
fold scores within 4e-5 (amount best iterations unchanged). The classifier p is cached, so
after changing --threads delete outputs/experiments/task3_v4_honest/classifier_p.npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from honest_es import (CHAMPION, OUT, OUT_DIR, PROTOCOL, ROOT, SEED, Data, classifier_cache,  # noqa: E402
                       read_champion, run_pipeline, save_predictions)
from amount_model import set_threads  # noqa: E402  (on sys.path via honest_es)
from pis.tabular import _score  # noqa: E402

V8_JSON = Path(__file__).with_name('v8.json')
V8_DIR = OUT / 'task3_v8'
TOL = 1e-4


class Steps:
    """Prints the elapsed time of each step and keeps them for the summary."""

    def __init__(self):
        self.t0 = self.last = time.time()
        self.rows = []

    def done(self, name):
        now = time.time()
        self.rows.append((name, now - self.last))
        print(f'[{now - self.last:6.0f}s] {name}', flush=True)
        self.last = now

    def summary(self):
        lines = [f'  {name:<54} {sec:7.0f}s' for name, sec in self.rows]
        return '\n'.join(['Elapsed per step', *lines, f'  {"total":<54} {time.time() - self.t0:7.0f}s'])


def expected_scores(v8):
    """Fold scores to reproduce: champion.json if V8 is the champion, else v8.json (same numbers)."""
    champ = read_champion()
    if champ.get('variant') == 'v8':
        if champ.get('protocol') != PROTOCOL:
            sys.exit(f'champion.json V8 is on protocol {champ.get("protocol")}, expected {PROTOCOL}')
        return champ['fold_scores'], 'champion.json'
    return v8['fold_scores'], 'v8.json'


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--threads', type=int, default=-1, help='CatBoost thread_count (default -1 = all cores)')
    args = parser.parse_args()
    set_threads(args.threads)
    if not V8_JSON.exists():
        sys.exit(f'{V8_JSON.relative_to(ROOT)} not found')
    v8 = json.loads(V8_JSON.read_text(encoding='utf-8'))
    if v8.get('protocol') != PROTOCOL:
        sys.exit(f'v8.json is on protocol {v8.get("protocol")}, expected {PROTOCOL}')
    expected, source = expected_scores(v8)
    print(f'V8 = trial {v8["trial"]} of study {v8["study"]}; amount settings {v8["amount_params"]}, '
          f'limit {v8["amount_iterations_limit"]}, seed {SEED} + fold; CatBoost threads: '
          f'{"all cores" if args.threads == -1 else args.threads}', flush=True)
    steps = Steps()

    d = Data()
    steps.done('data + folds + baseline check')
    d.inputs()
    steps.done('fold inputs (expected_spend, features, fit/es split)')

    clf_path = OUT_DIR / 'classifier_p.npz'
    cached = clf_path.exists()
    if not cached:
        print(f'{clf_path.relative_to(ROOT)} not found: building the honest-es classifier p '
              '(honest_es.classifier_cache)', flush=True)
    clf = classifier_cache(d)
    steps.done(f'classifier p ({"loaded" if cached else "built and saved"})')
    clf_ok = clf['best'] == v8['classifier_best_iterations']
    print(f'classifier best iterations {clf["best"]} vs v8.json {v8["classifier_best_iterations"]}: '
          f'{"match" if clf_ok else "DIFFER"}', flush=True)
    if not clf_ok:
        print(f'  -> the classifier p was {"loaded from a cache built" if cached else "built"} with another thread count '
              f'than V8\'s run (12 threads); for an exact reproduction delete {clf_path.relative_to(ROOT)} and re-run '
              'with --threads 12', flush=True)

    def on_fold(fold, score):
        steps.done(f'amount model fold {fold}: {score:.6f} (expected {expected[fold - 1]:.6f})')

    res = run_pipeline(d, clf, v8['amount_params'], SEED, v8['amount_iterations_limit'], on_fold)
    amt_ok = res['best'] == v8['amount_best_iterations']
    print(f'amount best iterations {res["best"]} vs v8.json {v8["amount_best_iterations"]}: '
          f'{"match" if amt_ok else "DIFFER"}', flush=True)

    # --- Check against the recorded V8 fold scores ---
    deltas = [a - b for a, b in zip(res['folds'], expected)]
    folds_ok = all(abs(x) <= TOL for x in deltas)
    champ_oof = pd.read_csv(CHAMPION / 'task3_oof.csv')['prediction'].to_numpy(dtype=float)
    oof_diff = float(np.max(np.abs(np.round(res['oof'], 2) - champ_oof))) if read_champion()['variant'] == 'v8' else None
    table = pd.DataFrame({'fold': [1, 2, 3, 'OOF'],
                          'rebuilt': [f'{s:.6f}' for s in [*res['folds'], res['score']]],
                          f'expected ({source})': [f'{s:.6f}' for s in [*expected, v8['oof']]],
                          'Δ': [f'{x:+.2e}' for x in [*deltas, res['score'] - v8['oof']]],
                          f'|Δ| ≤ {TOL:g}': [('ok' if abs(x) <= TOL else 'FAIL') for x in deltas] + ['—']})
    print(f'\n{table.to_string(index=False)}')
    if oof_diff is not None:
        print(f'OOF predictions vs champion/task3_oof.csv (rounded to 0.01): max |Δ| {oof_diff:.4g} THB')
    if not folds_ok:
        print(f'\nFAIL: fold scores differ from {source} by more than {TOL:g}; nothing written to '
              f'{V8_DIR.relative_to(ROOT)}.\nBest iterations: classifier {"match" if clf_ok else "differ"}, amount '
              f'{"match" if amt_ok else "differ"}. Different best iterations with the same data and seeds usually mean '
              'CatBoost built slightly different trees: a thread count other than V8\'s 12 (--threads 12, after '
              f'deleting {clf_path.relative_to(ROOT)}) or another CatBoost version.')
        print(steps.summary())
        sys.exit(1)

    oof = save_predictions(d, V8_DIR, res)
    assert len(oof) == len(d.y) and abs(_score(d.y, oof['prediction'].to_numpy()) - res['score']) < 1e-12
    steps.done('write predictions')
    print(f'\nOK: V8 reproduced within {TOL:g} on every fold. Wrote {(V8_DIR / "task3_predictions.csv").relative_to(ROOT)} '
          f'and task3_oof.csv; next: python experiments/task3/model/refit/refit.py --variant v8')
    print(steps.summary())


if __name__ == '__main__':
    main()
