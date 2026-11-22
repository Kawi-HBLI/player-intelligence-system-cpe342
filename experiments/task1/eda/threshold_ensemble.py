r"""Task 1: is the OOF-tuned threshold right for the 10-model average we submit? (train rows only)

Task1/train.py tunes the threshold on the 10-fold OOF, where every row is scored by ONE fold
model, but the test column thresholds the AVERAGE of the 10 fold models. This script nests the
pipeline inside an outer StratifiedKFold(3, seed 342) on the labelled train rows: for each
outer fold H with the rest R it runs Task1/train.py's own entry point (run.main, 10 folds,
same seed, profile and CatBoost settings) with R as train.csv and H's features (no label) as
test.csv. The pipeline fits its imputers on the train part only (pis/task1_features.build), so
H never influences the features of R. Saved per outer fold: the inner OOF on R (single-model
scores) and the 10-model average on H.

test.csv and test predictions are never read.

    .\.venv\Scripts\python.exe experiments/task1/eda/threshold_ensemble.py --threads 12

Outputs (gitignored): outputs/experiments/task1_threshold_ensemble/fold{1,2,3}/scores.npz.
An outer fold that already has scores.npz is skipped (delete it to re-run).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').is_file())
sys.path.insert(0, str(ROOT))
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.tabular import _threshold  # noqa: E402
from run import main as run_pipeline  # noqa: E402

TASK_DIR = '01_binary_classification'
TARGET = 'is_cheater'
OUT = ROOT / 'outputs' / 'experiments' / 'task1_threshold_ensemble'
OUTER_SEED, INNER_FOLDS = 342, '10'   # Task1/train.py: TASK1_FOLDS = '10'


def labelled_train(data_root=DEFAULT_DATA):
    train = pd.read_csv(Path(data_root) / TASK_DIR / 'train.csv')
    return train.loc[train[TARGET].notna()].reset_index(drop=True)


def outer_folds(y):
    return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SEED).split(np.zeros(len(y)), y))


def run_outer_fold(k, rest, held, threads):
    """Run the Task 1 pipeline on `rest`, scoring `held` as its test set; save both score sets."""
    fold_dir = OUT / f'fold{k}'
    if (fold_dir / 'scores.npz').exists():
        return None
    started = time.time()
    run_dir = fold_dir / 'run'
    if not ((run_dir / 'task1_oof.csv').exists() and (run_dir / 'task1_probabilities.csv').exists()):
        data = fold_dir / 'data' / TASK_DIR
        data.mkdir(parents=True, exist_ok=True)
        rest.to_csv(data / 'train.csv', index=False)
        held.drop(columns=TARGET).to_csv(data / 'test.csv', index=False)   # features only, never H's labels
        run_pipeline(['train', '--tasks', '1', '--folds', INNER_FOLDS, '--threads', str(threads),
                      '--data-root', str(fold_dir / 'data'), '--output-dir', str(run_dir)])
        shutil.rmtree(fold_dir / 'data')                                  # copies of train rows only; not needed again
    inner = pd.read_csv(run_dir / 'task1_oof.csv')
    ensemble = pd.read_csv(run_dir / 'task1_probabilities.csv')
    assert (inner['player_id'].to_numpy() == rest['player_id'].to_numpy()).all()
    assert (ensemble['player_id'].to_numpy() == held['player_id'].to_numpy()).all()
    # The label is read as float (the full file has NaN labels), so the column is 'probability_1.0'.
    cheater_col = next(c for c in ensemble.columns if c.startswith('probability_') and float(c.split('_', 1)[1]) == 1)
    np.savez(fold_dir / 'scores.npz', r_ids=inner['player_id'].to_numpy(), r_y=inner['target'].to_numpy().astype(int),
             r_oof=inner['probability_1'].to_numpy(), h_ids=held['player_id'].to_numpy(),
             h_y=held[TARGET].to_numpy().astype(int), h_ens=ensemble[cheater_col].to_numpy())
    return time.time() - started


def load_scores():
    return [dict(np.load(OUT / f'fold{k}' / 'scores.npz', allow_pickle=True)) for k in (1, 2, 3)]


def f2(y, p, t):
    return fbeta_score(y, (p >= t).astype(int), beta=2)


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def per_fold_table(scores):
    rows = []
    for k, s in enumerate(scores, 1):
        t_oof, f2_r = _threshold(s['r_y'], s['r_oof'])
        t_best, f2_best = _threshold(s['h_y'], s['h_ens'])
        f2_at_oof = f2(s['h_y'], s['h_ens'], t_oof)
        rows.append({'outer fold': k, 't_oof': t_oof, 'OOF F2 on R': f2_r, 't_best_H': t_best, 'shift': t_best - t_oof,
                     'logit shift': logit(t_best) - logit(t_oof), 'F2 on H at t_oof': f2_at_oof,
                     'F2 on H at t_best_H': f2_best, 'gap (oracle)': f2_best - f2_at_oof})
    return pd.DataFrame(rows).set_index('outer fold')


def honest_correction(scores, table):
    """Shift estimated on 2 outer folds, applied to t_oof of the 3rd (additive and in logit)."""
    rows = []
    for k, s in enumerate(scores, 1):
        others = table.drop(index=k)
        t_oof = table.loc[k, 't_oof']
        t_add = t_oof + others['shift'].mean()
        t_log = expit(logit(t_oof) + others['logit shift'].mean())
        base = table.loc[k, 'F2 on H at t_oof']
        rows.append({'outer fold': k, 't_oof': t_oof, 'F2 at t_oof': base,
                     't additive': t_add, 'F2 additive': f2(s['h_y'], s['h_ens'], t_add),
                     't logit': t_log, 'F2 logit': f2(s['h_y'], s['h_ens'], t_log)})
    out = pd.DataFrame(rows).set_index('outer fold')
    out['gain additive'] = out['F2 additive'] - out['F2 at t_oof']
    out['gain logit'] = out['F2 logit'] - out['F2 at t_oof']
    return out


def pooled(scores, thresholds):
    """F2 over all held-out rows, each outer fold thresholded with its own value."""
    y = np.concatenate([s['h_y'] for s in scores])
    pred = np.concatenate([(s['h_ens'] >= t).astype(int) for s, t in zip(scores, thresholds)])
    return fbeta_score(y, pred, beta=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    train = labelled_train(args.data_root)
    y = train[TARGET].astype(int).to_numpy()
    for k, (r, h) in enumerate(outer_folds(y), 1):
        seconds = run_outer_fold(k, train.iloc[r].reset_index(drop=True), train.iloc[h].reset_index(drop=True), args.threads)
        print(f'outer fold {k}: ' + ('already done' if seconds is None else f'{seconds:.0f}s'), flush=True)
    scores = load_scores()
    table = per_fold_table(scores)
    honest = honest_correction(scores, table)
    pd.set_option('display.width', 200)
    print(table.round(4).to_string())
    print(honest.round(4).to_string())
    print(f"pooled F2 on H: t_oof {pooled(scores, honest['t_oof']):.4f} | additive {pooled(scores, honest['t additive']):.4f} | "
          f"logit {pooled(scores, honest['t logit']):.4f} | oracle {pooled(scores, table['t_best_H']):.4f}")


if __name__ == '__main__':
    main()
