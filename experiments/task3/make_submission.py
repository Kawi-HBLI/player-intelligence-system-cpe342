"""Build a Kaggle submission that changes ONLY the Task 3 column of an existing one.

Kaggle gives one score for all 5 tasks. To measure a Task 3 change on the leaderboard,
start from a submission whose score we already know and swap in new Task 3 predictions:
any score change then comes from Task 3 alone.

Run from the repo root:
    python experiments/task3/make_submission.py --base <submission.csv> [--task3 <task3_predictions.csv>] [--out <file>]
--task3 defaults to the current champion's test predictions
(experiments/task3/champion/champion.json -> outputs/experiments/task3_<variant>/task3_predictions.csv).
--out defaults to outputs/kaggle/<base-stem>_task3_<variant>.csv. The result is checked
with the repo's validator (pis/submission.py); the script exits non-zero if it fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Make `pis` importable no matter where the script is launched from ---
# Reuse the project's data path, template reader and validator, so the file is checked
# by exactly the same rules as a submission built with `python run.py assemble`.
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').exists())
sys.path.insert(0, str(ROOT))
from pis.common import DEFAULT_DATA  # noqa: E402
from pis.submission import read_template, validate_file  # noqa: E402
from pis.tabular import _score  # noqa: E402

# --- Settings ---
TASK3 = Path(__file__).resolve().parent
CHAMPION = TASK3 / 'champion'
OUTPUTS = ROOT / 'outputs'
COLUMNS = ['id'] + [f'task{i}' for i in range(1, 6)]
N_ROWS = 25_889


def fail(message):
    sys.exit(f'ERROR: {message}')


# --- Which Task 3 predictions to use ---
# Default: the champion. Its OOF is tracked in git, but its test predictions live in
# outputs/ (gitignored), so they exist only after amount_model.py has been run locally.
# A --quick run writes to the same folder, so we also re-score that run's OOF and
# require it to match the champion: otherwise we could submit a 250-iteration model
# under the champion's name.

def champion_predictions():
    meta = json.loads((CHAMPION / 'champion.json').read_text(encoding='utf-8'))
    folder = OUTPUTS / 'experiments' / f'task3_{meta["variant"]}'
    path, oof_path = folder / 'task3_predictions.csv', folder / 'task3_oof.csv'
    if not path.exists() or not oof_path.exists():
        fail(f'{path.relative_to(ROOT)} not found. Run amount_model.py first: '
             f'python experiments/task3/amount_model/amount_model.py --variants {meta["variant"]}')
    oof = pd.read_csv(oof_path)
    local = _score(oof['target'].to_numpy(), oof['prediction'].to_numpy())
    if abs(local - meta['oof']) > 1e-4:
        fail(f'{oof_path.relative_to(ROOT)} scores OOF {local:.4f} but the champion has {meta["oof"]:.4f}; '
             f'outputs/ was overwritten by another run (e.g. --quick). Re-run: '
             f'python experiments/task3/amount_model/amount_model.py --variants {meta["variant"]}')
    return path, meta['variant']


def variant_name(path):
    """Label for the output name: `v3` for outputs/experiments/task3_v3/..., else the file stem."""
    parent = path.resolve().parent.name
    return parent[len('task3_'):] if parent.startswith('task3_') else path.stem


def summary(values):
    return f'sum {values.sum():,.0f} | median {np.median(values):,.2f} | zero share {(values == 0).mean():.3f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base', type=Path, required=True, help='full submission (id, task1..task5)')
    parser.add_argument('--task3', type=Path, help='player_id,prediction in 03_regression/test.csv order '
                                                    '(default: the champion)')
    parser.add_argument('--out', type=Path, help='default: outputs/kaggle/<base-stem>_task3_<variant>.csv')
    args = parser.parse_args()

    task3_path, variant = (args.task3, variant_name(args.task3)) if args.task3 else champion_predictions()
    out = args.out or OUTPUTS / 'kaggle' / f'{args.base.stem}_task3_{variant}.csv'

    # --- Never touch the inputs ---
    # baseline/ is the team's tracked reference, and --base is often the only copy of a
    # scored Kaggle file; neither may be overwritten by a typo in --out.
    if not args.base.exists():
        fail(f'--base {args.base} not found')
    if out.resolve() == args.base.resolve():
        fail('--out must differ from --base')
    if out.resolve().is_relative_to((ROOT / 'baseline').resolve()):
        fail('--out must not be inside baseline/')

    # --- Read the base as text ---
    # Every column is read as a string so the 4 untouched columns are written back exactly
    # as they came (no int -> float, no re-formatted numbers). Same separator detection as
    # pis/submission.py; utf-8-sig drops a BOM that downloaded files sometimes carry.
    template, sep = read_template(DEFAULT_DATA)
    base = pd.read_csv(args.base, sep=sep, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if list(base.columns) != COLUMNS or len(base) != N_ROWS:
        fail(f'--base must have columns {COLUMNS} and {N_ROWS:,} rows; got {list(base.columns)}, {len(base):,} rows')
    if base['id'].tolist() != template['id'].astype(str).tolist():
        fail('--base ids are not in sample_submission.csv order')

    # --- Map Task 3 predictions onto ANS ids ---
    # Same convention as pis/submission.py: row i of 03_regression/test.csv is ANS row i.
    # The predictions must already be in test.csv order (amount_model.py writes them so);
    # we check rather than re-sort, because a different order means a different file.
    test_ids = pd.read_csv(DEFAULT_DATA / '03_regression' / 'test.csv', usecols=['player_id'])['player_id']
    # round_trip: pandas' default fast float parser can be off by one ulp, which would make
    # the submitted values differ (invisibly) from the predictions file.
    pred = pd.read_csv(task3_path, float_precision='round_trip')
    if list(pred.columns) != ['player_id', 'prediction'] or len(pred) != N_ROWS:
        fail(f'{task3_path} must have columns player_id,prediction and {N_ROWS:,} rows')
    if not (pred['player_id'].astype(str).to_numpy() == test_ids.astype(str).to_numpy()).all():
        fail(f'{task3_path} is not in 03_regression/test.csv player_id order')
    if pred['prediction'].isna().any():
        fail(f'{task3_path} has NaN predictions')
    new = np.maximum(0.0, pred['prediction'].to_numpy(dtype=float))  # THB can't be negative

    # --- Write: swap task3 only ---
    old = pd.to_numeric(base['task3']).to_numpy(dtype=float)
    result = base.copy()
    result['task3'] = new  # float column -> written with full precision; others stay text
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, sep=sep, lineterminator='\n')

    print(f'base:  {args.base}')
    print(f'task3: {task3_path} (variant {variant})')
    print(f'rows:  {len(result):,}')
    print(f'old task3: {summary(old)}')
    print(f'new task3: {summary(new)}')

    # --- Validate with the repo's own checker; an invalid file is removed ---
    # Same rules as `python run.py validate --file`. Deleting a failed file means it can't
    # be uploaded by mistake.
    try:
        check = validate_file(out, DEFAULT_DATA)
    except ValueError as error:
        out.unlink()
        fail(f'validation failed, file removed: {error}')
    written = pd.read_csv(out, sep=sep, dtype=str, keep_default_na=False)
    if not written.drop(columns='task3').equals(base.drop(columns='task3')):
        out.unlink()
        fail('columns other than task3 changed, file removed')
    print(f'valid: {check["valid"]} ({check["rows"]:,} rows); other columns identical to base')
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
