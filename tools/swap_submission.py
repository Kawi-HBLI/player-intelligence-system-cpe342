#!/usr/bin/env python
"""Swap one or more task columns into an existing full Kaggle submission.

Why this exists: each teammate owns one task. The team's best Kaggle file is a
mix of everyone's best columns, so improving one task should mean replacing
only that column and leaving every other column exactly as it was.

    python tools/swap_submission.py --base best.csv --from other.csv --tasks task4
    python tools/swap_submission.py --base best.csv --pred task3=task3_predictions.csv
    (both modes can be combined in one call; see tools/README.md)

The tool never edits the base file, never touches task code, and refuses to
leave an output behind that the repo validator rejects.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Repo imports --------------------------------------------------------------
# The tool lives in tools/, so put the repo root on sys.path to import pis/.
# Mapping and validation come from pis/ so that a swapped column is built
# exactly like a column made by `python run.py assemble`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pis.common import DEFAULT_DATA, TASKS, image_folder  # noqa: E402
from pis.submission import (  # noqa: E402
    _ids,  # private helper, used on purpose: pis/ must not be edited for this tool.
    read_template,
    task4_image_keys,
    task4_to_index,
    validate_file,
)

COLUMNS = ['id'] + [f'task{i}' for i in range(1, 6)]
CLASS_LABELS = {'task1': {0, 1}, 'task2': {0, 1, 2, 3}, 'task4': {0, 1, 2, 3, 4}, 'task5': {0, 1}}


class SwapError(Exception):
    """A user-facing problem: printed without a traceback, exit code 1."""


# --- Reading a full submission as raw text --------------------------------------
# Every column is read as a string (no NaN parsing, no float conversion) so the
# untouched columns can be written back byte for byte. Separator, UTF-8 BOM,
# line ending and the trailing newline are recorded and reused when writing.

def read_submission(path):
    path = Path(path)
    if not path.is_file():
        raise SwapError(f'File not found: {path}')
    raw = path.read_bytes()
    bom = raw.startswith(b'\xef\xbb\xbf')
    text = raw.decode('utf-8-sig')
    header = text.split('\n', 1)[0]
    style = {
        'sep': '|' if '|' in header else ',',  # same rule as pis.submission.read_template
        'bom': bom,
        'newline': '\r\n' if header.endswith('\r') else '\n',
        'final_newline': text.endswith('\n'),
    }
    frame = pd.read_csv(path, sep=style['sep'], dtype=str, keep_default_na=False,
                        na_filter=False, encoding='utf-8-sig')
    if list(frame.columns) != COLUMNS:
        raise SwapError(f'{path.name}: columns are {list(frame.columns)}, expected {COLUMNS}')
    if not frame['id'].is_unique or (frame['id'] == '').any():
        raise SwapError(f'{path.name}: id column has duplicates or blanks')
    return frame, style


def write_submission(frame, style, path):
    # Written by hand (not DataFrame.to_csv) so that only the swapped strings
    # differ: no re-quoting, no float reformatting of untouched columns.
    lines = [style['sep'].join(frame.columns)]
    lines += [style['sep'].join(row) for row in frame.itertuples(index=False, name=None)]
    body = style['newline'].join(lines) + (style['newline'] if style['final_newline'] else '')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode('utf-8-sig' if style['bom'] else 'utf-8'))


def raw_fields(path, sep):
    """Each data line split into its literal text fields (used for the byte check)."""
    text = Path(path).read_bytes().decode('utf-8-sig')
    return [line.rstrip('\r').split(sep) for line in text.split('\n') if line != '']


# --- Task 4 label format ---------------------------------------------------------
# Kaggle scores Task 4 as integers 0-4. A submission with text labels
# ("freefried") scored lower, so a column is only accepted in the base format.

def is_numeric_column(values):
    return pd.to_numeric(pd.Series(values), errors='coerce').notna().all()


# --- Comparing two versions of a column ---------------------------------------------
# One rule for "same value", shared by the swap report and the --from note:
# task3 uses a tight isclose (files saved with fewer digits differ only in the
# last bits), class columns compare exactly. astype(float) uses Python's exact
# float parser. Text that is not a number (e.g. task4 "pube") falls back to a
# plain string comparison, so this never raises.

def same_values(task, before, after):
    before, after = np.asarray(before, dtype=object), np.asarray(after, dtype=object)
    try:
        b = pd.Series(before).astype(float).to_numpy()
        a = pd.Series(after).astype(float).to_numpy()
    except ValueError:
        return before == after
    return np.isclose(b, a, rtol=1e-9, atol=1e-9) if task == 'task3' else (b == a)


# --- Mode 1: copy columns from another full submission ----------------------------
# Rows are matched by id, never by position, so a file sorted differently still
# lines up correctly. Values are copied as their original text.

def align_other(base, other, other_path):
    """Return the --from file reordered to the base's id order."""
    if len(other) != len(base) or set(other['id']) != set(base['id']):
        raise SwapError(f'{other_path.name}: ids differ from the base file '
                        f'({len(other)} vs {len(base)} rows); cannot match rows by id')
    return other.set_index('id').loc[base['id']].reset_index()


def note_other_differences(base, other, plan):
    # Informational only, never fails. A teammate's file may have been built on
    # an older base, so other columns can differ too; say so, so nobody assumes
    # those columns are in the output (they are not: only --tasks are copied).
    differ = {task: int((~same_values(task, base[task], other[task])).sum())
              for task in COLUMNS[1:] if task not in plan}
    differ = {task: n for task, n in differ.items() if n}
    if not differ:
        print('--from matches the base in all other columns')
        return
    copied = ', '.join(task for task, (kind, _) in plan.items() if kind == 'from')
    print('Note: --from also differs from the base in columns you are NOT swapping:')
    for task, n in differ.items():
        print(f'  {task}: {n:,} rows differ')
    print(f'  (only {copied} {"is" if "," not in copied else "are"} copied; these differences are ignored)')


# --- Mode 2: map a per-task prediction file to ANS ids -----------------------------
# pis.submission.assemble() does this mapping inline (there is no public per-task
# function), so the steps below follow its loop body one to one, with the same
# helpers: tabular tasks use test.csv row order (ANS row i = test.csv row i,
# looked up by player_id), Task 4 joins on the file_name column and, when the
# fixed test.csv has an ANS `id` column, on that id.

def column_from_prediction(task_num, pred_path, data_root, image_root, sample):
    root = Path(data_root)
    if not pred_path.is_file():
        raise SwapError(f'File not found: {pred_path}')
    prediction = pd.read_csv(pred_path, dtype={'player_id': str})  # same read as assemble()
    folder, _ = TASKS[task_num]
    try:
        task_root = image_folder(root, image_root) if task_num == 4 else root / folder
    except FileNotFoundError as exc:
        raise SwapError(f'Task 4: {exc}') from None
    test_path = task_root / 'test.csv'
    if not test_path.is_file():
        raise SwapError(f'Task {task_num}: {test_path} not found (set --data-root / --image-root)')
    test = pd.read_csv(test_path)
    ids = task4_image_keys(test) if task_num == 4 else _ids(test)
    if len(ids) != len(sample) or not ids.is_unique:
        raise SwapError(f'Task {task_num}: wrong test count or duplicate IDs in {test_path}')
    if not {'player_id', 'prediction'} <= set(prediction.columns) or not prediction.player_id.is_unique:
        raise SwapError(f'Task {task_num}: {pred_path.name} needs unique player_id and a prediction column')
    if set(ids) != set(prediction.player_id) or len(prediction) != len(ids):
        raise SwapError(f'Task {task_num}: {pred_path.name} ids do not match this task test.csv')
    values = prediction.set_index('player_id').loc[ids, 'prediction'].to_numpy()
    if task_num == 4:
        try:
            values = task4_to_index(values)
        except ValueError as exc:
            raise SwapError(str(exc)) from None
        if 'file_name' in test and 'id' in test:
            by_answer = pd.Series(values, index=test['id'].astype(str))
            if not by_answer.index.is_unique or set(by_answer.index) != set(sample.id.astype(str)):
                raise SwapError('Task 4: test.csv ids do not match the submission template ids')
            values = by_answer.loc[sample.id.astype(str)].to_numpy()
    if task_num == 3:
        # Re-read Task 3 as text and parse with Python's float(): it gives the
        # exact double that was written on every pandas version (pandas' fast
        # parser can drift in the last bits). repr() writes the shortest text
        # that parses back to that same double, so no precision is lost.
        text = pd.read_csv(pred_path, dtype=str, keep_default_na=False).set_index('player_id')
        return pd.Series([_to_float_text(v) for v in text.loc[ids, 'prediction']],
                         index=sample.id.astype(str))
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise SwapError(f'Task {task_num}: fractional, missing or nonnumeric class label')
    return pd.Series([str(int(v)) for v in numeric], index=sample.id.astype(str))


def _to_float_text(value):
    try:
        return repr(float(value))
    except ValueError:
        return ''  # caught by check_column as "missing or not numeric"


# --- Per-task checks ----------------------------------------------------------------
# Run on the new column before anything is written, so a bad file is caught with
# a message that names the task, not just by the generic validator afterwards.

def check_column(task, new_values, base_values):
    numbers = pd.to_numeric(pd.Series(new_values), errors='coerce')
    if task == 'task3':
        if numbers.isna().any():
            raise SwapError('task3: some values are missing or not numeric')
        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise SwapError('task3: values must be finite and >= 0')
        return
    if task == 'task4':
        base_numeric = is_numeric_column(base_values)
        new_numeric = numbers.notna().all()
        if base_numeric != new_numeric:
            raise SwapError(
                'task4: label format mismatch. The base column is '
                f'{"numeric 0-4" if base_numeric else "text"} but the new column is '
                f'{"numeric" if new_numeric else "text"}. Kaggle scores Task 4 as numeric 0-4 '
                '(a text submission scored lower), so convert the labels first.')
        if not base_numeric:
            raise SwapError('task4: the base column holds text labels; Kaggle expects numeric 0-4. '
                            'Fix the base file first.')
    if numbers.isna().any():
        raise SwapError(f'{task}: some values are missing or not numeric')
    if not np.equal(numbers, np.floor(numbers)).all() or not set(numbers.astype(int)) <= CLASS_LABELS[task]:
        raise SwapError(f'{task}: labels must be in {sorted(CLASS_LABELS[task])}, '
                        f'found {sorted(set(numbers))}')


# --- Report -------------------------------------------------------------------------

def describe(task, values):
    numbers = pd.to_numeric(pd.Series(values), errors='coerce')
    if task == 'task3':
        return f'sum={numbers.sum():,.2f}  median={numbers.median():,.4f}  zero_share={(numbers == 0).mean():.2%}'
    counts = numbers.astype(int).value_counts().sort_index()
    return '  '.join(f'{label}:{count} ({count / len(numbers):.1%})' for label, count in counts.items())


def report(task, before, after):
    # Rows that differ only in formatting or float rounding (see same_values)
    # are counted separately instead of being reported as real changes.
    same = same_values(task, before, after)
    changed = int((~same).sum())
    a = np.asarray(after)
    text_only = int(((np.asarray(before) != np.asarray(after)) & same).sum())
    print(f'\n[{task}] rows changed: {changed:,} / {len(a):,}'
          + (f'  (+{text_only:,} rows differ only in formatting / float rounding)' if text_only else ''))
    print(f'  before: {describe(task, before)}')
    print(f'  after : {describe(task, after)}')


# --- Command line ---------------------------------------------------------------------

def task_name(text):
    name = text.strip().lower()
    name = name if name.startswith('task') else f'task{name}'
    if name not in COLUMNS[1:]:
        raise SwapError(f'Unknown task {text!r}; use task1..task5')
    return name


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Swap task columns into a full Kaggle submission.')
    parser.add_argument('--base', type=Path, required=True, help='full submission (id, task1..task5); never modified')
    parser.add_argument('--from', dest='other', type=Path, help='another full submission to copy columns from')
    parser.add_argument('--tasks', help='comma list of columns to copy from --from, e.g. task4,task5')
    parser.add_argument('--pred', action='append', default=[], metavar='taskN=PATH',
                        help='per-task prediction csv (player_id, prediction); repeatable')
    parser.add_argument('--out', type=Path, help='output file (default outputs/kaggle/<base>__<tasks>.csv)')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA, help='Kaggle data folder (for --pred and validation)')
    parser.add_argument('--image-root', type=Path, help='Task 4 dataset root, if not under --data-root')
    return parser.parse_args(argv)


def plan_swaps(args):
    """Return {task: ('from', None) | ('pred', path)}; a task named twice is an error."""
    if bool(args.other) != bool(args.tasks):
        raise SwapError('--from and --tasks must be given together')
    plan = {}

    def add(task, source):
        if task in plan:
            raise SwapError(f'{task} is given more than once; choose one source per task')
        plan[task] = source

    for item in (args.tasks.split(',') if args.tasks else []):
        if item.strip():
            add(task_name(item), ('from', None))
    for item in args.pred:
        if '=' not in item:
            raise SwapError(f'--pred expects taskN=path, got {item!r}')
        task, path = item.split('=', 1)
        add(task_name(task), ('pred', Path(path)))
    if not plan:
        raise SwapError('Nothing to swap: give --from with --tasks, and/or --pred taskN=path')
    return dict(sorted(plan.items()))


def main(argv=None):
    args = parse_args(argv)
    plan = plan_swaps(args)
    base, style = read_submission(args.base)
    try:
        sample, _ = read_template(args.data_root)
    except FileNotFoundError:
        raise SwapError(f'sample_submission.csv not found under {args.data_root} (set --data-root)') from None
    if set(sample.id.astype(str)) != set(base['id']):
        raise SwapError(f'{args.base.name}: ids differ from the template in {args.data_root}')

    # Output path: default name records which columns were swapped. Refuse to
    # write over any input, so the base (often the best Kaggle file) is safe.
    out = args.out or REPO_ROOT / 'outputs' / 'kaggle' / f'{args.base.stem}__{"_".join(plan)}.csv'
    inputs = [args.base] + ([args.other] if args.other else []) + [p for kind, p in plan.values() if p]
    if any(out.resolve() == Path(p).resolve() for p in inputs):
        raise SwapError(f'--out {out} would overwrite an input file; choose another path')

    other = None
    if args.other:
        other = align_other(base, read_submission(args.other)[0], args.other)
        note_other_differences(base, other, plan)
    result = base.copy()
    for task, (kind, path) in plan.items():
        if kind == 'from':
            new = other[task].to_numpy(dtype=object)
        else:
            mapped = column_from_prediction(int(task[-1]), path, args.data_root, args.image_root, sample)
            new = mapped.loc[base['id']].to_numpy(dtype=object)  # align by id, not position
        check_column(task, new, base[task].to_numpy())
        result[task] = new
        report(task, base[task].to_numpy(), new)

    write_submission(result, style, out)

    # --- Post-write checks -------------------------------------------------------
    # 1) Untouched columns: compare the literal text of every field in the output
    #    with the base file (header included), so "byte for byte" is checked, not assumed.
    # 2) Repo validator: the same check `python run.py validate --file` runs.
    # Either failure deletes the output so nobody uploads a broken file.
    try:
        base_rows, out_rows = raw_fields(args.base, style['sep']), raw_fields(out, style['sep'])
        keep = [i for i, col in enumerate(COLUMNS) if col not in plan]
        if len(base_rows) != len(out_rows) or any(
                [b[i] for i in keep] != [o[i] for i in keep] for b, o in zip(base_rows, out_rows)):
            raise SwapError('Untouched columns are not identical to the base file')
        try:
            validate_file(out, args.data_root, args.image_root)
        except ValueError as exc:
            raise SwapError(f'Repo validator rejected the output: {exc}') from None
    except SwapError:
        out.unlink(missing_ok=True)
        raise
    untouched = ', '.join(c for c in COLUMNS[1:] if c not in plan)
    print(f'\nUntouched columns identical to base: {untouched or "(none)"}')
    print(f'Validator: OK ({len(result):,} rows)')
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SwapError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
