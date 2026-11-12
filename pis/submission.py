"""Submission assembly: align within each task, never join different player pools."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .common import TASKS, write_json, image_folder

# Kaggle Overview and the Task 4 datacard define the submitted classes as integers:
# 0 7-Eleven Knight, 1 FiveN, 2 Free Fried, 3 PUBE Mobile, 4 Roblock.
TASK4_CLASS_INDEX = {'7-11k': 0, 'fiveN': 1, 'freefried': 2, 'pube': 3, 'roblock': 4}


def task4_to_index(values):
    """Map Task 4 train.csv text labels to the integer classes Kaggle scores."""
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()
    unknown = sorted(set(series) - set(TASK4_CLASS_INDEX))
    if unknown:
        raise ValueError(f'Task 4: unknown labels {unknown}; expected {sorted(TASK4_CLASS_INDEX)}')
    return series.map(TASK4_CLASS_INDEX).to_numpy(dtype=int)


def read_template(data_root):
    path = Path(data_root) / 'sample_submission.csv'
    with path.open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    sep = '|' if '|' in header else ','
    sample = pd.read_csv(path, sep=sep)
    expected = ['id'] + [f'task{i}' for i in range(1, 6)]
    if list(sample.columns) != expected or not sample.id.is_unique or sample.id.isna().any():
        raise ValueError('Invalid template schema or IDs')
    return sample, sep


def _ids(frame):
    for col in ('player_id', 'id', 'image_id', 'filename'):
        if col in frame:
            return frame[col].astype(str)
    raise ValueError('Cannot identify task ID column')


def task4_image_keys(frame):
    """Image key (file stem) per row of a Task 4 csv, for both csv layouts.

    Old layout: `id` holds the image name. New layout (12 Nov data fix): `id` holds the
    ANS/PLY code and `file_name` holds the image file, so `file_name` wins when present.
    """
    column = 'file_name' if 'file_name' in frame else next(
        (c for c in ('id', 'image_id', 'filename') if c in frame), None)
    if column is None:
        raise ValueError('Cannot identify the Task 4 image column')
    return frame[column].astype(str).str.replace(r'\.[A-Za-z0-9]+$', '', regex=True)


def _allowed(data_root, task, image_root=None):
    if task in (1, 5):
        return {0, 1}
    if task == 2:
        return {0, 1, 2, 3}
    return set(TASK4_CLASS_INDEX.values())


def validate_frame(frame, data_root, image_root=None):
    template, _ = read_template(data_root)
    if list(frame.columns) != list(template.columns) or len(frame) != len(template):
        raise ValueError('Submission column order or row count differs from template')
    if frame.isna().any().any():
        raise ValueError('Submission contains missing predictions')
    if frame.id.astype(str).tolist() != template.id.astype(str).tolist():
        raise ValueError('Submission IDs or their order differ from template')
    for task in (1, 2, 4, 5):
        allowed = _allowed(data_root, task, image_root)
        if not set(frame[f'task{task}'].unique()) <= allowed:
            raise ValueError(f'Task {task} has invalid labels; expected {allowed}')
    spending = pd.to_numeric(frame.task3, errors='raise').to_numpy()
    if not np.isfinite(spending).all() or (spending < 0).any():
        raise ValueError('Task 3 predictions must be finite, nonnegative THB')
    return {'rows': len(frame), 'columns': list(frame.columns), 'valid': True}


def assemble(data_root, output_dir, image_root=None):
    root, out = Path(data_root), Path(output_dir)
    sample, sep = read_template(root)
    result = sample[['id']].copy()
    sources = {}
    # Fail before writing a submit-ready file if any task is absent.
    for task, (folder, _) in TASKS.items():
        path = out / f'task{task}_predictions.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path.name}; train Task {task} first. No final submission was written.')
        prediction = pd.read_csv(path, dtype={'player_id': str})
        task_root = image_folder(root, image_root) if task == 4 else root / folder
        test = pd.read_csv(task_root / 'test.csv')
        ids = task4_image_keys(test) if task == 4 else _ids(test)
        if len(ids) != len(sample) or not ids.is_unique:
            raise ValueError(f'Task {task}: wrong test count or duplicate IDs')
        if not {'player_id', 'prediction'} <= set(prediction.columns) or not prediction.player_id.is_unique:
            raise ValueError(f'Task {task}: invalid prediction ID schema')
        if set(ids) != set(prediction.player_id) or len(prediction) != len(ids):
            raise ValueError(f'Task {task}: prediction IDs do not match this task test.csv')
        values = prediction.set_index('player_id').loc[ids, 'prediction'].to_numpy()
        if task == 4:
            values = task4_to_index(values)
            if 'file_name' in test and 'id' in test:
                # The fixed test.csv names the ANS id of every image: join on it
                # instead of trusting row order.
                by_answer = pd.Series(values, index=test['id'].astype(str))
                if not by_answer.index.is_unique or set(by_answer.index) != set(sample.id.astype(str)):
                    raise ValueError('Task 4: test.csv ids do not match the submission template ids')
                values = by_answer.loc[sample.id.astype(str)].to_numpy()
        if task != 3 and pd.api.types.is_numeric_dtype(values):
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f'Task {task}: fractional or nonfinite class label')
            values = values.astype(int)
        result[f'task{task}'] = values
        sources[f'task{task}'] = {'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'test_sha256': hashlib.sha256((task_root/'test.csv').read_bytes()).hexdigest()}
    check = validate_frame(result, root, image_root)
    path = out / 'submission.csv'
    result.to_csv(path, index=False, sep=sep)
    validate_frame(pd.read_csv(path, sep=sep), root, image_root)
    manifest = {**check, 'separator': sep, 'file': str(path.resolve()), 'sources': sources,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'alignment': 'ANS row i uses row i of each task test.csv. Inferred convention; no explicit ANS/player mapping supplied.',
                'scope': 'Local format checks passed; not uploaded to Kaggle and evaluator acceptance unverified.'}
    write_json(out / 'submission_manifest.json', manifest)
    return manifest


def validate_file(path, data_root, image_root=None):
    _, sep = read_template(data_root)
    return validate_frame(pd.read_csv(path, sep=sep), data_root, image_root)
