"""Inventory and data quality checks. Does not change the competition data."""
from pathlib import Path
import hashlib
import pandas as pd
from .common import TASKS, environment, write_json, image_folder


def audit_data(data_root, output_dir, image_root=None):
    root, out = Path(data_root), Path(output_dir)
    report = {'environment': environment(), 'tasks': {}, 'warnings': []}
    template_path = root / 'sample_submission.csv'
    with template_path.open(encoding='utf-8-sig') as handle:
        header = handle.readline()
    separator = '|' if '|' in header else ','
    template = pd.read_csv(template_path, sep=separator)
    report['submission'] = {'rows': len(template), 'columns': list(template.columns), 'separator': separator,
                            'unique_ids': bool(template['id'].is_unique),
                            'sha256': hashlib.sha256(template_path.read_bytes()).hexdigest()}
    test_ids = {}
    for number, (folder, target) in TASKS.items():
        info = {}
        task_root = root / folder
        if number == 4 and image_root is not None:
            task_root = image_folder(root, image_root)
        for split in ('train', 'test'):
            path = task_root / f'{split}.csv'
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            id_col = 'player_id' if 'player_id' in frame else 'id' if 'id' in frame else frame.columns[0]
            item = {'rows': len(frame), 'columns': list(frame.columns), 'id_column': id_col,
                    'dtypes': {col: str(dtype) for col, dtype in frame.dtypes.items()},
                    'unique_ids': bool(frame[id_col].is_unique),
                    'missing_per_column': {k: int(v) for k, v in frame.isna().sum().items() if v},
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if target in frame:
                y = frame[target]
                item['missing_target_rows'] = int(y.isna().sum())
                if number == 3:
                    item['target_summary'] = {k: float(v) for k, v in y.describe().items()}
                    item['target_zero_fraction'] = float(y.eq(0).mean())
                else:
                    item['target_counts'] = {str(k): int(v) for k, v in y.value_counts().items()}
            if split == 'test':
                test_ids[number] = set(frame[id_col].astype(str))
                item['matches_submission_row_count'] = len(frame) == len(template)
            info[split] = item
        if number == 4:
            files = [p for p in task_root.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            info['image_count'] = len(files)
            if files:
                from PIL import Image
                samples = []
                for path in sorted(files)[:5]:
                    with Image.open(path) as img:
                        samples.append({'file': str(path.relative_to(task_root)),
                                        'width_px': img.width, 'height_px': img.height, 'mode': img.mode})
                info['image_sample'] = samples
            if not files or 'train' not in info or 'test' not in info:
                report['warnings'].append('Task 4 image dataset is incomplete; follow its README download link.')
        report['tasks'][str(number)] = info
    report['test_player_id_intersections'] = {f'{a}-{b}': len(test_ids[a] & test_ids[b])
                                             for a in test_ids for b in test_ids if a < b}
    report['warnings'].extend([
        'Submission ANS IDs have no supplied player-ID key. Assembly uses each task test.csv row order; confirm this convention with competition organizers.',
        'Web text and local template disagree on separator and Task 4 label mapping. Template delimiter and actual training labels are authoritative locally; live evaluator is not verified.',
        'Task 5 has only unlabeled test.csv. Its specific unsupervised task implies fitting that dataset, but this conflicts with the general no-test-training slide. Confirm the exception with the TA.',
    ])
    write_json(out / 'data_audit.json', report)
    return report
