"""Dataset and result preparation shared by hosted notebooks and local Python."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import shutil
import uuid
import zipfile
import zlib

from .common import TASKS


def _dataset_root(folder):
    folder = Path(folder).resolve()
    samples = [folder/'sample_submission.csv'] if (folder/'sample_submission.csv').is_file() else list(folder.rglob('sample_submission.csv'))
    roots = []
    for sample in samples:
        root = sample.parent
        expected = [root/TASKS[t][0]/f'{split}.csv' for t in (1, 2, 3, 5)
                    for split in (('test',) if t == 5 else ('train', 'test'))]
        if all(p.is_file() for p in expected):
            roots.append(root)
    if len(roots) != 1:
        raise ValueError('Select one competition data folder containing sample_submission.csv and '
                         'the original 01_binary_classification, 02_multiclass_classification, '
                         '03_regression and 05_anomaly_detection folders. '
                         f'Found {len(roots)} complete datasets under {folder}.')
    return roots[0]


def _safe_members(bundle, destination):
    members = []
    for item in bundle.infolist():
        # Normalize Windows ZIP names before checking on either host OS.
        relative = PurePosixPath(item.filename.replace('\\', '/'))
        if (relative.is_absolute() or '..' in relative.parts or
                any(':' in part for part in relative.parts) or
                ((item.external_attr >> 16) & 0o170000) == 0o120000):
            raise ValueError(f'Unsafe ZIP member: {item.filename}')
        target = destination.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f'ZIP member leaves the extraction folder: {item.filename}')
        members.append((item, target))
    return members


def prepare_dataset(source, work_dir):
    """Resolve a data folder, or copy one ZIP locally and extract it by content hash.

    The source is never modified. ZIP input is read once from Drive while copying
    and hashing; image extraction then uses local storage. Repeated extraction
    of an unchanged archive reuses the completed dataset in this runtime.
    """
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        return _dataset_root(source)
    if not source.is_file():
        raise FileNotFoundError(f'Data not found: {source}. Set DATA_PATH to the competition ZIP or extracted folder.')
    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    incoming = work/f'.incoming-{uuid.uuid4().hex}.zip'
    digest = hashlib.sha256()
    try:
        with source.open('rb') as src, incoming.open('xb') as dst:
            for chunk in iter(lambda: src.read(1024*1024), b''):
                digest.update(chunk)
                dst.write(chunk)
        if not zipfile.is_zipfile(incoming):
            raise ValueError('DATA_PATH must be an extracted competition folder or a ZIP archive.')
        signature = digest.hexdigest()
        destination = (work/('dataset-'+signature[:20])).resolve()
        destination.mkdir(exist_ok=True)
        marker = destination/'.dataset_ready.json'
        if marker.exists():
            cached = json.loads(marker.read_text(encoding='utf-8'))
            if cached.get('archive_sha256') == signature:
                return _dataset_root(destination)
        with zipfile.ZipFile(incoming) as bundle:
            members = _safe_members(bundle, destination)
            for item, target in members:
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if target.exists():
                    crc = 0
                    with target.open('rb') as existing:
                        for chunk in iter(lambda: existing.read(1024*1024), b''):
                            crc = zlib.crc32(chunk, crc)
                    if target.stat().st_size != item.file_size or crc & 0xffffffff != item.CRC:
                        raise FileExistsError(f'Extracted file differs: {target}. Use a new work_dir to keep both copies.')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        root = _dataset_root(destination)
        marker.write_text(json.dumps({'archive_sha256': signature, 'root': str(root.relative_to(destination))}),encoding='utf-8')
        return root
    finally:
        # This is a single temporary file created inside the verified work folder.
        if incoming.is_relative_to(work) and incoming.exists():
            incoming.unlink()


def create_run_directory(output_root, run_name=None):
    """Create a distinct run, or reopen an explicitly named run for inspection."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = run_name or (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', name):
        raise ValueError('RUN_NAME must be a single folder name using letters, digits, periods, hyphens or underscores.')
    destination = (root/name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError('RUN_NAME must remain within OUTPUT_ROOT.')
    destination.mkdir(exist_ok=True)
    return destination


def archive_results(output_dir):
    """Package team prediction files and summaries without model weights or OOF labels."""
    folder = Path(output_dir).resolve()
    names = {'task5_scores.csv', 'submission.csv', 'submission_manifest.json', 'RESULTS_TH.md',
             'validation_results.png', 'results_summary.json', 'environment_verified.json'}
    patterns = ('task[1-5]_predictions.csv', 'task[1-5]_metrics.json', 'run_*.json')
    files = {p for pattern in patterns for p in folder.glob(pattern)}
    if not any(folder.glob('task[1-5]_predictions.csv')) and not any(folder.glob('task[1-5]_metrics.json')):
        raise FileNotFoundError('No completed predictions or metrics are available to download.')
    files.update(folder/name for name in names if (folder/name).is_file())
    files = sorted(p for p in files if p.is_file() and not p.is_symlink())
    if not files:
        raise FileNotFoundError('No completed predictions or summaries are available to download.')
    path = folder/'results.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            bundle.write(file, file.name)
    return path
