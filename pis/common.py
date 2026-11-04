from pathlib import Path
import json
import platform
import importlib.metadata

TASKS = {
    1: ('01_binary_classification', 'is_cheater'),
    2: ('02_multiclass_classification', 'segment'),
    3: ('03_regression', 'spending_30d'),
    4: ('04_image_classification', 'class'),
    5: ('05_anomaly_detection', None),
}
SOLUTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = SOLUTION_ROOT.parent / 'cpe342-kaggle-main'
DEFAULT_OUTPUT = SOLUTION_ROOT / 'outputs' / 'main'


def image_folder(data_root, image_root=None):
    base = Path(image_root) if image_root is not None else Path(data_root) / TASKS[4][0]
    if (base/'train.csv').is_file() and (base/'test.csv').is_file():
        return base
    candidates = [p.parent for p in base.rglob('train.csv')
                  if (p.parent/'test.csv').exists() and
                  ((p.parent/'train').is_dir() or (p.parent/'data'/'train').is_dir())] if base.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f'Cannot locate unique Task 4 metadata/images under {base}; set --image-root explicitly.')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def environment():
    packages = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'torch', 'torchvision', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
