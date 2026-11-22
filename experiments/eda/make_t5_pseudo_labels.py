r"""Task 5 pseudo-labels for experiments/eda/metric_recheck.ipynb (Task 5 has no train labels).

Exports pis/, Task5/ and run.py from origin/task5-boost (the v2 detector, PR #22) into a
temporary folder with `git archive` (nothing is merged or checked out), runs it on Task 5
test.csv and copies task5_scores.csv to outputs/experiments/metric_recheck/ (gitignored).

    .\.venv\Scripts\python.exe experiments/eda/make_t5_pseudo_labels.py

The v2 README reports 2,097 flags covering 2,046 of the 2,049 rows in the 8 anomaly groups
found by EDA, identical across seeds and profiles; its flags serve as stand-in labels and its
posteriors give label-free expected scores.
"""
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'run.py').is_file())
BRANCH = 'origin/task5-boost'

if __name__ == '__main__':
    out = ROOT / 'outputs' / 'experiments' / 'metric_recheck'
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(['git', 'archive', BRANCH, 'pis', 'Task5', 'run.py'], cwd=ROOT,
                                 check=True, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp)
        subprocess.run([sys.executable, 'Task5/train.py', '--data-root', str(ROOT.parent / 'cpe342-kaggle-main'),
                        '--profile', 'strong', '--contamination', 'auto', '--seed', '342',
                        '--output-dir', str(Path(tmp) / 'out')], cwd=tmp, check=True)
        shutil.copy(Path(tmp) / 'out' / 'task5_scores.csv', out / 'task5_v2_scores.csv')
    print('saved', out / 'task5_v2_scores.csv')
