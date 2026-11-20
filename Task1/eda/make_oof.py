r"""Save OOF predictions of the current Task 1 model for the missing-value EDA.

Runs the unchanged pipeline (pis.tabular.train_task through run.py) with the folds the
README tables use (StratifiedKFold 3, seed 342) and copies the OOF to
outputs/experiments/task1_oof.csv (gitignored). README: AUC 0.8926, tuned macro F2 0.7807; this
laptop gives AUC 0.8922 / 0.7808, and so does the README's own Task1/experiments.py here
(environment difference, see the notebook).

    .\.venv\Scripts\python.exe Task1/eda/make_oof.py --threads 12

CatBoost on CPU builds slightly different trees with a different thread count, so keep
--threads the same between runs.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run import main  # noqa: E402

TASK = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args()
    run_dir = ROOT / 'outputs' / 'experiments' / f'task{TASK}_missing_eda_run'
    main(['train', '--tasks', str(TASK), '--folds', '3', '--seed', '342', '--threads', str(args.threads),
          '--output-dir', str(run_dir)])
    shutil.copy(run_dir / f'task{TASK}_oof.csv', ROOT / 'outputs' / 'experiments' / f'task{TASK}_oof.csv')
    print('saved', ROOT / 'outputs' / 'experiments' / f'task{TASK}_oof.csv')
