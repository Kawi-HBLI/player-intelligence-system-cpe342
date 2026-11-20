r"""Save OOF predictions of the current Task 1 model for the missing-value EDA.

Runs the unchanged pipeline (pis.tabular.train_task through run.py) with the folds the
README tables use (StratifiedKFold 3, seed 342) and copies the OOF to
outputs/experiments/task1_oof.csv (gitignored). This laptop gives AUC 0.8922 (README 0.8926, an
environment difference: the README's own Task1/experiments.py gives 0.8922 here too). Since
task1-2 commit 080d9a0 the threshold maximises cheater-class F2: threshold 0.1805, F2 0.8381.

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
