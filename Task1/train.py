"""Task 1 entry point: anti-cheat detection (binary classification, F2 of the cheater class).

Run from the repository root:  python Task1/train.py
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.

Task 1 uses 10 folds unless --folds is given: each fold model then trains on 90% of the
labelled rows instead of 67%, which raised OOF macro F2 from 0.7807 to 0.7818 (see
experiments/task1/experiments_round3.py). The test prediction is the mean of the 10 fold models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

TASK1_FOLDS = '10'

if __name__ == '__main__':
    args = sys.argv[1:]
    if not any(a == '--folds' or a.startswith('--folds=') for a in args):
        args = ['--folds', TASK1_FOLDS, *args]
    main(['train', '--tasks', '1', *args])
