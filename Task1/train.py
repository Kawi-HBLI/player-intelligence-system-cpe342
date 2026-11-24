"""Task 1 entry point: anti-cheat detection (binary classification, F2 of the cheater class).

Run from the repository root:  python Task1/train.py --threads 24
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.

Task 1 uses 10 folds unless --folds is given: each fold model then trains on 90% of the
labelled rows instead of 67%, which raised OOF macro F2 from 0.7807 to 0.7818 (see
experiments/task1/experiments_round3.py). The test prediction is the mean of the 10 fold models.

The whole 10-fold run is repeated for seeds 342-351 (new fold split and CatBoost seeds each
time) unless --seed or --seeds is given. The test probability is the mean of the 10 runs and
the threshold is chosen on their averaged OOF. Averaging lifted OOF F2 of the cheater class
from 0.8388 to 0.8397; a single seed varies by about 0.0002, so the gain is above seed noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

TASK1_FOLDS = '10'
TASK1_SEEDS = [str(s) for s in range(342, 352)]


def _given(args, flag):
    return any(a == flag or a.startswith(flag + '=') for a in args)


if __name__ == '__main__':
    args = sys.argv[1:]
    if not _given(args, '--folds'):
        args = ['--folds', TASK1_FOLDS, *args]
    if not (_given(args, '--seed') or _given(args, '--seeds')):
        args = ['--seeds', *TASK1_SEEDS, *args]
    main(['train', '--tasks', '1', *args])
