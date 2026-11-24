"""Task 2 entry point: player segmentation (4 classes, macro F1).

Run from the repository root:  python Task2/train.py --folds 3 --threads 24
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.

The 3-fold run (CatBoost with axis features, blended with naive Bayes) is repeated for seeds
342-344 unless --seed or --seeds is given, and the test probabilities of the 3 runs are averaged.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

TASK2_SEEDS = ['342', '343', '344']


def _given(args, flag):
    return any(a == flag or a.startswith(flag + '=') for a in args)


if __name__ == '__main__':
    args = sys.argv[1:]
    if not (_given(args, '--seed') or _given(args, '--seeds')):
        args = ['--seeds', *TASK2_SEEDS, *args]
    main(['train', '--tasks', '2', *args])
