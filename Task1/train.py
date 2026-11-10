"""Task 1 entry point: anti-cheat detection (binary classification, macro F2).

Run from the repository root:  python Task1/train.py --folds 3
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

if __name__ == '__main__':
    main(['train', '--tasks', '1', *sys.argv[1:]])
