"""Task 3 entry point: 30-day spending prediction (regression, 1/(1+NMAE)).

Run from the repository root:  python Task3/train.py --folds 3
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

if __name__ == '__main__':
    main(['train', '--tasks', '3', *sys.argv[1:]])
