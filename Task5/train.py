"""Task 5 entry point: account anomaly detection (unsupervised, macro F3).

Run from the repository root:  python Task5/train.py --contamination 0.20
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

if __name__ == '__main__':
    main(['train', '--tasks', '5', *sys.argv[1:]])
