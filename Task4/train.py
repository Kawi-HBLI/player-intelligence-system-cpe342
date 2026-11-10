"""Task 4 entry point: game title detection from screenshots (5 classes, macro F1).

Run from the repository root:  python Task4/train.py --epochs 5 --batch-size 128
Extra arguments are passed on to run.py (see `python run.py --help`).
The shared implementation lives in the `pis` package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main  # noqa: E402

if __name__ == '__main__':
    main(['train', '--tasks', '4', *sys.argv[1:]])
