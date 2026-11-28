"""Task 3 entry point: 30-day spending prediction (regression, 1 / (1 + NMAE)) with the final model V9a.

Run from the repository root:  python Task3/train.py --threads 12

The V9a code and its experiment history live in experiments/task3/ (see LOG.md). This script runs it:
  1. experiments/task3/model/v9_impute/v9_impute.py: 3-fold CV of V8, V9a and V9b, about 17 min. It writes the
     CV outputs that step 2 checks against to outputs/experiments/task3_v9a/.
  2. experiments/task3/model/refit/refit.py --variant v9a: V9a trained on the full train, 3 seeds, about 6 min.
     It writes outputs/experiments/task3_v9a_refit/task3_predictions.csv only if all of its checks pass.
  3. Copy that file to outputs/main/task3_predictions.csv, where `python run.py assemble` reads it.
Steps 1 and 2 also refresh their results files and their rows in experiments/task3/LOG.md.

--threads 12 (the default) reproduces the submitted predictions: CatBoost builds slightly different trees
with another thread count, and step 1 stops if V8 does not reproduce v8.json.
--skip-cv skips step 1 when its outputs already exist.
--baseline runs the old CatBoost MAE baseline through run.py instead (comparison only, not used for the
submission); the other arguments then go to run.py, e.g. python Task3/train.py --baseline --folds 3
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import main as run_main  # noqa: E402

MODEL = ROOT / 'experiments' / 'task3' / 'model'
CV_SCRIPT = MODEL / 'v9_impute' / 'v9_impute.py'
REFIT_SCRIPT = MODEL / 'refit' / 'refit.py'
CV_OUTPUT = ROOT / 'outputs' / 'experiments' / 'task3_v9a'
REFIT_OUTPUT = ROOT / 'outputs' / 'experiments' / 'task3_v9a_refit' / 'task3_predictions.csv'
SUBMISSION_INPUT = ROOT / 'outputs' / 'main' / 'task3_predictions.csv'


def v9a_commands(threads, skip_cv):
    """The V9a steps as subprocess commands, run with the current interpreter."""
    refit = [sys.executable, str(REFIT_SCRIPT), '--variant', 'v9a', '--threads', str(threads)]
    return [refit] if skip_cv else [[sys.executable, str(CV_SCRIPT), '--threads', str(threads)], refit]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--baseline' in argv:
        argv.remove('--baseline')
        run_main(['train', '--tasks', '3', *argv])
        return

    parser = argparse.ArgumentParser(description='Task 3 final model V9a: CV, full-train refit, copy for assemble.')
    parser.add_argument('--threads', type=int, default=12, help='CatBoost thread_count (default 12, as submitted)')
    parser.add_argument('--skip-cv', action='store_true', help='reuse the CV outputs of an earlier run')
    parser.add_argument('--baseline', action='store_true', help='run the old CatBoost MAE baseline via run.py')
    args = parser.parse_args(argv)
    if args.skip_cv and not (CV_OUTPUT / 'task3_predictions.csv').is_file():
        parser.error(f'--skip-cv needs the CV outputs in {CV_OUTPUT}; run once without it')

    for cmd in v9a_commands(args.threads, args.skip_cv):
        print('>', ' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)   # refit.py exits 1 when a check fails
    SUBMISSION_INPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REFIT_OUTPUT, SUBMISSION_INPUT)
    print(f'Copied {REFIT_OUTPUT} to {SUBMISSION_INPUT}')


if __name__ == '__main__':
    main()
