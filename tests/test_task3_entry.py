"""Task3/train.py runs the final V9a pipeline (CV, refit, copy to outputs/main); --baseline keeps the old run."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_entry():
    spec = importlib.util.spec_from_file_location('task3_train', ROOT / 'Task3' / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task3EntryTest(unittest.TestCase):
    def setUp(self):
        self.entry = load_entry()

    def test_default_runs_cv_then_refit_with_12_threads(self):
        cmds = self.entry.v9a_commands(threads=12, skip_cv=False)
        self.assertEqual([c[1] for c in cmds], [str(self.entry.CV_SCRIPT), str(self.entry.REFIT_SCRIPT)])
        self.assertTrue(all(c[0] == sys.executable for c in cmds))
        self.assertEqual(cmds[0][2:], ['--threads', '12'])
        self.assertEqual(cmds[1][2:], ['--variant', 'v9a', '--threads', '12'])
        self.assertTrue(self.entry.CV_SCRIPT.is_file() and self.entry.REFIT_SCRIPT.is_file())

    def test_skip_cv_runs_only_the_refit(self):
        cmds = self.entry.v9a_commands(threads=4, skip_cv=True)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0][1:], [str(self.entry.REFIT_SCRIPT), '--variant', 'v9a', '--threads', '4'])

    def test_baseline_forwards_the_other_arguments_to_run_py(self):
        with mock.patch.object(self.entry, 'run_main') as run_main, mock.patch.object(self.entry.subprocess, 'run') as sub:
            self.entry.main(['--baseline', '--folds', '3', '--threads', '24'])
        run_main.assert_called_once_with(['train', '--tasks', '3', '--folds', '3', '--threads', '24'])
        sub.assert_not_called()

    def test_skip_cv_without_cv_outputs_stops_before_running(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.entry.subprocess, 'run') as sub:
            self.entry.CV_OUTPUT = Path(tmp) / 'missing'
            with self.assertRaises(SystemExit):
                self.entry.main(['--skip-cv'])
        sub.assert_not_called()

    def test_unknown_argument_is_an_error(self):
        with mock.patch.object(self.entry.subprocess, 'run') as sub, self.assertRaises(SystemExit):
            self.entry.main(['--folds', '3'])
        sub.assert_not_called()

    def test_refit_predictions_are_copied_for_assemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.entry.REFIT_OUTPUT = tmp / 'refit' / 'task3_predictions.csv'
            self.entry.SUBMISSION_INPUT = tmp / 'main' / 'task3_predictions.csv'

            def fake_run(cmd, cwd, check):
                if '--variant' in cmd:
                    self.entry.REFIT_OUTPUT.parent.mkdir(parents=True)
                    self.entry.REFIT_OUTPUT.write_text('player_id,prediction\na,1.5\n')

            with mock.patch.object(self.entry.subprocess, 'run', side_effect=fake_run) as sub:
                self.entry.main([])
            self.assertEqual(sub.call_count, 2)
            self.assertEqual(self.entry.SUBMISSION_INPUT.read_text(), 'player_id,prediction\na,1.5\n')


if __name__ == '__main__':
    unittest.main()
