"""Tests for tools/swap_submission.py on small synthetic files (no Kaggle data needed).

The fake data root holds only what the tool reads: sample_submission.csv (for
read_template / validate_file) and 03_regression/test.csv (for --pred task3).
Patched with unittest.mock: validate_file (to force a validator failure) and
swap.DEFAULT_DATA (pointed at a missing folder for the no-data-folder tests,
because the real default folder may exist on a teammate's machine).
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import swap_submission as swap  # noqa: E402

HEADER = 'id,task1,task2,task3,task4,task5'
ROWS = [
    'ANS00001,1,3,0.0018028174448782,0,1',
    'ANS00002,0,0,2515.6856080098723,3,0',
    'ANS00003,1,2,0.0,4,0',
    'ANS00004,0,1,12.5,1,1',
    'ANS00005,0,3,301.25,2,0',
    'ANS00006,1,0,7.0,0,0',
]
PLAYERS = ['P006', 'P001', 'P004', 'P003', 'P005', 'P002']  # ANS row i = test.csv row i


def csv_text(rows, header=HEADER, newline='\n'):
    return newline.join([header] + rows) + newline


def set_field(rows, column, values):
    index = HEADER.split(',').index(column)
    out = []
    for row, value in zip(rows, values):
        fields = row.split(',')
        fields[index] = value
        out.append(','.join(fields))
    return out


class SwapSubmissionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / 'data'
        (self.data / '03_regression').mkdir(parents=True)
        sample = [f'{row.split(",")[0]},0,0,0,0,0' for row in ROWS]
        (self.data / 'sample_submission.csv').write_text(csv_text(sample), encoding='utf-8')
        (self.data / '03_regression' / 'test.csv').write_text(
            'player_id,feature\n' + ''.join(f'{p},1\n' for p in PLAYERS), encoding='utf-8')
        self.base = self.write('base.csv', csv_text(ROWS))
        self.out = self.root / 'out.csv'

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text, encoding='utf-8'):
        path = self.root / name
        path.write_bytes(text.encode(encoding))
        return path

    def run_tool(self, *args, base=None, data_root=True):
        stdout = io.StringIO()
        root_args = ['--data-root', str(self.data)] if data_root else []
        with contextlib.redirect_stdout(stdout):
            swap.main(['--base', str(base or self.base), '--out', str(self.out),
                       *root_args, *map(str, args)])
        return stdout.getvalue()

    def assert_fails(self, *args, message=None, base=None, data_root=True, keep_output=False):
        before = self.out.read_bytes() if self.out.exists() else None
        with self.assertRaises(swap.SwapError) as caught:
            self.run_tool(*args, base=base, data_root=data_root)
        if keep_output:
            self.assertEqual(self.out.read_bytes(), before, 'a failed run must not change a good output')
        else:
            self.assertFalse(self.out.exists(), 'a failed swap must not leave an output file')
        self.assertEqual(list(self.root.glob('*.tmp.csv')), [], 'temporary file left behind')
        if message:
            self.assertIn(message, str(caught.exception))
        return caught.exception

    # --- Mode 1 -----------------------------------------------------------------

    def test_mode1_swaps_only_requested_column_bom_crlf(self):
        base = self.write('base_bom.csv', csv_text(ROWS, newline='\r\n'), encoding='utf-8-sig')
        new_task4 = ['4', '0', '1', '2', '3', '4']
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task4', new_task4)))
        self.run_tool('--from', other, '--tasks', 'task4', base=base)
        expected = csv_text(set_field(ROWS, 'task4', new_task4), newline='\r\n').encode('utf-8-sig')
        self.assertEqual(self.out.read_bytes(), expected)  # BOM, CRLF and all other fields kept

    def test_mode1_matches_rows_by_id_not_position(self):
        new_task2 = ['0', '1', '2', '3', '0', '1']
        other_rows = set_field(ROWS, 'task2', new_task2)[::-1]  # reversed row order
        other = self.write('other.csv', csv_text(other_rows))
        self.run_tool('--from', other, '--tasks', 'task2')
        self.assertEqual(self.out.read_text(), csv_text(set_field(ROWS, 'task2', new_task2)))

    def test_mode1_notes_other_differences(self):
        rows = set_field(ROWS, 'task4', ['1'] * 6)
        rows = set_field(rows, 'task1', ['0', '1', '1', '0', '0', '1'])  # base 1,0,1,0,0,1: rows 1-2 differ
        rows = set_field(rows, 'task3', ['0.0018028174448782388', '2515.6856080098723',
                                         '0.0', '12.5', '999.0', '7.0'])  # rounding only + 1 real
        other = self.write('other.csv', csv_text(rows))
        output = self.run_tool('--from', other, '--tasks', 'task4')
        self.assertIn('Note: --from also differs from the base in columns you are NOT swapping', output)
        self.assertIn('task1: 2 rows differ', output)
        self.assertIn('task3: 1 rows differ', output)
        self.assertNotIn('task2:', output)
        self.assertIn('only task4 is copied', output)
        self.assertEqual(self.out.read_text(), csv_text(set_field(ROWS, 'task4', ['1'] * 6)))

    def test_mode1_says_when_other_columns_match(self):
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task4', ['1'] * 6)))
        output = self.run_tool('--from', other, '--tasks', 'task4')
        self.assertIn('--from matches the base in all other columns', output)

    # --- Mode 2 -----------------------------------------------------------------

    def test_mode2_task3_round_trip(self):
        task3 = [row.split(',')[3] for row in ROWS]
        pred = self.write('task3_predictions.csv', 'player_id,prediction\n' + ''.join(
            f'{p},{v}\n' for p, v in sorted(zip(PLAYERS, task3))))  # sorted: not test.csv order
        output = self.run_tool('--pred', f'task3={pred}')
        self.assertEqual(self.out.read_bytes(), self.base.read_bytes())
        self.assertIn('[task3] rows changed: 0', output)

    # --- Errors: SwapError and no output file -------------------------------------

    def test_error_task_given_twice(self):
        other = self.write('other.csv', csv_text(ROWS))
        self.assert_fails('--from', other, '--tasks', 'task3', '--pred', f'task3={other}',
                          message='more than once')

    def test_error_out_equal_to_base(self):
        before = self.base.read_bytes()
        with self.assertRaises(swap.SwapError), contextlib.redirect_stdout(io.StringIO()):
            swap.main(['--base', str(self.base), '--out', str(self.base), '--data-root', str(self.data),
                       '--from', str(self.base), '--tasks', 'task1'])
        self.assertEqual(self.base.read_bytes(), before)

    def test_error_task4_text_vs_numeric(self):
        text = ['pube', 'roblock', 'fiveN', '7-11k', 'freefried', 'pube']
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task4', text)))
        self.assert_fails('--from', other, '--tasks', 'task4', message='label format mismatch')

    def test_error_task1_label_outside_0_1(self):
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task1', ['0', '1', '2', '0', '1', '0'])))
        self.assert_fails('--from', other, '--tasks', 'task1', message='task1: labels must be in [0, 1]')

    def test_error_ids_do_not_match_base(self):
        rows = list(ROWS)
        rows[0] = rows[0].replace('ANS00001', 'ANS99999')
        other = self.write('other.csv', csv_text(rows))
        self.assert_fails('--from', other, '--tasks', 'task2', message='ids differ from the base file')

    def test_invalid_output_is_deleted(self):
        # validate_file is patched: the synthetic files are valid, so this is the
        # simplest way to make the final validator reject the written output.
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task4', ['1'] * 6)))
        with mock.patch.object(swap, 'validate_file', side_effect=ValueError('forced failure')):
            self.assert_fails('--from', other, '--tasks', 'task4', message='forced failure')

    # --- Review fixes (PR #14) --------------------------------------------------------

    def test_mode1_class_labels_normalized_to_whole_numbers(self):  # item 1
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task2', ['3.0', '0.0', '2.0', '1.0', '3', '0'])))
        output = self.run_tool('--from', other, '--tasks', 'task2')
        self.assertEqual(self.out.read_text(), csv_text(ROWS))  # "3.0" -> "3": same text as the base
        self.assertIn('[task2] rows changed: 0', output)

    def test_failed_rerun_keeps_existing_good_output(self):  # item 2
        good = self.write('good.csv', csv_text(set_field(ROWS, 'task4', ['1'] * 6)))
        self.run_tool('--from', good, '--tasks', 'task4')
        bad = self.write('bad.csv', csv_text(set_field(ROWS, 'task1', ['0', '1', '2', '0', '1', '0'])))
        self.assert_fails('--from', bad, '--tasks', 'task1', keep_output=True)  # fails before writing
        with mock.patch.object(swap, 'validate_file', side_effect=ValueError('forced failure')):
            self.assert_fails('--from', good, '--tasks', 'task4', keep_output=True,
                              message='forced failure')  # fails after the temp file is written

    def test_mode1_without_data_folder_uses_base_ids(self):  # item 3
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task4', ['1'] * 6)))
        with mock.patch.object(swap, 'DEFAULT_DATA', self.root / 'missing'):
            output = self.run_tool('--from', other, '--tasks', 'task4', data_root=False)
            self.assertIn('Repo validator skipped (no data folder found); tool checks passed', output)
            self.assertEqual(self.out.read_text(), csv_text(set_field(ROWS, 'task4', ['1'] * 6)))
            # Tool checks still run without the validator.
            self.out.unlink()
            bad = self.write('bad.csv', csv_text(set_field(ROWS, 'task5', ['0', '1', '5', '0', '1', '0'])))
            self.assert_fails('--from', bad, '--tasks', 'task5', data_root=False, message='task5: labels')
            # Mode 2 still needs the data folder.
            pred = self.write('p.csv', 'player_id,prediction\nP001,1.0\n')
            self.assert_fails('--pred', f'task3={pred}', data_root=False, message='needs the Kaggle data folder')
        # An explicit --data-root that does not exist is still an error.
        self.assert_fails('--from', other, '--tasks', 'task4', '--data-root', self.root / 'typo',
                          data_root=False, message='sample_submission.csv not found')

    def test_mode2_task4_image_root_with_only_test_csv(self):  # item 4
        images = self.root / 'Task4'
        images.mkdir()
        ans = [row.split(',')[0] for row in ROWS]
        order = [2, 0, 5, 1, 4, 3]  # test.csv rows in a different order from the ANS ids
        (images / 'test.csv').write_text('id,file_name\n' + ''.join(
            f'{ans[i]},img{i}.jpg\n' for i in order), encoding='utf-8')
        labels = ['pube', 'roblock', 'fiveN', '7-11k', 'freefried', 'pube']  # for ANS rows 1..6
        pred = self.write('task4_predictions.csv', 'player_id,prediction\n' + ''.join(
            f'img{i},{labels[i]}\n' for i in range(6)))
        self.assertFalse((images / 'train.csv').exists())
        self.run_tool('--pred', f'task4={pred}', '--image-root', images)
        self.assertEqual(self.out.read_text(), csv_text(set_field(ROWS, 'task4', ['3', '4', '1', '0', '2', '3'])))

    def test_blank_value_reports_missing(self):  # item 5
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task2', ['0', '', '2', '1', '3', '0'])))
        self.assert_fails('--from', other, '--tasks', 'task2', message='task2: 1 missing values')
        other4 = self.write('other4.csv', csv_text(set_field(ROWS, 'task4', ['0', '', '2', '1', '3', '0'])))
        error = self.assert_fails('--from', other4, '--tasks', 'task4', message='task4: 1 missing values')
        self.assertNotIn('text', str(error))

    def test_inf_in_class_column_is_swap_error(self):  # item 5
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task1', ['0', 'inf', '1', '-inf', '1', '0'])))
        self.assert_fails('--from', other, '--tasks', 'task1', message='task1: 2 non-finite values')

    def test_numeric_task4_into_text_base(self):  # item 5
        text = ['pube', 'roblock', 'fiveN', '7-11k', 'freefried', 'pube']
        base = self.write('text_base.csv', csv_text(set_field(ROWS, 'task4', text)))
        numeric = set_field(ROWS, 'task4', ['3', '4', '1', '0', '2', '3'])
        other = self.write('other.csv', csv_text(numeric))
        output = self.run_tool('--from', other, '--tasks', 'task4', base=base)
        self.assertIn('the base task4 column was text', output)
        self.assertEqual(self.out.read_text(), csv_text(numeric))

    def test_share_warning_after_class_columns_only(self):  # item 6
        other = self.write('other.csv', csv_text(set_field(ROWS, 'task1', ['0'] * 6)))
        task3 = [row.split(',')[3] for row in ROWS]
        pred = self.write('t3.csv', 'player_id,prediction\n' + ''.join(f'{p},{v}\n' for p, v in zip(PLAYERS, task3)))
        output = self.run_tool('--from', other, '--tasks', 'task1', '--pred', f'task3={pred}')
        warning = ('For sanity checks only. Never use these test-prediction shares to adjust a model '
                   '(class weights, thresholds, priors): the TA counts that as pseudo-labeling.')
        self.assertEqual(output.count(warning), 1)
        task1_block = output.split('[task1]')[1].split('[task3]')[0]
        self.assertIn(warning, task1_block)
        self.assertNotIn(warning, output.split('[task3]')[1])


if __name__ == '__main__':
    unittest.main()
