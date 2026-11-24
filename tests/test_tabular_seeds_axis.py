import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pis import tabular
from pis.tabular import _threshold, train_task_seeds
from pis.task2_axis import NOISE_COLUMNS, add_axis_features


def _segments(n_per_class=200, seed=0):
    """Four segments: engagement high for 1 and 3, spending high for 2 and 3."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2, 3], n_per_class)
    engaged, spending = np.isin(y, [1, 3]).astype(float), np.isin(y, [2, 3]).astype(float)
    frame = pd.DataFrame({
        'play_frequency': rng.normal(engaged * 2.0, 1.0),
        'days_since_last_login': rng.normal(-engaged * 2.0, 1.0),   # higher means less engaged
        'total_spending_thb': rng.normal(spending * 3.0, 1.0),
        'random_metric_1': rng.normal(size=len(y)),
        'tier': rng.choice(['a', 'b'], len(y)),
    })
    frame.loc[::7, 'play_frequency'] = np.nan
    return frame, y


class AxisFeatureTests(unittest.TestCase):
    def test_axes_are_fitted_on_train_and_signed(self):
        x, y = _segments()
        xt, _ = _segments(seed=1)
        out, out_test = add_axis_features(x, xt, y, ['tier'])
        self.assertEqual(list(out.columns[:len(x.columns)]), list(x.columns))   # originals kept, in order
        extra = [c for c in out.columns if c not in x.columns]
        self.assertEqual(extra, [f'{a}_{s}' for a in ('eng_axis', 'spend_axis') for s in ('mean', 'min', 'max', 'present')])
        self.assertIn('random_metric_1', NOISE_COLUMNS)
        # The engagement axis averages play_frequency and the negated days_since_last_login.
        cols = ['play_frequency', 'days_since_last_login']
        mean, std = x[cols].mean(), x[cols].std()
        z, zt = (x[cols] - mean) / std, (xt[cols] - mean) / std
        np.testing.assert_allclose(out.eng_axis_mean, pd.concat([z.play_frequency, -z.days_since_last_login], axis=1).mean(axis=1))
        self.assertEqual(out.loc[0, 'eng_axis_present'], 1)   # play_frequency is missing on row 0
        self.assertGreater(out.eng_axis_mean[np.isin(y, [1, 3])].mean(), out.eng_axis_mean[np.isin(y, [0, 2])].mean())
        # Test rows are only transformed, with the train means and standard deviations.
        np.testing.assert_allclose(out_test.eng_axis_mean,
                                   pd.concat([zt.play_frequency, -zt.days_since_last_login], axis=1).mean(axis=1))
        train_again, _ = add_axis_features(x, xt.assign(play_frequency=xt.play_frequency + 5), y, ['tier'])
        np.testing.assert_allclose(train_again.eng_axis_mean, out.eng_axis_mean)

    def test_axes_need_all_four_segments(self):
        x, y = _segments()
        keep = y != 3
        with self.assertRaises(ValueError):
            add_axis_features(x[keep], x[keep], y[keep], ['tier'])


class SeedAverageTests(unittest.TestCase):
    def _fake_train_task(self, task, data_root, output_dir, folds, seed, profile, threads, engine):
        rng = np.random.default_rng(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        p = np.clip(y * 0.5 + rng.uniform(0, 0.5, len(y)), 0, 1)
        pd.DataFrame({'player_id': [f'p{i}' for i in range(8)], 'target': y, 'probability_1': p,
                      'prediction': (p >= 0.5).astype(int)}).to_csv(output_dir / 'task1_oof.csv', index=False)
        q = rng.uniform(size=5)
        pd.DataFrame({'player_id': [f't{i}' for i in range(5)], 'probability_0.0': 1 - q,
                      'probability_1.0': q}).to_csv(output_dir / 'task1_probabilities.csv', index=False)
        return {'engine': 'catboost', 'oof_score': 0.5}

    def test_task1_runs_are_averaged_and_threshold_rechosen(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, \
                mock.patch.object(tabular, 'train_task', side_effect=self._fake_train_task):
            out = Path(tmp)
            metrics = train_task_seeds(1, Path('unused'), out, folds=2, seeds=[1, 2, 3])
            runs = [pd.read_csv(out / 'seeds' / f'seed{s}' / 'task1_probabilities.csv') for s in (1, 2, 3)]
            oofs = [pd.read_csv(out / 'seeds' / f'seed{s}' / 'task1_oof.csv') for s in (1, 2, 3)]
            mean_test = np.mean([r['probability_1.0'] for r in runs], axis=0)
            mean_oof = np.mean([o.probability_1 for o in oofs], axis=0)
            threshold, score = _threshold(oofs[0].target.to_numpy(), mean_oof)
            self.assertAlmostEqual(metrics['threshold'], threshold)
            self.assertAlmostEqual(metrics['oof_score'], score)
            written = pd.read_csv(out / 'task1_probabilities.csv')
            np.testing.assert_allclose(written['probability_1.0'], mean_test)
            predictions = pd.read_csv(out / 'task1_predictions.csv')
            self.assertEqual(predictions.prediction.tolist(), (mean_test >= threshold).astype(int).tolist())
            self.assertEqual(json.loads((out / 'task1_metrics.json').read_text())['seeds'], [1, 2, 3])

    def test_seed_average_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            train_task_seeds(3, Path('unused'), Path('unused'), folds=3, seeds=[1, 2])
        with self.assertRaises(ValueError):
            train_task_seeds(1, Path('unused'), Path('unused'), folds=3, seeds=[1, 1])


if __name__ == '__main__':
    unittest.main()
