import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from pis.tabular import _threshold, _score
from pis.anomaly import (
    calibrated_shift_score,
    extract_multiview_features,
    score_anomalies,
    select_expected_macro_f3,
    shift_score,
    temporal_features,
    topological_island_score,
    train_anomaly,
)
from pis.submission import assemble, validate_frame
from pis.common import TASKS
from prepare_images import extract


class IntegrityTests(unittest.TestCase):
    def test_existing_archive_member_requires_content_match(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp)
            archive=root/'images.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('task4_image_classification/train.csv','id,class\na,pube\n')
            extract(archive,root/'data')
            metadata=root/'data'/'04_image_classification'/'train.csv'
            metadata.write_text('id,class\nb,pube\n',encoding='utf-8')
            with self.assertRaises(FileExistsError):extract(archive,root/'data')

    def test_competition_spending_score(self):
        self.assertAlmostEqual(_score(np.array([0,100,200]),np.array([0,80,220])), 1/(1+40/300))
        self.assertEqual(_score(np.array([0,100,200]),np.zeros(3)), 0.5)

    def test_threshold_handles_ties_and_optimizes_macro(self):
        y = np.array([0,0,0,1,1,1,0,1])
        p = np.array([.1,.2,.5,.5,.8,.9,.8,.2])
        threshold, score = _threshold(y,p)
        candidates = np.r_[p, np.nextafter(max(p),np.inf)]
        expected = max(fbeta_score(y,p>=t,beta=2,average='macro') for t in candidates)
        self.assertAlmostEqual(score,expected)
        self.assertAlmostEqual(score,fbeta_score(y,p>=threshold,beta=2,average='macro'))

    def test_shift_score_ranks_abrupt_period_jump_highest(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({f'login_count_{i}': rng.normal(50, 2, 200) for i in range(1, 5)})
        frame['vpn_usage'] = 0.0
        frame.loc[7, 'login_count_4'] += 40   # abrupt jump in the last period
        frame.loc[9, ['login_count_1', 'login_count_2', 'login_count_3', 'login_count_4']] = np.nan
        score = shift_score(frame)
        self.assertEqual(int(np.argmax(score)), 7)
        self.assertTrue(np.isfinite(score).all())

    def test_time_change_uses_previous_periods(self):
        features = temporal_features(pd.DataFrame({'x_1':[1.], 'x_2':[2.], 'x_3':[3.], 'x_4':[10.]}))
        self.assertEqual(features.x_last_delta.iloc[0],8.)

    def test_task4_fixed_csv_joins_on_answer_id_not_row_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root,out = Path(tmp)/'data',Path(tmp)/'out'
            root.mkdir();out.mkdir()
            sample = pd.DataFrame({'id':['ANS00001','ANS00002'], **{f'task{i}':[0,0] for i in range(1,6)}})
            sample.to_csv(root/'sample_submission.csv',index=False)
            for i,(folder,_) in TASKS.items():
                path=root/folder;path.mkdir()
                if i==4:
                    # Rows deliberately listed in reverse ANS order.
                    pd.DataFrame({'id':['ANS00002','ANS00001'],'file_name':['imgb.jpg','imga.jpg'],'label':[None,None]}).to_csv(path/'test.csv',index=False)
                    pd.DataFrame({'id':['PLY00001'],'file_name':['x.jpg'],'label':[3]}).to_csv(path/'train.csv',index=False)
                    pd.DataFrame({'player_id':['imga','imgb'],'prediction':['roblock','7-11k']}).to_csv(out/'task4_predictions.csv',index=False)
                else:
                    pd.DataFrame({'player_id':[f'T{i}a',f'T{i}b']}).to_csv(path/'test.csv',index=False)
                    pd.DataFrame({'player_id':[f'T{i}a',f'T{i}b'],'prediction':[1,0]}).to_csv(out/f'task{i}_predictions.csv',index=False)
            assemble(root,out)
            frame=pd.read_csv(out/'submission.csv')
            self.assertEqual(frame.task4.tolist(),[4,0])  # ANS00001 -> imga -> roblock, ANS00002 -> imgb -> 7-11k

    def test_submission_aligns_within_task_and_rejects_missing_task(self):
        # Keep temporary files inside the verified workspace test directory.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root,out = Path(tmp)/'data',Path(tmp)/'out'
            root.mkdir();out.mkdir()
            sample = pd.DataFrame({'id':['ANS00001','ANS00002'], **{f'task{i}':[0,0] for i in range(1,6)}})
            sample.to_csv(root/'sample_submission.csv',index=False)
            for i,(folder,_) in TASKS.items():
                path=root/folder;path.mkdir()
                col='id' if i==4 else 'player_id'
                pd.DataFrame({col:[f'T{i}b',f'T{i}a']}).to_csv(path/'test.csv',index=False)
                if i==4:
                    pd.DataFrame({'id':['x','y'],'class':['pube','fiveN']}).to_csv(path/'train.csv',index=False)
                values=['pube','fiveN'] if i==4 else [1,0]
                pd.DataFrame({'player_id':[f'T{i}a',f'T{i}b'],'prediction':values}).to_csv(out/f'task{i}_predictions.csv',index=False)
            assemble(root,out)
            frame=pd.read_csv(out/'submission.csv')
            self.assertEqual(frame.task1.tolist(),[0,1])
            self.assertEqual(frame.task4.tolist(),[1,3])
            frame.loc[0,'task3']=-1
            with self.assertRaises(ValueError):validate_frame(frame,root)
            (out/'task4_predictions.csv').unlink()
            with self.assertRaises(FileNotFoundError):assemble(root,out)

    def test_topological_island_and_shift_views_isolate_planted_modes(self):
        rng = np.random.default_rng(7)
        n = 600
        frame = pd.DataFrame({
            'player_id': [f'P{i:04d}' for i in range(n)],
            'metric_a_1': rng.normal(10.0, 0.5, n),
            'metric_a_2': rng.normal(10.0, 0.5, n),
            'metric_a_3': rng.normal(10.0, 0.5, n),
            'metric_a_4': rng.normal(10.0, 0.5, n),
            'metric_b_1': rng.uniform(0.7, 0.95, n),
            'metric_b_2': rng.uniform(0.7, 0.95, n),
            'metric_b_3': rng.uniform(0.7, 0.95, n),
            'metric_b_4': rng.uniform(0.7, 0.95, n),
            'login_lat_1': rng.uniform(-50, 50, n),
            'login_lon_1': rng.uniform(-120, 120, n),
            'login_lat_2': np.zeros(n),
            'login_lon_2': np.zeros(n),
            'login_lat_3': np.zeros(n),
            'login_lon_3': np.zeros(n),
            'login_lat_4': np.zeros(n),
            'login_lon_4': np.zeros(n),
            'discrete_flag': rng.integers(0, 3, n).astype(float),
        })
        for p in (2, 3, 4):
            frame[f'login_lat_{p}'] = frame['login_lat_1'] + rng.normal(0, 0.01, n)
            frame[f'login_lon_{p}'] = frame['login_lon_1'] + rng.normal(0, 0.01, n)
        # Plant a clean secondary island on metric_b (indices 0..29) and a pure temporal shift + geo jump (30..54)
        for p in range(1, 5):
            frame.loc[:29, f'metric_b_{p}'] = rng.uniform(0.20, 0.35, 30)
        frame.loc[30:54, ['metric_a_1', 'metric_a_2', 'metric_a_3']] -= 2.0
        frame.loc[30:54, 'metric_a_4'] += 6.0
        frame.loc[30:54, 'login_lat_4'] += 25.0
        frame.loc[100, 'metric_a_1'] = np.nan

        x = frame.drop(columns='player_id')
        levels, shift_mag = extract_multiview_features(x)
        self.assertNotIn('login_lat', levels.columns)
        self.assertNotIn('login_lon', levels.columns)
        self.assertIn('geo_hav', shift_mag.columns)

        isl = topological_island_score(levels)
        self.assertTrue((isl[:30] > 0.0).all())
        self.assertEqual(float(isl[30:].sum()), 0.0)

        _, p_shift, _ = calibrated_shift_score(shift_mag, seed=7)
        self.assertTrue(p_shift[30:55].mean() > 0.85)
        self.assertTrue(p_shift[60:].mean() < 0.05)

    def test_select_expected_macro_f3_handles_null_and_high_posterior(self):
        pred_zero, k_zero, _, _ = select_expected_macro_f3(np.zeros(500))
        self.assertEqual(k_zero, 0)
        self.assertEqual(int(pred_zero.sum()), 0)

        probs = np.concatenate([np.full(40, 0.95), np.full(10, 0.30), np.zeros(450)])
        pred, k_opt, est_prev, _ = select_expected_macro_f3(probs)
        self.assertAlmostEqual(est_prev, probs.sum() / 500.0, places=5)
        self.assertGreaterEqual(k_opt, 40)
        self.assertTrue((pred[:40] == 1).all())

    def test_train_anomaly_outputs_and_unsupervised_guards(self):
        rng = np.random.default_rng(19)
        n = 300
        frame = pd.DataFrame({
            'player_id': [f'T5_{i:04d}' for i in range(n)],
            'feat_a_1': rng.normal(0, 1, n),
            'feat_a_2': rng.normal(0, 1, n),
            'feat_a_3': rng.normal(0, 1, n),
            'feat_a_4': rng.normal(0, 1, n),
            'feat_b_1': rng.normal(0, 1, n),
            'feat_b_2': rng.normal(0, 1, n),
            'feat_b_3': rng.normal(0, 1, n),
            'feat_b_4': rng.normal(0, 1, n),
            'feat_c_1': rng.normal(0, 1, n),
            'feat_c_2': rng.normal(0, 1, n),
            'feat_c_3': rng.normal(0, 1, n),
            'feat_c_4': rng.normal(0, 1, n),
        })
        frame.loc[:19, 'feat_a_4'] += 8.0
        frame.loc[:19, 'feat_b_4'] -= 8.0
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root, out = Path(tmp) / 'data', Path(tmp) / 'out'
            (root / TASKS[5][0]).mkdir(parents=True)
            frame.to_csv(root / TASKS[5][0] / 'test.csv', index=False)
            metrics = train_anomaly(root, out, seed=19, profile='quick', contamination='auto', threads=1)
            self.assertEqual(metrics['contamination_mode'], 'auto_expected_macro_f3')
            self.assertIsNone(metrics['macro_f3'])
            preds = pd.read_csv(out / 'task5_predictions.csv')
            scores = pd.read_csv(out / 'task5_scores.csv')
            self.assertEqual(list(preds.columns), ['player_id', 'prediction'])
            self.assertIn('anomaly_probability', scores.columns)
            self.assertGreaterEqual(int(preds.loc[:19, 'prediction'].sum()), 18)
            self.assertEqual(int(preds.loc[20:, 'prediction'].sum()), 0)

            metrics_fixed = train_anomaly(root, out, seed=19, profile='quick', contamination=0.10, threads=1)
            self.assertEqual(metrics_fixed['flagged_rows'], 30)
            self.assertEqual(metrics_fixed['contamination_mode'], 'fixed_fraction')

            frame.assign(is_anomaly=0).to_csv(root / TASKS[5][0] / 'test.csv', index=False)
            with self.assertRaises(ValueError):
                train_anomaly(root, out, seed=19, profile='quick')


if __name__=='__main__':unittest.main()
