import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from pis.tabular import _threshold, _score
from pis.anomaly import temporal_features, shift_score
from pis.task2_naive_bayes import SmoothNaiveBayes, geometric_blend
from pis.task1_features import add_range_rules, add_imputed, fit_imputers, IMPUTED
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

    def test_task1_range_rules_count_certain_cases_and_ignore_missing(self):
        frame = pd.DataFrame({'crosshair_placement':[0.81,0.95,np.nan],'reports_received':[4,25,np.nan],
                              'level':[26.,99.,np.nan],'account_age_days':[100.,1200.,np.nan],
                              'friend_network_size':[50.,300.,np.nan],'level_progression_speed':[1.,7.,np.nan]})
        out = add_range_rules(frame)
        self.assertEqual(out.n_sure_cheat_rules.tolist(),[3,0,0])   # below every legit floor
        self.assertEqual(out.n_sure_legit_rules.tolist(),[0,6,0])   # above every cheater ceiling
        self.assertTrue(np.isnan(out.loc[2,'level_above_legit_floor']))

    def test_task1_imputed_columns_keep_observed_values(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({c: rng.normal(50, 5, 300) for c in IMPUTED})
        frame['other'] = rng.normal(size=300)
        frame.loc[:20, 'level'] = np.nan
        out = add_imputed(frame, fit_imputers(frame))
        self.assertTrue(out['level_imp'].notna().all())
        self.assertTrue((out.loc[21:, 'level_imp'] == frame.loc[21:, 'level']).all())

    def test_smooth_naive_bayes_separates_classes_and_ignores_missing(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 500)
        frame = pd.DataFrame({'spend': np.r_[rng.normal(0, 1, 500), rng.normal(4, 1, 500)],
                              'tier': np.r_[rng.choice(['a', 'b'], 500, p=[.9, .1]), rng.choice(['a', 'b'], 500, p=[.1, .9])]})
        model = SmoothNaiveBayes().fit(frame, y)
        self.assertGreater((model.predict_proba(frame).argmax(1) == y).mean(), 0.95)
        empty = pd.DataFrame({'spend': [np.nan], 'tier': [None]})
        np.testing.assert_allclose(model.predict_proba(empty)[0], [0.5, 0.5], atol=1e-9)   # only the prior is left
        unseen = pd.DataFrame({'spend': [np.nan], 'tier': ['zzz']})
        np.testing.assert_allclose(model.predict_proba(unseen)[0], [0.5, 0.5], atol=1e-9)
        blend = geometric_blend(np.array([[0.8, 0.2]]), np.array([[0.2, 0.8]]), 0.5)
        np.testing.assert_allclose(blend, [[0.5, 0.5]], atol=1e-6)

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


if __name__=='__main__':unittest.main()
