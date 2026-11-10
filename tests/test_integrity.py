import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from pis.tabular import _threshold, _score
from pis.anomaly import temporal_features, shift_score
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
