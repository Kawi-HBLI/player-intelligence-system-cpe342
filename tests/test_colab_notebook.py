"""Exercise the actual notebook cells locally without cloud auth, pip or model fitting."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nbformat
import pandas as pd

from pis.common import TASKS

ROOT = Path(__file__).resolve().parents[1]


def write_image_metadata(root):
    folder=root/TASKS[4][0]
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':['train-a','train-b'],'class':['pube','fiveN']}).to_csv(folder/'train.csv',index=False)
    pd.DataFrame({'id':['T4a','T4b'],'class':[None,None]}).to_csv(folder/'test.csv',index=False)


class ColabNotebookTests(unittest.TestCase):
    def exercise(self, late_prediction=False):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base=Path(temporary).resolve()
            data=base/'input'; data.mkdir()
            repo=base/'checkout'; repo.mkdir(); (repo/'.git').mkdir()
            output=base/'runs'/'group'; output.mkdir(parents=True)
            pd.DataFrame({'id':['ANS00001','ANS00002'],**{f'task{i}':[0,0] for i in range(1,6)}}).to_csv(data/'sample_submission.csv',index=False)
            for task in (1,2,3,5):
                folder=data/TASKS[task][0];folder.mkdir()
                test=pd.DataFrame({'player_id':[f'T{task}a',f'T{task}b'],'x':[0.,1.]})
                test.to_csv(folder/'test.csv',index=False)
                if task!=5:
                    test.assign(**{TASKS[task][1]:[0,1]}).to_csv(folder/'train.csv',index=False)
            def predictions(task):
                values=['pube','fiveN'] if task==4 else [0,1]
                pd.DataFrame({'player_id':[f'T{task}a',f'T{task}b'],'prediction':values}).to_csv(output/f'task{task}_predictions.csv',index=False)
                (output/f'task{task}_metrics.json').write_text(json.dumps({'task':task}),encoding='utf-8')
            for task in range(1,6):
                if not (late_prediction and task==3):predictions(task)
            old_cwd=Path.cwd(); old_path=sys.path[:]
            namespace={}
            calls=[]
            def command_output(args, **kwargs):
                if 'get-url' in args:return 'https://github.com/Kawi-HBLI/player-intelligence-system-cpe342.git\n'
                if 'rev-parse' in args:return '0'*40+'\n'
                if 'status' in args:return ''
                raise AssertionError(f'Unexpected command: {args}')
            def command_run(args, **kwargs):
                calls.append(args)
                if '-m' in args and 'pip' in args:
                    self.assertIn('requirements-colab.txt',str(args[-1]))
                elif any(str(value).endswith('prepare_images.py') for value in args):
                    target=Path(args[args.index('--data-root')+1]).resolve()
                    self.assertNotEqual(target,data)
                    self.assertTrue(target.is_relative_to(base))
                    write_image_metadata(target)
                else:raise AssertionError(f'Unexpected command: {args}')
                return subprocess.CompletedProcess(args,0)
            notebook=nbformat.read(ROOT/'Player_Intelligence_Colab.ipynb',as_version=4)
            try:
                with contextlib.redirect_stdout(io.StringIO()), patch('subprocess.check_output',side_effect=command_output), \
                     patch('subprocess.run',side_effect=command_run), \
                     patch('platform.platform',return_value='test-host'), \
                     patch('pis.tabular.train_task',side_effect=lambda task,*a,**k:predictions(task)) as fit, \
                     patch('pis.image.train_image') as image_fit, patch('pis.anomaly.train_anomaly') as anomaly_fit:
                    for cell in notebook.cells:
                        if cell.cell_type!='code':continue
                        source=cell.source
                        tag=cell.metadata['tags'][0]
                        source=source.replace("Path('/content/player-intelligence-system-cpe342')",f'Path({str(repo)!r})')
                        source=source.replace("Path('/content/pis_image_data')",f'Path({str(base/"images")!r})')
                        source=source.replace("Path('/content/pis_data')",f'Path({str(base/"work")!r})')
                        exec(compile(source,f'colab:{tag}','exec'),namespace)
                        if tag=='settings':
                            namespace.update(USE_DRIVE=False,DATA_SOURCE='localfolder',DATA_PATH=str(data),
                                             OUTPUT_ROOT=str(output.parent),RUN_NAME='group',
                                             RUN_TRAINING=late_prediction,TASKS=[3] if late_prediction else [1,2,3,4,5])
                    self.assertEqual(fit.call_count,1 if late_prediction else 0)
                    image_fit.assert_not_called();anomaly_fit.assert_not_called()
                self.assertFalse((data/TASKS[4][0]).exists(),'Task 4 preparation changed the input folder')
                self.assertTrue((output/'submission.csv').exists())
                frame=pd.read_csv(output/'submission.csv')
                self.assertEqual(frame.task4.tolist(),['pube','fiveN'])
                with zipfile.ZipFile(output/'results.zip') as bundle:
                    self.assertIn('submission.csv',bundle.namelist())
                    self.assertNotIn('task3_oof.csv',bundle.namelist())
                self.assertEqual(sum(any(str(v).endswith('prepare_images.py') for v in call) for call in calls),1)
            finally:
                os.chdir(old_cwd);sys.path[:]=old_path

    def test_inspection_assembles_peer_predictions_without_model_training(self):
        self.exercise()

    def test_late_task_completion_prepares_metadata_without_changing_input(self):
        self.exercise(late_prediction=True)


if __name__=='__main__':unittest.main()
