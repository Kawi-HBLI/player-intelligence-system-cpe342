from pathlib import Path
import tempfile
import unittest
import zipfile

from pis.notebook_setup import prepare_dataset, create_run_directory, archive_results
from pis.common import TASKS


class NotebookSetupTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def dataset_archive(self, path):
        with zipfile.ZipFile(path, 'w') as bundle:
            bundle.writestr('download/data/sample_submission.csv', 'id,task1,task2,task3,task4,task5\nANS00001,0,0,0,pube,0\n')
            for task in (1,2,3,5):
                for split in (('test',) if task==5 else ('train','test')):
                    bundle.writestr(f'download/data/{TASKS[task][0]}/{split}.csv','player_id,x\nP1,1\n')

    def test_nested_zip_is_extracted_and_reused_without_changing_source(self):
        with self.temporary() as tmp:
            root=Path(tmp); archive=root/'data.zip'
            self.dataset_archive(archive)
            before=archive.read_bytes()
            data=prepare_dataset(archive,root/'work')
            self.assertTrue((data/'sample_submission.csv').exists())
            self.assertEqual(data,prepare_dataset(archive,root/'work'))
            self.assertEqual(before,archive.read_bytes())
            self.assertEqual([],list((root/'work').glob('.incoming-*')))
            self.assertEqual(data,prepare_dataset(data,root/'unused'))

    def test_archive_escape_is_rejected_before_extraction(self):
        with self.temporary() as tmp:
            root=Path(tmp); archive=root/'data.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('valid.txt','first')
                bundle.writestr('..\\escape.txt','unsafe')
            with self.assertRaises(ValueError):prepare_dataset(archive,root/'work')
            self.assertFalse((root/'escape.txt').exists())
            self.assertEqual([],list((root/'work').rglob('valid.txt')))

    def test_run_directories_are_distinct_and_named_runs_can_be_reopened(self):
        with self.temporary() as tmp:
            root=Path(tmp)
            self.assertNotEqual(create_run_directory(root),create_run_directory(root))
            self.assertEqual(create_run_directory(root,'task1-team'),create_run_directory(root,'task1-team'))
            for name in ('../escape','..\\escape','/absolute','.'):
                with self.assertRaises(ValueError):create_run_directory(root,name)

    def test_download_archive_excludes_weights_raw_files_and_oof_targets(self):
        with self.temporary() as tmp:
            root=Path(tmp)
            for name in ('task1_predictions.csv','task1_metrics.json','task1_fold1.cbm',
                         'task1_oof.csv','train.csv','submission.csv'):
                (root/name).write_text('test',encoding='utf-8')
            path=archive_results(root)
            with zipfile.ZipFile(path) as bundle:
                self.assertEqual(set(bundle.namelist()),{'task1_predictions.csv','task1_metrics.json','submission.csv'})

    def test_environment_only_run_does_not_create_a_result_archive(self):
        with self.temporary() as tmp:
            root=Path(tmp)
            (root/'run_environment.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(FileNotFoundError):archive_results(root)
            self.assertFalse((root/'results.zip').exists())


if __name__=='__main__':unittest.main()
