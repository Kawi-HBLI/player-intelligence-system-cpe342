# Player Intelligence System

This repository contains the source code, notebooks, and documentation for the CPE342 Machine Learning competition. It covers anti-cheat detection, player segmentation, monthly spending prediction, game screenshot classification, and account security monitoring.

[Open Player Intelligence System in Google Colab](https://colab.research.google.com/github/Kawi-HBLI/player-intelligence-system-cpe342/blob/main/Player_Intelligence_Colab.ipynb)

## Run as a group in Google Colab

The Colab notebook is the recommended entry point for team members. It uses the implementation from this repository and keeps competition data and run artifacts in Google Drive rather than GitHub.

1. Open the Colab link above. In Colab, choose Save a copy in Drive from the File menu so you have a notebook in your own Google Drive. Each team member should run their own copy. Sharing a notebook shares its code and saved outputs, but it does not share the running virtual machine or its memory with other people. See Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) for notebook sharing and runtime details.
2. Put the competition ZIP in `My Drive/CPE342/data.zip`, or choose the notebook's upload option. The raw competition data is not stored in this GitHub repository. If you use a folder that is already present in the Colab runtime, select the local-folder data source and set `DATA_PATH` to it.
3. In the notebook's Settings cell, review `DATA_SOURCE`, `DATA_PATH`, `TASKS`, `PROFILE`, `RUN_NAME`, `RUN_TRAINING`, and `USE_DRIVE`. The defaults use Drive for data and output, run all five tasks, use the balanced profile, and enable training. A new run gets a unique folder under `My Drive/CPE342/runs/`; set `RUN_NAME` when you want to use a specific run folder. If you set `USE_DRIVE=False`, set `OUTPUT_ROOT` to a path under `/content`; those outputs will be temporary unless you download or copy them before the runtime ends.
4. For Task 4, select a GPU in the Change runtime type dialog under Runtime if one is available. The notebook prepares the separate image dataset when needed. Colab GPU availability and runtime duration vary, so a GPU or uninterrupted run is not guaranteed. Google's [resource information](https://research.google.com/colaboratory/faq.html#resource-limits) describes these limits.
5. Run the cells from top to bottom. The notebook installs `requirements-colab.txt` from this repository and uses the PyTorch installation already available in the Colab runtime. It records the repository commit used for the run. Do not maintain a separate copy of the model implementation in the notebook.

The notebook extracts ZIP data under `/content` for training and saves predictions, metrics, and checkpoints to the selected Drive run folder. `/content` belongs to the active runtime and is temporary. Drive artifacts remain available after a runtime disconnect, but training does not automatically resume from a checkpoint. A named `RUN_NAME` selects an existing folder; with `RUN_TRAINING=False`, use it to inspect results or assemble shared predictions. With training enabled, selected tasks run again. After reconnecting, mount Drive again, point `RUN_NAME` to the existing folder, and choose the tasks you need.

### Split work across teammates

Each teammate can set `TASKS` to one task and use a distinct `RUN_NAME`. Share the resulting `taskN_predictions.csv` files through a shared Drive folder. To assemble the final file, copy all five prediction files into one run folder, set `RUN_NAME` to that folder, set `RUN_TRAINING=False`, and run the notebook through its results and assembly cells. The final check also needs the matching Task 4 metadata; the notebook prepares the separate image dataset if it is missing. Each member's runtime remains separate, even when they share the same notebook.

Set `DOWNLOAD_RESULTS=True` if you also want the notebook to download a curated results archive. It contains predictions and metrics, not trained weights, raw competition data, or temporary caches. Checkpoints remain in the run folder on Drive.

## Run locally on Windows

For fast local iterations, the command-line interface remains available alongside the group notebook. The recorded local experiments used Python 3.12. Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the competition data from the [competition page](https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml) and place the original task folders in `../cpe342-kaggle-main`, relative to this repository. Keep the folder names from `01_binary_classification` to `05_anomaly_detection` and include `sample_submission.csv`. Use `--data-root` for a different location.

Task 4 images are distributed separately by the instructor. To download and extract them into the data root, run:

```powershell
.\.venv\Scripts\python.exe prepare_images.py --download
```

Use `--image-root` if the Task 4 dataset is in a separate folder. The local environment installs PyTorch from `requirements.txt`; the Colab notebook uses `requirements-colab.txt` and the runtime's existing PyTorch installation.

```powershell
.\.venv\Scripts\python.exe run.py audit
.\.venv\Scripts\python.exe run.py train --tasks 1 2 3 5 --profile balanced --folds 3 --threads 8
.\.venv\Scripts\python.exe run.py train --tasks 4 --profile balanced --epochs 5 --batch-size 128
.\.venv\Scripts\python.exe run.py assemble
.\.venv\Scripts\python.exe run.py validate --file outputs/main/submission.csv
.\.venv\Scripts\python.exe report_results.py
```

Generated files are written to `outputs/main/`. To rebuild the local teaching notebook after changing the implementation, run `python build_notebook.py`. The shared Colab notebook is generated by `python build_colab_notebook.py`.

## Repository contents

- `Task1/` to `Task5/` hold one folder per task: a `README.md` with the method and local results, and a `train.py` entry point (for example `python Task1/train.py`). `Task5/synthetic_eval.py` is the label-free check for Task 5.
- `pis/` is the shared library those entry points call: `tabular.py` (Tasks 1 to 3), `image.py` (Task 4), `anomaly.py` (Task 5), and the submission utilities.
- `experiments/` holds experiments that span several tasks, and `docs/KAGGLE_FINDINGS_TH.md` records leaderboard results and findings.
- `run.py` provides the local command-line interface.
- `Player_Intelligence_Colab.ipynb` is the group notebook for Colab; `Player_Intelligence_System.ipynb` is the local teaching notebook.
- `README_TH.md` contains Thai setup notes, and `STUDY_GUIDE_TH.md` explains the tasks and metrics.
- `requirements.txt` is for the local environment. `requirements-colab.txt` avoids reinstalling PyTorch in Colab.

Competition data, generated outputs, trained models, and local Python environments are excluded from version control. The Colab notebook clones this repository at run time; it does not require a fork or a separate model-code copy.

## Validation and limitations

Tasks 1 to 3 use cross-validation. Task 4 uses a holdout split that groups byte-identical images. Task 1's threshold-tuned OOF score is optimistic because the same OOF labels select and evaluate the threshold. Task 5 has no ground-truth labels, so its local macro F3 score cannot be calculated. Local validation scores are not Kaggle leaderboard scores.

The competition materials do not provide an explicit mapping from the `ANS` template IDs to each task's `player_id`; the implementation uses each task's test row order. Task 4 is the exception: its corrected `test.csv` (12 Nov) lists the `ANS` id of every image, and the submission joins on that id. The template's Task 4 values are numeric, while the Task 4 training CSV labels are strings; the submission maps them to the integer classes in the datacard (7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4). The organizers replaced the Task 2 and Task 5 datasets and the Task 4 `test.csv` on 12 Nov; `docs/KAGGLE_FINDINGS_TH.md` records what changed. The Task 5 instructions also need clarification about fitting an unsupervised model on the supplied unlabeled test data.

For team code changes, use separate branches and open pull requests for review. Each pull request should identify the task affected, describe the change, and include relevant validation results.
