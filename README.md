# Player Intelligence System

This repository contains the implementation for the CPE342 Machine Learning competition, Player Intelligence System. The project covers anti-cheat detection, player segmentation, monthly spending prediction, game screenshot classification, and account security monitoring.

The tabular tasks use CatBoost, the image classification task uses ResNet18, and the anomaly detection task uses an Isolation Forest ensemble. The implementation includes data checks, validation, model training, and submission file generation.

## Repository contents

The `pis` directory contains the task implementations. `run.py` provides the command-line interface, and `Player_Intelligence_System.ipynb` provides a self-contained notebook. Detailed learning notes are available in [STUDY_GUIDE_TH.md](STUDY_GUIDE_TH.md), with additional setup instructions in [README_TH.md](README_TH.md).

Competition datasets, generated predictions, trained models, and local Python environments are excluded from version control. The published notebook has no saved execution outputs. An executed copy is kept locally under `outputs/main/Player_Intelligence_System.executed.ipynb`.

## Setup

Python 3.12 was used for the recorded experiments. Install the dependencies in a virtual environment. The following commands use Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the data from the [competition page](https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml) and place the original task folders in `../cpe342-kaggle-main`, relative to the repository root. Keep the original folder names, from `01_binary_classification` to `05_anomaly_detection`, together with `sample_submission.csv`. A different location can be supplied with `--data-root`.

Task 4 images are distributed separately through the link provided by the instructor. `prepare_images.py --download` downloads and extracts that archive. If images are stored in a separate dataset directory, supply its location with `--image-root`.

## Running the project

```powershell
.\.venv\Scripts\python.exe run.py audit
.\.venv\Scripts\python.exe run.py train --tasks 1 2 3 5 --profile balanced --folds 3 --threads 8
.\.venv\Scripts\python.exe run.py train --tasks 4 --profile balanced --epochs 5 --batch-size 128
.\.venv\Scripts\python.exe run.py assemble
.\.venv\Scripts\python.exe run.py validate --file outputs/main/submission.csv
.\.venv\Scripts\python.exe report_results.py
```

Generated files are written to `outputs/main`. The notebook can also be used on Kaggle with the competition datasets attached. Set `RUN_TRAINING=True` to train the models in a new environment. Rebuild the notebook with `python build_notebook.py` after changing the Python implementation.

## Validation and limitations

Tasks 1 to 3 use cross-validation. Task 4 uses a holdout split that keeps exact duplicate images together. Task 5 has no ground-truth labels, so its F3 score cannot be measured locally. Local validation results should not be presented as Kaggle leaderboard scores.

The competition materials contain differences in the submission format and Task 4 labels. The current implementation uses each task's test row order for the submission IDs and preserves the image labels from the training CSV. The Task 5 instructions also require clarification about fitting an unsupervised model on the supplied unlabeled test data. These assumptions should be confirmed with the teaching team before the final submission.

Team members should make changes on separate branches and open pull requests for review. Each pull request should identify the task affected, describe the change, and include relevant validation results.
