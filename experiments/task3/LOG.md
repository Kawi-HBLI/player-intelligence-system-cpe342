# Task 3 experiment log

`step` = the order experiments were run in; rows with the same step come from one run.
Dates are when the result was first committed (from git history). Fold scores use
`KFold(3, shuffle=True, random_state=342)` like the baseline unless the note says otherwise;
metric is `_score` from `pis/tabular.py`. LB = Kaggle leaderboard (— = not submitted).

Layout: each experiment has its own folder (`eda/`, `oracle/`, `amount_model/`) with its script
or notebook and its `*_results.md`; this log is shared and scripts append to it.

| step | date | branch | change | fold scores | OOF | LB | note |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-10 | main | Baseline: CatBoost MAE, 1000 iters, depth 6, lr 0.05, ES 80 | 0.7873 / 0.7587 / 0.7807 | 0.7753 | — | `pis/tabular.py`, `baseline/task3_oof.csv`; LB only as part of the full submission |
| 2 | 2026-11-10 | main | CatBoost longer: 5 folds, ES 300 | — | 0.7737 | — | `experiments/long_catboost_5fold.py`; 5-fold split, not directly comparable; see docs/KAGGLE_FINDINGS_TH.md |
| 3 | 2026-11-11 | main | log1p target + RMSE | — | 0.7716 | — | docs/KAGGLE_FINDINGS_TH.md; worse than baseline |
| 3 | 2026-11-11 | main | log1p target + 3 spend features | — | 0.7737 | — | docs/KAGGLE_FINDINGS_TH.md; worse than baseline |
| 3 | 2026-11-11 | main | Blend 70% baseline + 30% log1p model | — | 0.7767 | — | docs/KAGGLE_FINDINGS_TH.md; +0.0014 < fold spread, not adopted |
| 4 | 2026-11-15 | task3/eda | EDA notebook (no model) | — | — | — | `eda/eda.ipynb` (PR #1); >20k = 63% of baseline error; y ≈ prev × segment×VIP multiplier |
| 5 | 2026-11-15 | task3/oracle | Oracle B: true zeros + baseline amount | 0.8406 / 0.8135 / 0.8335 | 0.8290 | — | `oracle/` (PR #2); upper bound, NOT submittable |
| 5 | 2026-11-15 | task3/oracle | Oracle C: true zeros + prev × segment ratio | 0.7591 / 0.7556 / 0.7650 | 0.7599 | — | upper bound, NOT submittable |
| 5 | 2026-11-15 | task3/oracle | Oracle D: true zeros + prev × segment×VIP ratio | 0.7906 / 0.7855 / 0.7939 | 0.7900 | — | upper bound, NOT submittable |
| 5 | 2026-11-15 | task3/oracle | Oracle E: baseline zeros + true amount | 0.9255 / 0.9184 / 0.9250 | 0.9229 | — | upper bound, NOT submittable |
| 6 | 2026-11-15 | task3/amount-model | V1: baseline + expected_spend features | 0.7737 / 0.7610 / 0.7691 | 0.7679 | — | wins all folds: no |
| 6 | 2026-11-15 | task3/amount-model | V2: two-stage (Logloss clf + MAE reg on spenders, THB) | 0.7919 / 0.7766 / 0.7863 | 0.7849 | — | t=0.5; OOF-tuned t=0.500: 0.7849 (optimistic); wins all folds: yes |
| 6 | 2026-11-15 | task3/amount-model | V3: two-stage (Logloss clf + MAE reg on log(y/prev)) | 0.7958 / 0.7823 / 0.7923 | 0.7900 | — | t=0.5; OOF-tuned t=0.500: 0.7900 (optimistic); wins all folds: yes |
| ? | ? | ? | Huber loss | ? | ? | — | TODO: numbers and date not recorded in the repo, fill in |
