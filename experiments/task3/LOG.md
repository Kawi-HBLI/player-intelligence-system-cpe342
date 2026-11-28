# Task 3 experiment log

`step` = the order experiments were run in; rows with the same step come from one run.
Dates are when the result was first committed (from git history). Fold scores use
`KFold(3, shuffle=True, random_state=342)` like the baseline unless the note says otherwise;
metric is `_score` from `pis/tabular.py`. LB = Kaggle leaderboard (— = not submitted).

Layout: `eda/` holds the notebooks (`eda.ipynb`, `missing_eda.ipynb`); `model/` holds one folder per model experiment
(`oracle/`, `amount_model/`, `refit/`, `clf_features/`, `ensemble/`, `quantile_calib/`, `tuning/`, `v9_impute/`), each with its script
and its `*_results.md`, plus `champion/` and `make_submission.py`. This log is shared and scripts append to it.

Dependencies: the pipeline (`python run.py ...`) needs only `requirements.txt`. The experiment scripts also need
`requirements-experiments.txt` (LightGBM, XGBoost, Optuna): `python -m pip install -r requirements-experiments.txt`.

Champion: `model/champion/` holds the current best model's OOF (`task3_oof.csv`) and `champion.json`.
Every experiment is compared fold by fold against both the baseline and the champion. A model is
promoted only if it beats the champion on all 3 folds (t = 0.5 rows, never OOF-tuned ones); the
promotion updates `model/champion/` and this log in the same commit. Current champion: **V9a** (step 15, protocol honest-es:
V8 + label-free imputation of `prev_month_spending`; V8 before it, step 14). Kaggle submission: V9a refit,
public LB 0.82369 (base 0.82343, +0.00026; 2026-11-20), the final Task 3 model.

Protocols: steps 1–13 early-stop on the outer validation fold that is also scored (`val-es`, as in `pis/tabular.py`),
so their OOF scores are slightly optimistic. From step 14 on, scores use `honest-es`: each outer-training part is split
once into fit (85%) and es (15%) with seed 342 + fold (stratified on y > 0 for two-stage models), models early-stop on
es and the validation fold is only scored; final predictions are clipped to [0, 300,000] (`cap_predictions`).
`champion.json` names its `protocol` and keeps the old numbers under `previous_protocol`. **Compare scores only within
one protocol.**

Known issue (step 11, `model/clf_features/clf_features.py`, a negative-result screen): its reference AUC 0.7841 was recorded with 12 CatBoost threads; with all cores the reference is 0.7845 and the script now warns instead of stopping. `--threads 12` reproduces the step 11 table exactly; with all cores the decision is the same (STOP, best gain +0.0003 < +0.002), so the result is unchanged.

## Reproduce the final Task 3 predictions

**V9a (current champion).** From a fresh clone, after the venv setup below:

```powershell
.\.venv\Scripts\python.exe experiments/task3/model/v9_impute/v9_impute.py --threads 12
.\.venv\Scripts\python.exe experiments/task3/model/refit/refit.py --variant v9a --threads 12
```

- `v9_impute.py` (about 17 min) re-runs V8 with both seeds and stops unless it matches `model/tuning/v8.json`, then runs V9a/V9b;
  expected V9a 0.801877 / 0.787220 / 0.795327, OOF 0.794729 (seed 1342: OOF 0.794744). It writes `outputs/experiments/task3_v9a/`
  (OOF, test predictions, classifier p) and rewrites `model/v9_impute/v9_impute_results.md` and the step-15 rows here
  (`git checkout` them unless the results changed). Without `--promote` it does not touch `model/champion/`.
- `refit.py --variant v9a` (about 6 min) must pass all 7 checks; it writes `outputs/experiments/task3_v9a_refit/task3_predictions.csv`.
  Expected: fold-1 check 0.8020, sum 270,696,618 THB, median of non-zero 2,838.21, Spearman vs CV 0.9967
  (as in `model/refit/refit_v9a_results.md`). The imputer for the refit is fitted on the full train only.
- Report figures (chapter 4: OOF by version, error by bucket, regression to the mean) come from `report/report_figures.py`: `.\.venv\Scripts\python.exe experiments/task3/report/report_figures.py` writes `report/figures/` and must print `V9a OOF score 0.7947` and `top 1% rows share of error 39.5%`.

**V8 refit (previous champion, Kaggle 0.80684).** It can be rebuilt from the repo alone: no Optuna, no `outputs/` from another
machine. `v8_cv.py` rebuilds V8's CV (honest-es classifier p, OOF, per-fold test predictions) from
`model/tuning/v8.json` and checks the fold scores against `model/champion/champion.json`; `refit.py` then trains on the full train.
From a fresh clone, with the competition data in `../cpe342-kaggle-main` (see the root README):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiments/task3/model/tuning/v8_cv.py --threads 12
.\.venv\Scripts\python.exe experiments/task3/model/refit/refit.py --variant v8 --threads 12
```

- `v8_cv.py` must print `OK: V8 reproduced within 0.0001 on every fold` (expected 0.798653 / 0.784773 / 0.793209,
  OOF 0.792138); it writes `outputs/experiments/task3_v8/` and `outputs/experiments/task3_v4_honest/classifier_p.npz`.
- `refit.py` must pass all 7 checks; it writes `outputs/experiments/task3_v8_refit/task3_predictions.csv`, the Task 3
  column of the final submission. Expected sanity numbers: sum 270,049,550 THB, median of non-zero 2,848.09,
  Spearman vs CV 0.9966 (as in `model/refit/refit_v8_results.md`).
- `--threads 12` matters: CatBoost on CPU builds slightly different trees with a different thread count, and V8 was
  recorded with 12. With 12 threads the CV is exact (OOF identical to `model/champion/task3_oof.csv`). With 16 threads the
  classifier early-stopped at 95 / 106 / 273 instead of 98 / 182 / 240 and the fold scores moved by up to 4e-5,
  still within the 1e-4 check. The classifier p is cached, so after changing `--threads` delete
  `outputs/experiments/task3_v4_honest/classifier_p.npz`.
- Runtime on a 16-thread laptop with `--threads 12`: `v8_cv.py` about 2 min (classifier 8 s, amount model
  30–42 s per fold), `refit.py` about 5 min (fold-1 check 52 s, each of the 3 seeds about 75 s).
- `refit.py` rewrites `model/refit/refit_v8_results.md` (runtimes, and it drops the Determinism section when no earlier
  refit file exists); `git checkout` it unless the results changed.

| step | date | branch | change | fold scores | OOF | LB | note |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-10 | main | Baseline: CatBoost MAE, 1000 iters, depth 6, lr 0.05, ES 80 | 0.7873 / 0.7587 / 0.7807 | 0.7753 | — | `pis/tabular.py`, `baseline/task3_oof.csv`; LB only as part of the full submission |
| 2 | 2026-11-10 | main | CatBoost longer: 5 folds, ES 300 | — | 0.7737 | — | `experiments/long_catboost_5fold.py`; 5-fold split, not directly comparable; see docs/KAGGLE_FINDINGS_TH.md |
| 3 | 2026-11-11 | main | log1p target + RMSE | — | 0.7716 | — | docs/KAGGLE_FINDINGS_TH.md; worse than baseline |
| 3 | 2026-11-11 | main | log1p target + 3 spend features | — | 0.7737 | — | docs/KAGGLE_FINDINGS_TH.md; worse than baseline |
| 3 | 2026-11-11 | main | Blend 70% baseline + 30% log1p model | — | 0.7767 | — | docs/KAGGLE_FINDINGS_TH.md; +0.0014 < fold spread, not adopted |
| 4 | 2026-11-14 | not committed | CatBoost loss MAE → Huber:delta=1000 (rest as baseline) | 0.7838 / 0.7706 / 0.7808 | 0.7784 | — | onboarding guide example A4 (not in repo); loses fold 1 (−0.0035), so not adopted; delta 500/5000 untested |
| 5 | 2026-11-15 | task3/eda | EDA notebook (no model) | — | — | — | `eda/eda.ipynb` (PR #1); >20k = 63% of baseline error; y ≈ prev × segment×VIP multiplier |
| 6 | 2026-11-15 | task3/oracle | Oracle B: true zeros + baseline amount | 0.8406 / 0.8135 / 0.8335 | 0.8290 | — | `model/oracle/` (PR #2); upper bound, NOT submittable |
| 6 | 2026-11-15 | task3/oracle | Oracle C: true zeros + prev × segment ratio | 0.7591 / 0.7556 / 0.7650 | 0.7599 | — | upper bound, NOT submittable |
| 6 | 2026-11-15 | task3/oracle | Oracle D: true zeros + prev × segment×VIP ratio | 0.7906 / 0.7855 / 0.7939 | 0.7900 | — | upper bound, NOT submittable |
| 6 | 2026-11-15 | task3/oracle | Oracle E: baseline zeros + true amount | 0.9255 / 0.9184 / 0.9250 | 0.9229 | — | upper bound, NOT submittable |
| 7 | 2026-11-15 | task3/amount-model | V1: baseline + expected_spend features | 0.7737 / 0.7610 / 0.7691 | 0.7679 | — | wins all folds: no |
| 7 | 2026-11-15 | task3/amount-model | V2: two-stage (Logloss clf + MAE reg on spenders, THB) | 0.7919 / 0.7766 / 0.7863 | 0.7849 | — | t=0.5; OOF-tuned t=0.500: 0.7849 (optimistic); wins all folds: yes |
| 7 | 2026-11-15 | task3/amount-model | V3: two-stage (Logloss clf + MAE reg on log(y/prev)) | 0.7958 / 0.7823 / 0.7923 | 0.7900 | 0.80313 (base 0.80097, +0.00216) | t=0.5; OOF-tuned t=0.500: 0.7900 (optimistic); wins all folds: yes; **champion** (first, replaces baseline); Kaggle check 2026-11-15 (task3 column swap via `make_submission.py`) confirms the CV gain: ≈ +0.011 on Task 3 if the 5 task weights are equal, vs +0.0147 OOF |
| 8 | 2026-11-15 | task3/v3-tuning | V3a: V3 with amount regressor iterations 3000 | 0.7960 / 0.7823 / 0.7922 | 0.7901 | — | t=0.5; OOF-tuned t=0.500: 0.7901 (optimistic); wins all folds vs baseline: yes; vs champion (V3 two-stage, log-ratio amount, t=0.5): +0.0001, wins all folds: no |
| 8 | 2026-11-15 | task3/v3-tuning | V3b: V3a + classifier depth 8, lr 0.03 | 0.7960 / 0.7823 / 0.7922 | 0.7901 | — | t=0.5; OOF-tuned t=0.500: 0.7901 (optimistic); wins all folds vs baseline: yes; vs champion (V3 two-stage, log-ratio amount, t=0.5): +0.0001, wins all folds: no |
| 9 | 2026-11-15 | task3/quantile-combine | V4-median (sanity): V4 MultiQuantile amount model, α=0.5 output only, t=0.5 | 0.7967 / 0.7826 / 0.7925 | 0.7905 | — | sanity row, not a candidate; should be close to V3; wins all folds vs baseline: yes; vs champion (V3 two-stage, log-ratio amount, t=0.5): +0.0005, wins all folds: yes |
| 9 | 2026-11-15 | task3/quantile-combine | V4: V3 classifier + MultiQuantile log(y/prev) amount, amount = (1 − 0.5/p) quantile | 0.7982 / 0.7841 / 0.7934 | 0.7918 | — | wins all folds vs baseline: yes; vs champion (V3 two-stage, log-ratio amount, t=0.5): +0.0018, wins all folds: yes; **PROMOTED to champion** |
| 9 | 2026-11-15 | task3/quantile-combine | V4s: V4 with q = 1 − s·0.5/p, s tuned on OOF | 0.7982 / 0.7841 / 0.7934 | 0.7918 | — | best s=1.0 on the grid 0.6–1.4 (OOF-tuned, optimistic, not a candidate); wins all folds vs baseline: yes; vs champion (V3 two-stage, log-ratio amount, t=0.5): +0.0018, wins all folds: yes |
| 10 | 2026-11-15 | task3/refit | V4 refit (full train, 3 seeds) | — | — (not measurable) | 0.80371 (base 0.80097, +0.00274; +0.00058 vs V3) | no CV possible, judged on Kaggle; iterations clf 205, amount 1035 (CV mean × 1.2); fold-1 check 0.7983 vs V4 CV 0.7982 (+0.0002); vs V4 CV test preds: Spearman 0.9974, zero share 0.5035 vs 0.5052, sum +0.24%, median non-zero 2,859 vs 2,867, mean p 0.5176 vs 0.5176; runtime per seed clf 2s + amount 47s; checks: all pass; `model/refit/`, not promoted (champion rule needs fold scores); V4 alone was expected ≈ +0.0004 overall (OOF +0.0018 / 5), so refit likely adds a little; within public-LB noise, not separable. Adopted as the Task 3 submission. |
| 11 | 2026-11-15 | task3/clf-features | Classifier feature screen: V4 clf + timing / activity / spend_habit / zero_rate groups | AUC 0.7881 / 0.7826 / 0.7831 (all-in) | AUC 0.7845 (all-in) | — | classifier only, metric = AUC (not the Task 3 score); reference V4 clf AUC 0.7841 (0.7878 / 0.7825 / 0.7824); timing +0.0003 (wins every fold: yes); activity +0.0002 (wins every fold: no); spend_habit +0.0002 (wins every fold: no); zero_rate +0.0003 (wins every fold: no); all 4 groups +0.0003 (wins every fold: yes); STOP: kept groups: timing; best kept combination (+ timing) adds +0.0003 OOF AUC < +0.002, so no V5 run (best of all configs, kept or not: + all 4 groups +0.0003). `model/clf_features/`; runtime 37s |
| 12 | 2026-11-15 | task3/ensemble | V6 reference: V4 classifier + LightGBM quantile amount alone (w=0) | 0.7913 / 0.7787 / 0.7879 | 0.7859 | — | reference row, not a candidate; wins all folds vs baseline: yes; vs champion (V4): -0.0059 (-0.00683 / -0.00540 / -0.00548), wins all folds: no |
| 12 | 2026-11-15 | task3/ensemble | V6: V4 + amount quantiles blended 0.5 CatBoost MultiQuantile + 0.5 LightGBM quantile (log-ratio scale) | 0.7970 / 0.7829 / 0.7922 | 0.7906 | — | wins all folds vs baseline: yes; vs champion (V4): -0.0012 (-0.00117 / -0.00116 / -0.00117), wins all folds: no; spenders-only α=0.5: blend 0.8526 vs CatBoost 0.8541 vs LightGBM 0.8462; STOP: V6 does not beat V4 on all 3 folds (Δ per fold -0.00117 / -0.00116 / -0.00117); champion stays V4, no refit, no Kaggle file; `model/ensemble/`; LightGBM fits 102s over 3 folds |
| 12 | 2026-11-15 | task3/ensemble | V6w: V6 with blend weight w tuned on OOF | 0.7983 / 0.7841 / 0.7934 | 0.7918 | — | best w=0.9 on the grid 0.0–1.0 (OOF-tuned, optimistic, not a candidate); wins all folds vs baseline: yes; vs champion (V4): +0.0000 (+0.00012 / -0.00000 / +0.00001), wins all folds: no |
| 12 | 2026-11-15 | task3/ensemble | V6x reference: V4 classifier + XGBoost multi-quantile amount alone (w=0) | 0.7929 / 0.7800 / 0.7888 | 0.7872 | — | reference row, not a candidate; wins all folds vs baseline: yes; vs champion (V4): -0.0046 (-0.00531 / -0.00407 / -0.00456), wins all folds: no |
| 12 | 2026-11-15 | task3/ensemble | V6x: V4 + amount quantiles blended 0.5 CatBoost MultiQuantile + 0.5 XGBoost multi-quantile (log-ratio scale) | 0.7975 / 0.7834 / 0.7926 | 0.7911 | — | wins all folds vs baseline: yes; vs champion (V4): -0.0007 (-0.00065 / -0.00068 / -0.00081), wins all folds: no; spenders-only α=0.5: blend 0.8530 vs CatBoost 0.8541 vs XGBoost 0.8474; STOP: V6x does not beat V4 on all 3 folds (Δ per fold -0.00065 / -0.00068 / -0.00081); champion stays V4, no refit, no Kaggle file; `model/ensemble/`; XGBoost fits 223s over 3 folds |
| 12 | 2026-11-15 | task3/ensemble | V6xw: V6x with blend weight w tuned on OOF | 0.7983 / 0.7841 / 0.7934 | 0.7919 | — | best w=0.9 on the grid 0.0–1.0 (OOF-tuned, optimistic, not a candidate); wins all folds vs baseline: yes; vs champion (V4): +0.0001 (+0.00013 / +0.00003 / +0.00002), wins all folds: yes |
| 13 | 2026-11-15 | task3/quantile-calib | V7-grid (diagnostic): V4 rule (q ≥ 0.05, no remap, no cap) on a MultiQuantile model with 14 alphas 0.01–0.50 | 0.7985 / 0.7839 / 0.7930 | 0.7917 | — | diagnostic row, not a candidate; wins all folds vs baseline: yes; vs champion (V4): -0.0001 (+0.00036 / -0.00016 / -0.00037), wins all folds: no |
| 13 | 2026-11-15 | task3/quantile-calib | V7a: V4 + 14-alpha MultiQuantile (0.01–0.50), q clipped to ≥ 0.01, prediction capped at outer-train max | 0.7987 / 0.7841 / 0.7930 | 0.7919 | — | cap at outer-train max: 8 OOF rows / 1 test rows capped; wins all folds vs baseline: yes; vs champion (V4): +0.0001 (+0.00049 / +0.00002 / -0.00036), wins all folds: no; STOP: no V7 variant beats V4 on all folds; champion stays V4, no refit, no Kaggle file |
| 13 | 2026-11-15 | task3/quantile-calib | V7b: V7a model + in-fold coverage calibration of q (inner KFold(5) OOF), q ≥ 0.05 before remap, cap | 0.7986 / 0.7841 / 0.7931 | 0.7919 | — | cap at outer-train max: 8 OOF rows / 1 test rows capped; wins all folds vs baseline: yes; vs champion (V4): +0.0001 (+0.00046 / +0.00003 / -0.00029), wins all folds: no; calibration: 5 inner fits per fold, 718s over 3 folds; STOP: no V7 variant beats V4 on all folds; champion stays V4, no refit, no Kaggle file |
| 13 | 2026-11-15 | task3/quantile-calib | V7: V7a + V7b (q ≥ 0.01, coverage remap, cap) | 0.7986 / 0.7841 / 0.7931 | 0.7919 | — | cap at outer-train max: 8 OOF rows / 1 test rows capped; wins all folds vs baseline: yes; vs champion (V4): +0.0001 (+0.00045 / +0.00003 / -0.00029), wins all folds: no; calibration: 5 inner fits per fold, 718s over 3 folds; STOP: no V7 variant beats V4 on all folds; champion stays V4, no refit, no Kaggle file; `model/quantile_calib/`; run 1043s |
| 14 | 2026-11-15 | task3/tuning | V4-honest: V4 recipe under honest-es (fit/es 85/15 split of the outer-train part, stratified on y > 0), cap | 0.7970 / 0.7840 / 0.7926 | 0.7911 | — | protocol honest-es; old V4 (val-es) 0.7982 / 0.7841 / 0.7934, OOF 0.7918; Δ -0.0007 (-0.0011 / -0.0000 / -0.0008); best iterations clf 98/182/240, amount 731/947/608; cap: 8 OOF rows (0.7910 uncapped); **recorded in champion.json as the reference of protocol honest-es** (compare only within one protocol from here on) |
| 14 | 2026-11-15 | task3/tuning | Baseline-honest: baseline CatBoost MAE under honest-es (plain 85/15 fit/es split), cap | 0.7830 / 0.7721 / 0.7655 | 0.7735 | — | protocol honest-es; old baseline (val-es) 0.7753, Δ -0.0018; honest baseline → V4-honest gain +0.0177 (+0.0140 / +0.0119 / +0.0270) vs +0.0165 under val-es; best iterations 998/1000/883; `model/tuning/honest_es.py`, run 112s |
| 14 | 2026-11-15 | task3/tuning | Optuna: V4 amount model (CatBoost MultiQuantile) tuned under honest-es, TPE seed 342, MedianPruner after fold 1, budget 40 trials / 3 h | 0.7985 / 0.7848 / 0.7934 | 0.7922 | — | 40 finished trials (28 complete, 12 pruned, 0 failed), 0.75 h of trial time; best trial #18 OOF 0.7922 (+0.0010 vs V4-honest; max over trials, optimistic): depth 5, lr 0.0204, l2 4.62, rs 2.33, bt 0.27, border 64, SymmetricTree; tunes stage 2 only, classifier p fixed; `model/tuning/`, outputs/optuna/task3_amount.db |
| 14 | 2026-11-15 | task3/tuning | V8: V4 + Optuna-tuned amount model (best mean of 2 seed runs among the top 3 trials), seed 342 + fold | 0.7987 / 0.7848 / 0.7932 | 0.7921 | — | trial #16: depth 4, lr 0.0208, l2 7.81, rs 2.13, bt 0.11, border 64, SymmetricTree; vs champion (V4-honest): +0.0010 (+0.00163 / +0.00073 / +0.00063), wins all folds: yes; best iterations 2969/3127/2096; **PROMOTED to champion** (protocol honest-es) |
| 14 | 2026-11-15 | task3/tuning | V8 re-run: same model, seed 1342 + fold (selection-noise check) | 0.7983 / 0.7846 / 0.7934 | 0.7920 | — | vs V4-honest: +0.0009 (+0.00132 / +0.00059 / +0.00080), wins all folds: yes; mean of the 2 seed runs 0.7921; V4-honest with the same seed scheme: 0.7916 (+0.0004, seed noise) |
| 14 | 2026-11-15 | task3/tuning | V8 refit (full train, 3 seeds, cap) | — | — (not measurable) | 0.80684 (base 0.80676, +0.00008) | no CV possible, judged on Kaggle; iterations clf 208, amount 3277 (CV mean × 1.2); cap: 1 test row(s) changed; fold-1 check 0.7991 vs V8 CV (honest-es) 0.7987 (+0.0004); vs V8 CV test preds: Spearman 0.9966, zero share 0.5035 vs 0.5068, sum +0.21%, median non-zero 2,848 vs 2,874, mean p 0.5176 vs 0.5177; uncapped = earlier refit file (max |Δ| 0); runtime per seed clf 2s + amount 47s; checks: all pass; `model/refit/`, not promoted (champion rule needs fold scores) |
| 14 | 2026-11-15 | task3/tuning | V4 refit + cap (full train, 3 seeds, predictions clipped to [0, 300,000]) | — | — (not measurable) | — | no CV possible, judged on Kaggle; iterations clf 205, amount 1035 (CV mean × 1.2); cap: 1 test row(s) changed; fold-1 check 0.7985 vs V4 CV (val-es) 0.7982 (+0.0003); vs V4 CV test preds: Spearman 0.9974, zero share 0.5035 vs 0.5052, sum +0.19%, median non-zero 2,859 vs 2,867, mean p 0.5176 vs 0.5176; uncapped = earlier refit file (max |Δ| 0); runtime per seed clf 2s + amount 47s; checks: all pass; `model/refit/`, not promoted (champion rule needs fold scores) |
| 15 | 2026-11-19 | task3/v9-impute | V9a: V8 + label-free imputation of prev_month_spending (in-fold, outer-train rows only), amount seed 342 + fold | 0.8019 / 0.7872 / 0.7953 | 0.7947 | — | protocol honest-es; amount seed 342 + fold; vs V8 same seed: +0.0032 / +0.0024 / +0.0021 (OOF +0.0026, wins all folds: yes); prev-missing rows 0.7982 vs V8 0.7582; imputer R² 0.994+ (prev); eligible; **promoted**; `model/v9_impute/` |
| 15 | 2026-11-19 | task3/v9-impute | V9a: V8 + label-free imputation of prev_month_spending (in-fold, outer-train rows only), amount seed 1342 + fold | 0.8019 / 0.7872 / 0.7954 | 0.7947 | — | protocol honest-es; amount seed 1342 + fold; vs V8 same seed: +0.0036 / +0.0025 / +0.0020 (OOF +0.0027, wins all folds: yes); prev-missing rows 0.7982 vs V8 0.7566; imputer R² 0.994+ (prev); eligible; `model/v9_impute/` |
| 15 | 2026-11-19 | task3/v9-impute | V9b: V8 + label-free imputation of prev / historical / avg_transaction_value (in-fold), amount seed 342 + fold | 0.8019 / 0.7875 / 0.7954 | 0.7949 | — | protocol honest-es; amount seed 342 + fold; vs V8 same seed: +0.0033 / +0.0027 / +0.0022 (OOF +0.0027, wins all folds: yes); prev-missing rows 0.7984 vs V8 0.7582; imputer R² 0.994+ (prev); eligible; `model/v9_impute/` |
| 15 | 2026-11-19 | task3/v9-impute | V9b: V8 + label-free imputation of prev / historical / avg_transaction_value (in-fold), amount seed 1342 + fold | 0.8019 / 0.7874 / 0.7954 | 0.7948 | — | protocol honest-es; amount seed 1342 + fold; vs V8 same seed: +0.0036 / +0.0027 / +0.0020 (OOF +0.0028, wins all folds: yes); prev-missing rows 0.7984 vs V8 0.7566; imputer R² 0.994+ (prev); eligible; `model/v9_impute/` |
| 15 | 2026-11-19 | task3/v9-impute | V9a refit (full train, 3 seeds, cap, imputer fitted on the full train) | — | — (not measurable) | 0.82369 (base 0.82343, +0.00026) | no CV possible, judged on Kaggle; iterations clf 133, amount 2454 (CV mean × 1.2); cap: 1 test row(s) changed; fold-1 check 0.8020 vs V9a CV (honest-es) 0.8019 (+0.0002); vs V9a CV test preds: Spearman 0.9967, zero share 0.5027 vs 0.5066, sum +0.11%, median non-zero 2,838 vs 2,868, mean p 0.5177 vs 0.5177; runtime per seed clf 3s + amount 92s; checks: all pass; `model/refit/`, not promoted (champion rule needs fold scores) |
