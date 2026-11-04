"""Leakage-aware cross-validation for the three tabular competition tasks."""
from pathlib import Path
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             fbeta_score, mean_absolute_error)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

LOG = logging.getLogger(__name__)
TASKS = {1: ('01_binary_classification', 'is_cheater'),
         2: ('02_multiclass_classification', 'segment'),
         3: ('03_regression', 'spending_30d')}
ITERATIONS = {'quick': 250, 'balanced': 1000, 'strong': 2000}


def _score(y, prediction):
    denominator = float(np.sum(y))
    if denominator <= 0:
        raise ValueError('Task 3 score requires a positive sum of targets')
    return 1.0 / (1.0 + float(np.abs(np.asarray(y) - prediction).sum()) / denominator)


def _threshold(y, probabilities):
    # At each distinct probability include every observation with probability >= threshold.
    order = np.argsort(-probabilities, kind='stable')
    p, target = probabilities[order], np.asarray(y)[order].astype(int)
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(target)[ends].astype(float)
    fp = (ends + 1) - tp
    fn = target.sum() - tp
    tn = len(target) - tp - fp - fn
    pos = np.divide(5 * tp, 5 * tp + 4 * fn + fp, out=np.zeros_like(tp), where=(5 * tp + 4 * fn + fp) != 0)
    neg = np.divide(5 * tn, 5 * tn + 4 * fp + fn, out=np.zeros_like(tn), where=(5 * tn + 4 * fp + fn) != 0)
    scores = (pos + neg) / 2
    none_score = fbeta_score(target, np.zeros_like(target), beta=2, average='macro', zero_division=0)
    i = int(np.argmax(scores))
    return (float(np.nextafter(p[0], np.inf)), float(none_score)) if none_score > scores[i] else (float(p[ends[i]]), float(scores[i]))


def _prepare(train, test, target):
    if 'player_id' not in train or 'player_id' not in test or target not in train:
        raise ValueError('Missing player_id or training target')
    if target in test:
        raise ValueError('Test data must not include the target')
    columns = [c for c in train.columns if c not in ('player_id', target)]
    if set(columns) != set(test.columns) - {'player_id'}:
        raise ValueError('Training and test feature columns differ')
    x, xt = train[columns].copy(), test[columns].copy()
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(x[c])]
    for c in categorical:
        for frame in (x, xt):
            frame[c] = frame[c].astype('string').fillna('__MISSING__').astype(str)
    for c in set(columns) - set(categorical):
        for frame in (x, xt):
            frame[c] = pd.to_numeric(frame[c], errors='raise').replace([np.inf, -np.inf], np.nan)
    return x, xt, categorical


def train_task(task: int, data_root: Path, output_dir: Path, folds: int = 3,
               seed: int = 342, profile: str = 'balanced', threads: int = -1,
               engine: str = 'auto') -> dict:
    """Train fold models, write OOF/test predictions, models, and metrics.

    Task 1's threshold is selected on the same OOF labels used to report its
    tuned score, which makes that score optimistic. Fixed-threshold score is
    reported separately. Test predictions average fold probabilities/values.
    """
    if task not in TASKS or profile not in ITERATIONS or engine not in ('auto', 'catboost', 'sklearn'):
        raise ValueError('Invalid task, profile, or engine')
    if folds < 2 or threads == 0 or threads < -1:
        raise ValueError('folds must be >= 2 and threads must be -1 or positive')
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        has_catboost = True
    except ImportError:
        has_catboost = False
    if engine == 'catboost' and not has_catboost:
        raise ImportError('CatBoost requested but not installed')
    chosen = 'catboost' if has_catboost and engine != 'sklearn' else 'sklearn'
    folder, target = TASKS[task]
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_root / folder / 'train.csv')
    test = pd.read_csv(data_root / folder / 'test.csv')
    if target not in train:
        raise ValueError('Missing training target')
    n_train_total = len(train)
    unlabeled = train.loc[train[target].isna(), ['player_id']].copy()
    unlabeled.to_csv(output_dir / f'task{task}_unlabeled_ids.csv', index=False)
    train = train.loc[train[target].notna()].reset_index(drop=True)
    if train.empty:
        raise ValueError('No labeled training rows')
    x, xt, categorical = _prepare(train, test, target)
    y = train[target].to_numpy()
    if task == 3 and (np.any(y < 0) or np.sum(y) <= 0):
        raise ValueError('Task 3 targets must be nonnegative with positive sum')
    classes = np.unique(y) if task != 3 else None
    if task == 1 and not np.array_equal(classes, [0, 1]):
        raise ValueError('Task 1 requires binary target values 0 and 1')
    if task != 3 and (len(classes) < 2 or min(np.unique(y, return_counts=True)[1]) < folds):
        raise ValueError('Each class must have at least folds observations')
    splitter = (StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed) if task != 3
                else KFold(n_splits=folds, shuffle=True, random_state=seed))
    n_outputs = len(classes) if task != 3 else 1
    oof = np.zeros((len(train), n_outputs)); test_preds = np.zeros((len(test), n_outputs))
    baseline = np.empty(len(train), dtype=y.dtype)
    importances = []
    fold_scores, best_iterations = [], []
    iters = ITERATIONS[profile]
    nthreads = os.cpu_count() or 1 if threads == -1 else threads
    with threadpool_limits(limits=nthreads):
        for fold, (tr, va) in enumerate(splitter.split(x, y if task != 3 else None), 1):
            LOG.info('Task %s fold %s/%s (%s)', task, fold, folds, chosen)
            baseline[va] = (pd.Series(y[tr]).mode().iloc[0] if task != 3 else np.median(y[tr]))
            if chosen == 'catboost':
                params = dict(iterations=iters, depth=6, learning_rate=0.05, random_seed=seed + fold,
                              thread_count=threads, verbose=False, allow_writing_files=False,
                              early_stopping_rounds=80)
                if task == 3:
                    model = CatBoostRegressor(loss_function='MAE', **params)
                else:
                    model = CatBoostClassifier(loss_function='Logloss' if task == 1 else 'MultiClass', **params)
                model.fit(x.iloc[tr], y[tr], cat_features=categorical, eval_set=(x.iloc[va], y[va]), use_best_model=True)
                model.save_model(str(output_dir / f'task{task}_fold{fold}.cbm'))
                importances.append(model.get_feature_importance())
                best_iterations.append(int(model.get_best_iteration()) + 1)
                val_x, test_x = x.iloc[va], xt
            else:
                numeric = [c for c in x.columns if c not in categorical]
                preprocess = ColumnTransformer([
                    ('num', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical),
                ], remainder='drop')
                params = dict(max_iter=iters, learning_rate=0.05, max_leaf_nodes=31,
                              early_stopping=True, n_iter_no_change=40, random_state=seed + fold)
                estimator = (HistGradientBoostingRegressor(loss='absolute_error', **params) if task == 3
                             else HistGradientBoostingClassifier(**params))
                model = Pipeline([('preprocess', preprocess), ('model', estimator)])
                model.fit(x.iloc[tr], y[tr])
                best_iterations.append(int(model.named_steps['model'].n_iter_))
                joblib.dump(model, output_dir / f'task{task}_fold{fold}.joblib')
                val_x, test_x = x.iloc[va], xt
            if task == 3:
                oof[va, 0] = np.maximum(0, model.predict(val_x))
                test_preds[:, 0] += np.maximum(0, model.predict(test_x)) / folds
                fold_scores.append(_score(y[va], oof[va, 0]))
            else:
                fold_classes = model.classes_ if chosen == 'catboost' else model.named_steps['model'].classes_
                indices = [int(np.flatnonzero(fold_classes == c)[0]) for c in classes]
                oof[va] = model.predict_proba(val_x)[:, indices]
                test_preds += model.predict_proba(test_x)[:, indices] / folds
                fold_pred = (oof[va, 1] >= 0.5).astype(int) if task == 1 else classes[oof[va].argmax(axis=1)]
                fold_scores.append(float(fbeta_score(y[va], fold_pred, beta=2, average='macro', zero_division=0) if task == 1 else f1_score(y[va], fold_pred, average='macro', zero_division=0)))
    metrics = {'task': task, 'engine': chosen, 'profile': profile, 'folds': folds, 'seed': seed,
               'threads': threads, 'iterations_limit': iters, 'n_train': len(train), 'n_test': len(test),
               'n_train_total': n_train_total, 'n_labeled': len(train), 'n_unlabeled_dropped': len(unlabeled),
               'fold_scores': fold_scores, 'fold_score_note': 'Task 1 uses fixed threshold 0.5',
               'best_iterations': best_iterations}
    oof_frame = pd.DataFrame({'player_id': train.player_id, 'target': y})
    if task == 1:
        threshold, tuned = _threshold(y, oof[:, 1])
        oof_prediction = (oof[:, 1] >= threshold).astype(int)
        prediction = (test_preds[:, 1] >= threshold).astype(int)
        metrics.update(metric='macro_f2', oof_score=tuned, threshold=threshold,
                       threshold_note='Threshold tuned and evaluated on the same OOF labels; tuned score is optimistic.',
                       fixed_0_5_oof_score=float(fbeta_score(y, oof[:, 1] >= 0.5, beta=2, average='macro', zero_division=0)),
                       baseline_score=float(fbeta_score(y, baseline, beta=2, average='macro', zero_division=0)))
        oof_frame['probability_1'] = oof[:, 1]
    elif task == 2:
        oof_prediction, prediction = classes[oof.argmax(axis=1)], classes[test_preds.argmax(axis=1)]
        metrics.update(metric='macro_f1', oof_score=float(f1_score(y, oof_prediction, average='macro', zero_division=0)),
                       baseline_score=float(f1_score(y, baseline, average='macro', zero_division=0)))
        for i, label in enumerate(classes):
            oof_frame[f'probability_{label}'] = oof[:, i]
    else:
        oof_prediction, prediction = oof[:, 0], test_preds[:, 0]
        metrics.update(metric='1/(1+sum_absolute_error/sum_target)', oof_score=_score(y, oof_prediction),
                       mae=float(mean_absolute_error(y, oof_prediction)), baseline_score=_score(y, baseline),
                       baseline_mae=float(mean_absolute_error(y, baseline)))
    if task != 3:
        metrics['classification_report'] = classification_report(y, oof_prediction, output_dict=True, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y, oof_prediction, labels=classes).tolist()
        metrics['confusion_matrix_labels'] = [v.item() if isinstance(v, np.generic) else v for v in classes]
        metrics['classification_report_note'] = 'Task 1 uses OOF-tuned threshold and is optimistic.' if task == 1 else 'OOF predictions'
        probability_frame = pd.DataFrame({'player_id': test.player_id})
        for i, label in enumerate(classes):
            probability_frame[f'probability_{label}'] = test_preds[:, i]
        probability_frame.to_csv(output_dir / f'task{task}_probabilities.csv', index=False)
    if importances:
        metrics['feature_importance'] = dict(sorted(zip(x.columns, np.mean(importances, axis=0).tolist()), key=lambda item: item[1], reverse=True))
    oof_frame['prediction'] = oof_prediction
    oof_frame.to_csv(output_dir / f'task{task}_oof.csv', index=False)
    pd.DataFrame({'player_id': test.player_id, 'prediction': prediction}).to_csv(output_dir / f'task{task}_predictions.csv', index=False)
    (output_dir / f'task{task}_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    return metrics
