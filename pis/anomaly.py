"""Task 5: unlabeled, transductive anomaly ranking. No invented F3 score."""
from pathlib import Path
import re
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from .common import write_json

LOG = logging.getLogger(__name__)


def temporal_features(frame):
    result = {}
    families = sorted({re.sub(r'_[1-4]$', '', col) for col in frame if re.search(r'_[1-4]$', col)})
    for family in families:
        cols = [f'{family}_{i}' for i in range(1, 5)]
        if not all(c in frame for c in cols):
            continue
        values = frame[cols]
        history = values.iloc[:, :3]
        mean = history.mean(axis=1)
        delta = values.iloc[:, 3] - mean
        result[f'{family}_last_delta'] = delta
        result[f'{family}_relative_delta'] = delta / (mean.abs() + 1)
        result[f'{family}_range'] = values.max(axis=1) - values.min(axis=1)
        result[f'{family}_first_last'] = values.iloc[:, 3] - values.iloc[:, 0]
    # Geographic discontinuity across successive observed periods (distance, not speed).
    for i in range(1, 4):
        cols = [f'login_lat_{i}', f'login_lon_{i}', f'login_lat_{i+1}', f'login_lon_{i+1}']
        if all(c in frame for c in cols):
            lat1, lon1, lat2, lon2 = [np.deg2rad(frame[c]) for c in cols]
            a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
            result[f'location_distance_km_{i}_{i+1}'] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return pd.DataFrame(result, index=frame.index).replace([np.inf, -np.inf], np.nan)


def train_anomaly(data_root, output_dir, seed=342, profile='balanced', contamination=0.15, threads=-1):
    if not 0 < contamination < 0.5:
        raise ValueError('contamination must be between 0 and 0.5')
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / '05_anomaly_detection' / 'test.csv')
    if not frame['player_id'].is_unique:
        raise ValueError('Task 5 player_id must be unique')
    x = frame.drop(columns='player_id').apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    if 'is_anomaly' in x:
        raise ValueError('Task 5 must remain unsupervised; found label column')
    trees = {'quick': 150, 'balanced': 400, 'strong': 800}[profile]
    scores, artifacts = [], {}
    for offset, (name, features) in enumerate((('raw', x), ('temporal', temporal_features(x)))):
        if features.shape[1] == 0:
            continue
        LOG.info('Task 5 fitting %s view: %s features', name, features.shape[1])
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        values = imputer.fit_transform(features)
        model = IsolationForest(n_estimators=trees, max_samples=min(4096, len(x)),
                                contamination='auto', random_state=seed + offset, n_jobs=threads)
        model.fit(values)
        raw = -model.score_samples(values)  # larger is more unusual
        percentile = pd.Series(raw).rank(method='average', pct=True).to_numpy()
        scores.append(percentile)
        artifacts[name] = {'columns': list(features.columns), 'imputer': imputer, 'model': model}
    ensemble = np.mean(scores, axis=0)
    # Stable ranking gives exactly ceil(n * fraction) flagged rows even when scores tie.
    order = np.argsort(-ensemble, kind='stable')
    prediction = np.zeros(len(frame), dtype=int)
    prediction[order[:int(np.ceil(contamination * len(frame)))]] = 1
    pd.DataFrame({'player_id': frame.player_id, 'prediction': prediction}).to_csv(out / 'task5_predictions.csv', index=False)
    scored = pd.DataFrame({'player_id': frame.player_id, 'anomaly_score': ensemble, 'prediction': prediction})
    for name, view in zip(artifacts, scores):
        scored[f'{name}_rank'] = view
    scored.to_csv(out / 'task5_scores.csv', index=False)
    joblib.dump(artifacts, out / 'task5_models.joblib')
    metrics = {'task': 5, 'model': 'IsolationForest raw+temporal percentile-rank ensemble',
               'seed': seed, 'profile': profile, 'trees_per_view': trees, 'n_test': len(frame),
               'contamination_assumption': contamination, 'flagged_rows': int(prediction.sum()),
               'macro_f3': None, 'metric_note': 'No ground-truth labels; local F3 cannot be calculated.',
               'fit_policy': 'Task5-specific transductive unsupervised fit on provided unlabeled test.csv. General no-test-training rule needs TA clarification.',
               'sensitivity_flagged_rows': {str(q): int(np.ceil(q*len(frame))) for q in (.05,.1,.15,.2,.3)}}
    write_json(out / 'task5_metrics.json', metrics)
    return metrics
