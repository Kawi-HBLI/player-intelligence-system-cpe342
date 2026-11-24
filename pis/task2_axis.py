"""Task 2 axis features: per-row summaries along the two directions that separate the segments.

On train, the z-scored class means put the four segments on two axes: engagement (Grinder and
Whale above Casual and Social) and spending (Social and Whale above Casual and Grinder). Each
numeric column joins the axis it separates more, signed so that a higher value always means more
engaged or more spending. For every row the mean, min, max and number of observed values on each
axis become extra columns. A row with a few missing columns still gets an axis summary from the
columns it has, which is where the boosted trees gained (OOF macro F1 of the blend 0.7567 to
0.7573, better on three split seeds).

Everything is fitted on the training rows and labels only. Test rows are only transformed.
"""
import numpy as np
import pandas as pd

# Columns that carry no class signal on train; they stay out of both axes.
NOISE_COLUMNS = ('item_type_preference_social', 'responds_to_discounts', 'avg_match_length',
                 'preferred_game_mode', 'account_age_days', 'random_metric_1', 'random_metric_2',
                 'random_metric_3')
ENGAGED, LESS_ENGAGED = [1, 3], [0, 2]  # Grinder, Whale against Casual, Social
SPENDING, LESS_SPENDING = [2, 3], [0, 1]  # Social, Whale against Casual, Grinder


def fit_axes(x: pd.DataFrame, y, numeric: list) -> dict:
    """Column means and standard deviations, axis membership and sign, from training rows."""
    mean, std = x[numeric].mean(), x[numeric].std()
    class_means = ((x[numeric] - mean) / std).groupby(np.asarray(y)).mean()
    missing = {*ENGAGED, *LESS_ENGAGED} - set(class_means.index)
    if missing:
        raise ValueError(f'Task 2 axes need all four segments in train, missing {sorted(missing)}')
    engagement = class_means.loc[ENGAGED].mean() - class_means.loc[LESS_ENGAGED].mean()
    spending = class_means.loc[SPENDING].mean() - class_means.loc[LESS_SPENDING].mean()
    groups, signs = {'eng_axis': [], 'spend_axis': []}, {}
    for column in numeric:
        if column in NOISE_COLUMNS:
            continue
        if abs(engagement[column]) >= abs(spending[column]):
            groups['eng_axis'].append(column)
            signs[column] = np.sign(engagement[column])
        else:
            groups['spend_axis'].append(column)
            signs[column] = np.sign(spending[column])
    return {'mean': mean, 'std': std, 'groups': groups, 'signs': pd.Series(signs, dtype=float)}


def transform(frame: pd.DataFrame, axes: dict) -> pd.DataFrame:
    """Return a copy of frame with the axis columns appended."""
    z = (frame[axes['mean'].index] - axes['mean']) / axes['std']
    extra = pd.DataFrame(index=frame.index)
    for name, columns in axes['groups'].items():
        block = z[columns] * axes['signs'][columns]
        extra[f'{name}_mean'] = block.mean(axis=1)
        extra[f'{name}_min'] = block.min(axis=1)
        extra[f'{name}_max'] = block.max(axis=1)
        extra[f'{name}_present'] = block.notna().sum(axis=1)
    return pd.concat([frame, extra], axis=1)


def add_axis_features(x: pd.DataFrame, xt: pd.DataFrame, y, categorical: list):
    """Fit the axes on (x, y) and append the axis columns to both x and xt."""
    numeric = [c for c in x.columns if c not in categorical]
    axes = fit_axes(x, y, numeric)
    return transform(x, axes), transform(xt, axes)
