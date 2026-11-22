"""Task 1 feature engineering found by EDA on the labelled training rows.

Two additions, both measured on the pipeline's own folds (OOF AUC 0.8883 -> 0.8926,
macro F2 with tuned threshold 0.7775 -> 0.7807):

1. Class range rules. The synthetic generator draws each class from a fixed range, so
   some values settle the label: legit players never fall below LEGIT_FLOOR and cheaters
   never rise above CHEAT_CEILING. The count of rules a row triggers is hard for trees to
   build from single-feature splits.
2. Model-imputed copies of the account-history features. They are missing in 7-22% of
   rows and are partly recoverable from each other (account age: R2 0.71). The imputers
   are regressors on the features only and never see the label.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LEGIT_FLOOR = {'crosshair_placement': 0.82, 'reports_received': 5, 'level': 28.0}
CHEAT_CEILING = {'crosshair_placement': 0.94, 'reports_received': 19, 'account_age_days': 1000.0,
                 'friend_network_size': 269, 'level': 96.0, 'level_progression_speed': 6.26}
IMPUTED = ('account_age_days', 'friend_network_size', 'level_progression_speed', 'level', 'reports_received')


def fit_imputers(x, seed=342):
    """One regressor per IMPUTED column, trained on rows where that column is known."""
    imputers = {}
    for column in IMPUTED:
        known = x[column].notna()
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)
        imputers[column] = model.fit(x.loc[known].drop(columns=column), x.loc[known, column])
    return imputers


def add_imputed(x, imputers):
    out = x.copy()
    for column, model in imputers.items():
        estimate = pd.Series(model.predict(x.drop(columns=column)), index=x.index)
        out[f'{column}_imp'] = x[column].where(x[column].notna(), estimate)
    return out


def add_range_rules(x):
    """Distances to the class boundaries and how many certain rules a row triggers."""
    out = x.copy()
    for column, floor in LEGIT_FLOOR.items():
        out[f'{column}_above_legit_floor'] = x[column] - floor
    for column, ceiling in CHEAT_CEILING.items():
        out[f'{column}_below_cheat_ceiling'] = ceiling - x[column]
    # A missing value triggers no rule: comparisons with NaN are False.
    out['n_sure_legit_rules'] = sum((x[c] > v).astype(int) for c, v in CHEAT_CEILING.items())
    out['n_sure_cheat_rules'] = sum((x[c] < v).astype(int) for c, v in LEGIT_FLOOR.items())
    return out


def build(train_x, test_x, seed=342):
    """Return (train, test) with the Task 1 features; imputers are fit on train only."""
    missing = [c for c in set(LEGIT_FLOOR) | set(CHEAT_CEILING) | set(IMPUTED) if c not in train_x]
    if missing:
        raise ValueError(f'Task 1 features need columns {sorted(missing)}')
    imputers = fit_imputers(train_x, seed)
    return (add_range_rules(add_imputed(train_x, imputers)),
            add_range_rules(add_imputed(test_x, imputers)))
