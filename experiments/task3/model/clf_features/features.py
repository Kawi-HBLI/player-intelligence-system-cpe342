"""Candidate stage-1 classifier features for Task 3 ("will they spend next month?").

Used by clf_features.py (the screen) and, for the groups that pass it, by the V5 variant in
../amount_model/amount_model.py. Only the classifier gets these; the amount model is untouched.

Every group is built per outer fold with the same rule as expected_spend:
    build_group(name, fit_raw, fit_y, others) -> (features for fit rows, [features for each frame in others])
  - groups without the target (timing, activity, spend_habit) use only the row's own columns,
    except the within-segment ranks, which rank a row against the FIT rows' distribution;
  - the target group (zero_rate) is computed with inner KFold(5) out-of-fold values for the
    fit rows and fitted on the whole fit part for val/test rows.
All outputs are positional DataFrames aligned to the input rows.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 342
INNER_FOLDS = 5
SMOOTHING = 50  # zero_rate: m pseudo-rows at the global rate
GROUPS = ['timing', 'activity', 'spend_habit', 'zero_rate']


def _ratio(a, b):
    """a / b with inf (b == 0) turned into NaN; CatBoost handles NaN natively."""
    return (a / b).replace([np.inf, -np.inf], np.nan)


# --- Group 1: purchase timing ---
# purchase_frequency (1–25, no zeros) is an average purchase rate; the datacard gives no unit,
# so 30 / frequency is "days between purchases" only if it is per month. The constant does not
# change any tree split, only the readability.

def timing(raw):
    out = pd.DataFrame(index=raw.index)
    # WHY: a player who buys every 3 days and one who buys every 30 days are both "regular", but
    # the expected gap tells the classifier how likely at least one purchase falls in a 30-day window.
    out['expected_interval'] = 30.0 / raw['purchase_frequency']
    # WHY: days_since_last_purchase alone mixes fast and slow buyers. Measured in units of the
    # player's own gap, ≤ 1 means "on schedule, a purchase is due", ≫ 1 means the habit has
    # stopped (churn signal), which a single-column split cannot express.
    out['overdue'] = raw['days_since_last_purchase'] / out['expected_interval']
    # WHY: the 30-day target window; ≤7 / ≤30 / ≤60 / >60 days marks "bought this week", "bought
    # within one target window", "skipped one window", "long gone". Trees can split the raw column
    # too, so this only helps if the coarse boundaries are easier to reuse in feature combinations.
    d = raw['days_since_last_purchase']
    out['recency_bucket'] = np.select([d <= 7, d <= 30, d <= 60, d > 60], [0, 1, 2, 3], np.nan)
    return out


# --- Group 2: activity ---
# A player has to be in the game to spend in it. Ranks are within segment because segment
# already sets the spend level (EDA: zero rate 73% in segment 0 vs 9% in segment 3), so "busy
# for their segment" is a different signal from "busy overall".

ACTIVITY_RANKED = ['weekly_minutes', 'daily_login_streak', 'event_participation_rate']


def activity_base(raw):
    out = pd.DataFrame(index=raw.index)
    # WHY: sessions × length = total weekly engagement; either factor alone misses the player who
    # plays rarely but long (or often but briefly). More hours in game = more exposure to the shop.
    out['weekly_minutes'] = raw['sessions_per_week'] * raw['avg_session_length']
    # Raw columns needed for the ranks (already model features, not added again).
    out['daily_login_streak'] = raw['daily_login_streak']
    out['event_participation_rate'] = raw['event_participation_rate']
    return out


def segment_rank(fit_raw, fit_vals, apply_raw, apply_vals):
    """Percentile of each apply row's value among the FIT rows of the same segment (NaN segment = own group).

    WHY: a 60-day streak is ordinary in segment 3 and exceptional in segment 0; within-segment rank
    says how engaged the player is relative to peers with the same baseline spend propensity.
    """
    fit_seg = fit_raw['segment'].fillna(-1).to_numpy()
    app_seg = apply_raw['segment'].fillna(-1).to_numpy()
    fv, av = fit_vals.to_numpy(dtype=float), apply_vals.to_numpy(dtype=float)
    out = np.full(len(av), np.nan)
    for s in np.unique(app_seg):
        ref = np.sort(fv[(fit_seg == s) & ~np.isnan(fv)])
        m = (app_seg == s) & ~np.isnan(av)
        if len(ref):
            out[m] = np.searchsorted(ref, av[m], side='right') / len(ref)
    return out


def activity(fit_raw, others):
    fit_base = activity_base(fit_raw)
    frames = []
    for raw in [fit_raw, *others]:
        base = activity_base(raw)
        out = pd.DataFrame({'weekly_minutes': base['weekly_minutes'].to_numpy()})
        for c in ACTIVITY_RANKED:
            out[f'{c}_seg_rank'] = segment_rank(fit_raw, fit_base[c], raw, base[c])
        frames.append(out)
    return frames[0], frames[1:]


# --- Group 3: spend habit ---

def spend_habit(raw):
    out = pd.DataFrame(index=raw.index)
    # WHY: a player who buys mostly on sale only spends when a sale happens; if the next 30 days
    # have no sale they spend 0. Discount share separates sale-driven buyers from regular ones.
    out['discount_share'] = _ratio(raw['purchases_on_discount'], raw['total_transactions'])
    # WHY: never bought anything = the strongest "will not spend" profile; an explicit flag lets one
    # split isolate it instead of relying on total_transactions' lowest border.
    out['no_transactions'] = (raw['total_transactions'] == 0).astype(float).where(raw['total_transactions'].notna())
    # WHY: if one typical transaction is a large part of last month's spend, last month was one or two
    # purchases (lumpy, may not repeat); if it is a small part, the player buys often in small amounts.
    out['atv_to_prev'] = _ratio(raw['avg_transaction_value'], raw['prev_month_spending'])
    # WHY: last month relative to the long-run total = momentum (rising or fading spender).
    # NOTE: near-duplicate of add_features' prev_to_hist (prev / hist); the +1 only matters for tiny
    # historical_spending (min 0.026), so it is expected to add little.
    out['prev_to_hist1'] = raw['prev_month_spending'] / (raw['historical_spending'] + 1)
    return out


# --- Group 4: group zero rate (uses the target) ---
# WHY: the share of non-spenders in the player's segment × VIP cell is the prior for "will not
# spend" (EDA: from 73% in segment 0 to 5% in segment 3, VIP 5). segment and vip_status are
# numeric columns here, so CatBoost does not build its own target statistics for them; it can
# only reach the cell rate through two splits. Smoothed toward the global rate with m = 50
# pseudo-rows so small cells (e.g. segment 0 × VIP 3) do not get noisy rates.

def _cell(raw):
    return raw['segment'].fillna(-1).astype(int).astype(str) + '_' + raw['vip_status'].fillna(-1).astype(int).astype(str)


def zero_rate_fit(fit_raw, fit_y, apply_raw):
    zero = pd.Series((fit_y == 0).astype(float), index=fit_raw.index)
    prior = zero.mean()
    stats = zero.groupby(_cell(fit_raw)).agg(['sum', 'count'])
    rate = (stats['sum'] + SMOOTHING * prior) / (stats['count'] + SMOOTHING)
    return _cell(apply_raw).map(rate).fillna(prior).to_numpy()


def zero_rate(fit_raw, fit_y, others):
    inner = np.full(len(fit_raw), np.nan)
    for itr, iva in KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED).split(fit_raw):
        inner[iva] = zero_rate_fit(fit_raw.iloc[itr], fit_y[itr], fit_raw.iloc[iva])
    return (pd.DataFrame({'group_zero_rate': inner}),
            [pd.DataFrame({'group_zero_rate': zero_rate_fit(fit_raw, fit_y, raw)}) for raw in others])


# --- Dispatcher ---

def build_group(name, fit_raw, fit_y, others):
    """(fit features, [features per frame in others]); positional, index reset."""
    if name in ('timing', 'spend_habit'):
        fn = timing if name == 'timing' else spend_habit
        return fn(fit_raw).reset_index(drop=True), [fn(raw).reset_index(drop=True) for raw in others]
    if name == 'activity':
        return activity(fit_raw, others)
    if name == 'zero_rate':
        return zero_rate(fit_raw, np.asarray(fit_y), others)
    raise ValueError(f'unknown group {name}')


def with_groups(x, extra):
    """Append group feature frames (positional) to model features x."""
    out = x.copy()
    for frame in extra:
        for c in frame.columns:
            out[c] = frame[c].to_numpy()
    return out


def build_groups(names, fit_raw, fit_y, others):
    """All named groups at once: (list of fit frames, list over others of list of frames)."""
    fit_parts, other_parts = [], [[] for _ in others]
    for g in names:
        f, o = build_group(g, fit_raw, fit_y, others)
        fit_parts.append(f)
        for i, frame in enumerate(o):
            other_parts[i].append(frame)
    return fit_parts, other_parts
