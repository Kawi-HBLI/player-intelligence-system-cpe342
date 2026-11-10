"""Experiment: longer CatBoost training for T1-T3 (5-fold). Writes OOF + test preds to OUT."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import fbeta_score, f1_score

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
TASKS = {1: ('01_binary_classification', 'is_cheater'), 2: ('02_multiclass_classification', 'segment'),
         3: ('03_regression', 'spending_30d')}
CFG = {1: dict(iterations=4000, learning_rate=0.03, depth=6),
       2: dict(iterations=6000, learning_rate=0.05, depth=6),
       3: dict(iterations=8000, learning_rate=0.05, depth=6)}
FOLDS, SEED = 5, 342


def load(task):
    folder, target = TASKS[task]
    tr = pd.read_csv(DATA / folder / 'train.csv'); te = pd.read_csv(DATA / folder / 'test.csv')
    tr = tr[tr[target].notna()].reset_index(drop=True)
    cols = [c for c in te.columns if c != 'player_id' and not c.startswith('random_metric')]
    x, xt = tr[cols].copy(), te[cols].copy()
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(x[c])]
    for c in cats:
        x[c] = x[c].fillna('__MISSING__').astype(str); xt[c] = xt[c].fillna('__MISSING__').astype(str)
    y = tr[target].to_numpy()
    return x, xt, y.astype(int) if task != 3 else y, cats, te.player_id


def nmae_score(y, p):
    return 1 / (1 + np.abs(y - p).sum() / y.sum())


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    s = [fbeta_score(y, (p >= t).astype(int), beta=2, average='macro') for t in grid]
    i = int(np.argmax(s)); return float(grid[i]), float(s[i])


def run(task):
    x, xt, y, cats, ids = load(task)
    n_out = 1 if task != 2 else 4
    oof = np.zeros((len(x), n_out)); tp = np.zeros((len(xt), n_out)); iters = []
    split = (StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y) if task != 3
             else KFold(FOLDS, shuffle=True, random_state=SEED).split(x))
    t0 = time.time()
    for f, (a, b) in enumerate(split, 1):
        p = dict(**CFG[task], random_seed=SEED + f, thread_count=-1, verbose=False,
                 allow_writing_files=False, early_stopping_rounds=300)
        if task == 3:
            m = CatBoostRegressor(loss_function='MAE', **p)
        else:
            m = CatBoostClassifier(loss_function='Logloss' if task == 1 else 'MultiClass', **p)
        m.fit(x.iloc[a], y[a], cat_features=cats, eval_set=(x.iloc[b], y[b]), use_best_model=True)
        iters.append(m.get_best_iteration() + 1)
        if task == 3:
            oof[b, 0] = np.maximum(0, m.predict(x.iloc[b])); tp[:, 0] += np.maximum(0, m.predict(xt)) / FOLDS
        elif task == 1:
            oof[b, 0] = m.predict_proba(x.iloc[b])[:, 1]; tp[:, 0] += m.predict_proba(xt)[:, 1] / FOLDS
        else:
            oof[b] = m.predict_proba(x.iloc[b]); tp += m.predict_proba(xt) / FOLDS
        print(f'task{task} fold{f} iter={iters[-1]} {time.time()-t0:.0f}s', flush=True)
    res = {'task': task, 'best_iterations': iters}
    if task == 3:
        res['score'] = nmae_score(y, oof[:, 0]); pred = tp[:, 0]
    elif task == 1:
        res['score_0.5'] = fbeta_score(y, (oof[:, 0] >= .5).astype(int), beta=2, average='macro')
        res['threshold'], res['score_tuned'] = best_threshold(y, oof[:, 0])
        pred = (tp[:, 0] >= res['threshold']).astype(int)
    else:
        res['score'] = f1_score(y, oof.argmax(1), average='macro'); pred = tp.argmax(1)
    np.save(OUT / f'task{task}_oof.npy', oof); np.save(OUT / f'task{task}_test.npy', tp)
    pd.DataFrame({'player_id': ids, 'prediction': pred}).to_csv(OUT / f'task{task}_predictions.csv', index=False)
    (OUT / f'task{task}_result.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


for t in [int(v) for v in sys.argv[3:]] or [3, 2, 1]:
    run(t)
