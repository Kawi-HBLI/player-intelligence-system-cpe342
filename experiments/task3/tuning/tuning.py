"""Task 3 V8: Optuna on V4's amount model (stage 2) under honest early stopping.

Only stage 2 is tuned: the CatBoost MultiQuantile model on log(y / prev) with V4's alphas.
The classifier was shown not to benefit from tuning (V3b, PR #8), so its honest-es p per
fold is computed once (honest_es.classifier_cache) and kept fixed. Every trial scores the
FULL V4 pipeline (quantile_combine, s = 1, t = 0.5, cap) under protocol honest-es on the
same 3 outer folds, so a trial's value is directly comparable with V4-honest.

Run from the repo root (long: run it in the background; it resumes after an interruption):
    python experiments/task3/tuning/tuning.py [--trials 40] [--hours 3] [--promote] [--quick]
Stages, each resumable:
  1. reproducibility check: V4's own amount settings on fold 1 must reproduce V4-honest;
  2. Optuna study in outputs/optuna/task3_amount.db (SQLite): TPESampler(seed=342),
     MedianPruner after fold 1 (n_startup_trials=8), budget `--trials` trials or `--hours`
     hours of trial time, whichever comes first (counted over all runs of the study);
  3. selection-noise guard: the top 3 trials are re-run with random_seed 1342 + fold on the
     same folds; V8 = the one with the best mean OOF of the two runs;
  4. V8 is eligible only if it beats V4-honest on all 3 folds in BOTH seed runs; with
     --promote an eligible V8 becomes the champion (protocol honest-es).
Writes the Optuna section of experiments/task3/tuning/tuning_results.md, rows in
experiments/task3/LOG.md, and (if promoted) experiments/task3/tuning/v8.json, which
refit.py --variant v8 reads. Predictions go to outputs/ (gitignored).
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from honest_es import (AMT_ITERS, OUT, OUT_DIR, PROTOCOL, ROOT, SEED, Data,  # noqa: E402
                       classifier_cache, current_branch, current_commit, fit_amount, read_champion,
                       run_pipeline, save_predictions, write_champion, write_section)
from amount_model import ALPHAS, cap_predictions, quantile_combine, upsert_log  # noqa: E402
from oracle import BUCKETS, bucket_table, to_markdown  # noqa: E402
from pis.tabular import _score  # noqa: E402

# --- Settings ---
N_TRIALS, HOURS = 40, 3.0
MAX_ITERS = 5000  # iterations limit of every trial; honest early stopping (80 rounds) picks the count
RERUN_SEED = 1342  # second seed scheme: random_seed = 1342 + fold
TOP_K = 3
STUDY = 'task3_amount_v4_honest'
OPTUNA_DIR = ROOT / 'outputs' / 'optuna'
V8_JSON = Path(__file__).with_name('v8.json')
V8_DIR = OUT / 'task3_v8'
PARAMS = ['depth', 'learning_rate', 'l2_leaf_reg', 'random_strength', 'bagging_temperature', 'border_count',
          'grow_policy', 'min_data_in_leaf']
# V4's own amount settings written in the search space's terms. bagging_temperature only acts
# with Bayesian bootstrap; CatBoost's CPU default is MVS (subsample 0.8), which ignores it, so
# every trial sets bootstrap_type='Bayesian'. V4 itself (MVS) is therefore not a point of the
# space; trial 0 is its nearest point (V4's settings, Bayesian bootstrap at its default
# temperature 1), enqueued so the study starts from V4.
V4_POINT = dict(depth=6, learning_rate=0.05, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
                border_count=254, grow_policy='SymmetricTree')
LABELS = {'optuna': f'Optuna: V4 amount model (CatBoost MultiQuantile) tuned under honest-es, TPE seed {SEED}, '
                    f'MedianPruner after fold 1, budget {N_TRIALS} trials / {HOURS:g} h',
          'v8': 'V8: V4 + Optuna-tuned amount model (best mean of 2 seed runs among the top 3 trials), seed 342 + fold',
          'v8_rerun': 'V8 re-run: same model, seed 1342 + fold (selection-noise check)'}


def suggest(trial):
    """Search space; returns CatBoost overrides on top of cb_params (V4's settings)."""
    p = dict(depth=trial.suggest_int('depth', 4, 10),
             learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
             l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
             random_strength=trial.suggest_float('random_strength', 0.0, 5.0),
             bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 2.0),
             border_count=trial.suggest_categorical('border_count', [64, 128, 254]),
             grow_policy=trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']))
    if p['grow_policy'] == 'Depthwise':
        p['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 200)
    return to_catboost(p)


def to_catboost(params):
    return {**params, 'bootstrap_type': 'Bayesian'}


# --- Run storage: one .npz per (trial, seed scheme), so every stage can resume ---

def save_run(path, res):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, oof=res['oof'], test=res['test'], folds=res['folds'], best=res['best'], seconds=res['seconds'])


def load_run(path, y):
    z = np.load(path)
    return dict(oof=z['oof'], test=z['test'], folds=z['folds'].tolist(), score=_score(y, z['oof']),
                best=z['best'].tolist(), seconds=z['seconds'].tolist())


def cached_run(path, d, compute):
    if path.exists():
        return load_run(path, d.y)
    res = compute()
    save_run(path, res)
    return res


def fmt4(v):
    return f'{v:.4f}'


def per_fold(a, b, digits=5):
    return ' / '.join(f'{x - y:+.{digits}f}' for x, y in zip(a, b))


def describe(params):
    """Short text of a trial's params (for tables and the LOG)."""
    p = dict(params)
    out = [f'depth {p["depth"]}', f'lr {p["learning_rate"]:.4f}', f'l2 {p["l2_leaf_reg"]:.2f}',
           f'rs {p["random_strength"]:.2f}', f'bt {p["bagging_temperature"]:.2f}', f'border {p["border_count"]}',
           p['grow_policy']]
    if 'min_data_in_leaf' in p:
        out.append(f'min_leaf {p["min_data_in_leaf"]}')
    return ', '.join(out)


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # Windows consoles (cp874) cannot print Δ / ×
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--hours', type=float, default=HOURS)
    parser.add_argument('--promote', action='store_true', help='promote an eligible V8 (protocol honest-es)')
    parser.add_argument('--quick', action='store_true',
                        help='smoke test: 250 iterations, separate study/outputs, nothing tracked is written')
    args = parser.parse_args()
    if args.promote and args.quick:
        parser.error('--promote needs a full run, not --quick')
    t_run = time.time()
    max_iters = 250 if args.quick else MAX_ITERS
    work = OPTUNA_DIR / 'quick' if args.quick else OPTUNA_DIR
    work.mkdir(parents=True, exist_ok=True)

    d = Data()
    clf = classifier_cache(d, args.quick)
    ref_path = (OUT_DIR / 'quick' if args.quick else OUT_DIR) / 'task3_oof.csv'
    if args.quick:
        ref = run_pipeline(d, clf, iterations=250)
    else:
        if not ref_path.exists():
            sys.exit(f'{ref_path.relative_to(ROOT)} not found; run experiments/task3/tuning/honest_es.py first')
        oof = pd.read_csv(ref_path)['prediction'].to_numpy(dtype=float)
        ref = dict(oof=oof, folds=d.fold_scores(oof), score=_score(d.y, oof))
    champ = read_champion()
    if champ.get('protocol') != PROTOCOL:
        sys.exit(f'champion.json is on protocol {champ.get("protocol", "val-es")}; run honest_es.py --record first')
    if not args.quick and champ['variant'] == 'v4_honest' and abs(ref['score'] - champ['oof']) > 1e-5:
        sys.exit('outputs V4-honest OOF does not match champion.json; re-run honest_es.py')
    print(f'V4-honest reference: {" / ".join(map(fmt4, ref["folds"]))}, OOF {ref["score"]:.4f}', flush=True)

    # --- Study (SQLite: resumes after an interruption) ---
    storage = f'sqlite:///{(work / "task3_amount.db").as_posix()}'
    study = optuna.create_study(study_name=STUDY, storage=storage, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # A sampler's RNG is not stored in the DB: a resumed run would restart TPESampler(seed=342)
    # and replay the same random start-up draws (duplicate trials). So the first run uses seed
    # 342 and a resumed run reseeds with 342 + the number of trials already in the study.
    if study.trials:
        study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials))

    # 1. Reproducibility: V4's own settings (MVS bootstrap, 3000 limit) on fold 1 must give
    # V4-honest's fold-1 score, otherwise trials would not be comparable with it.
    if 'repro_fold1' not in study.user_attrs:
        fi = d.inputs()[0]
        q_va, _, b, s = fit_amount(d, fi, iterations=250 if args.quick else AMT_ITERS)
        got = _score(d.y[fi['va']], cap_predictions(quantile_combine(clf['p_oof'][fi['va']], q_va, fi['prev_va']),
                                                    d.y[fi['tr']]))
        print(f'reproducibility: V4 settings fold 1 {got:.6f} vs V4-honest {ref["folds"][0]:.6f} ({s:.0f}s)', flush=True)
        if abs(got - ref['folds'][0]) > 1e-6:
            sys.exit('V4-honest fold 1 does not reproduce; trials would not be comparable')
        study.set_user_attr('repro_fold1', got)

    # Interrupted trials stay RUNNING in the DB: mark them failed and queue their params again.
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
        study.enqueue_trial(t.params, skip_if_exists=False)
        print(f'trial {t.number} was interrupted: marked FAIL, params queued again', flush=True)
    if not study.trials:
        study.enqueue_trial(V4_POINT)

    # 2. Budget over all runs of the study: finished (complete + pruned) trials and their time.
    finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done = lambda: [t for t in study.get_trials(deepcopy=False, states=finished_states)]  # noqa: E731
    used = lambda: sum((t.datetime_complete - t.datetime_start).total_seconds() for t in done())  # noqa: E731
    remaining_n, remaining_s = args.trials - len(done()), args.hours * 3600 - used()
    print(f'study: {len(done())} finished trials, {used() / 60:.0f} min used; '
          f'running up to {max(remaining_n, 0)} more within {max(remaining_s, 0) / 60:.0f} min', flush=True)

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()

        def on_fold(fold, score):
            trial.set_user_attr(f'fold{fold}', score)
            if fold == 1:
                trial.report(score, step=1)
                if trial.should_prune():
                    print(f'  trial {trial.number}: fold 1 {score:.4f} -> pruned ({time.time() - t0:.0f}s)', flush=True)
                    raise optuna.TrialPruned()

        res = run_pipeline(d, clf, params, SEED, max_iters, on_fold)
        trial.set_user_attr('folds', res['folds'])
        trial.set_user_attr('best_iterations', res['best'])
        trial.set_user_attr('fit_seconds', res['seconds'])
        save_run(work / 'trials' / f'trial_{trial.number}.npz', res)
        return res['score']

    def progress(study_, trial):
        n_done = len(done())
        value = f'{trial.value:.4f}' if trial.value is not None else '—'
        folds = ' / '.join(fmt4(trial.user_attrs[f'fold{i}']) for i in (1, 2, 3) if f'fold{i}' in trial.user_attrs)
        best = study_.best_trial if any(t.state == optuna.trial.TrialState.COMPLETE for t in study_.trials) else None
        print(f'[{n_done}/{args.trials}] trial {trial.number} {trial.state.name} OOF {value} ({folds}) '
              f'{(trial.datetime_complete - trial.datetime_start).total_seconds():.0f}s | {describe(trial.params)} | '
              f'best #{best.number if best else "—"} {best.value if best else float("nan"):.4f} | '
              f'budget used {used() / 60:.0f} min', flush=True)

    if remaining_n > 0 and remaining_s > 0:
        study.optimize(objective, n_trials=remaining_n, timeout=remaining_s, callbacks=[progress],
                       catch=(CatBoostError,), gc_after_trial=True)
    complete = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                      key=lambda t: t.value, reverse=True)
    counts = {s.name: sum(t.state == s for t in study.trials) for s in optuna.trial.TrialState}
    if len(complete) < TOP_K:
        sys.exit(f'only {len(complete)} complete trials; need {TOP_K}')

    # 3. Top 3 trials re-run with random_seed 1342 + fold (same folds, same fixed classifier p).
    t_rerun = time.time()
    top = complete[:TOP_K]
    runs = {}
    for t in top:
        params = to_catboost(t.params)
        first = cached_run(work / 'trials' / f'trial_{t.number}.npz', d,
                           lambda: run_pipeline(d, clf, params, SEED, max_iters))
        second = cached_run(work / 'rerun' / f'trial_{t.number}_seed{RERUN_SEED}.npz', d,
                            lambda: run_pipeline(d, clf, params, RERUN_SEED, max_iters))
        runs[t.number] = (first, second)
        print(f'top trial {t.number}: seed {SEED}+fold {first["score"]:.4f}, seed {RERUN_SEED}+fold '
              f'{second["score"]:.4f}, mean {(first["score"] + second["score"]) / 2:.4f}', flush=True)
    # Seed-noise reference (not part of the decision): V4-honest itself with the second seed scheme.
    v4_rerun = cached_run(work / 'rerun' / f'v4_seed{RERUN_SEED}.npz', d,
                          lambda: run_pipeline(d, clf, None, RERUN_SEED, 250 if args.quick else AMT_ITERS))
    rerun_seconds = time.time() - t_rerun

    # 4. V8 = best mean of the two runs; eligible only if both runs beat V4-honest on every fold.
    v8_trial = max(top, key=lambda t: runs[t.number][0]['score'] + runs[t.number][1]['score'])
    v8, v8b = runs[v8_trial.number]
    wins = lambda r: all(a > b for a, b in zip(r['folds'], ref['folds']))  # noqa: E731
    eligible = wins(v8) and wins(v8b)

    # --- Tables ---
    best10 = pd.DataFrame([[t.number, fmt4(t.value), *[fmt4(f) for f in t.user_attrs['folds']],
                            per_fold(t.user_attrs['folds'], ref['folds'], 4), t.params['depth'],
                            f'{t.params["learning_rate"]:.4f}', f'{t.params["l2_leaf_reg"]:.2f}',
                            f'{t.params["random_strength"]:.2f}', f'{t.params["bagging_temperature"]:.2f}',
                            t.params['border_count'], t.params['grow_policy'], t.params.get('min_data_in_leaf', '—'),
                            ' / '.join(map(str, t.user_attrs['best_iterations'])),
                            f'{(t.datetime_complete - t.datetime_start).total_seconds() / 60:.1f}']
                           for t in complete[:10]],
                          columns=['trial', 'OOF', 'fold1', 'fold2', 'fold3', 'Δ per fold vs V4-honest', 'depth', 'lr',
                                   'l2_leaf_reg', 'random_strength', 'bagging_temp', 'border_count', 'grow_policy',
                                   'min_data_in_leaf', 'best iterations', 'minutes'])
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=SEED)
        imp = optuna.importance.get_param_importances(study, evaluator=evaluator)
        importance = pd.DataFrame({'parameter': list(imp), 'importance (fANOVA)': [f'{v:.3f}' for v in imp.values()]})
    except (RuntimeError, ValueError) as error:  # e.g. too few complete trials
        importance = pd.DataFrame({'parameter': ['—'], 'importance (fANOVA)': [f'not computed: {error}']})
    rerun_rows = []
    for t in top:
        a, b = runs[t.number]
        rerun_rows.append([t.number, *map(fmt4, a['folds']), fmt4(a['score']), *map(fmt4, b['folds']), fmt4(b['score']),
                           fmt4((a['score'] + b['score']) / 2), 'yes' if wins(a) else 'no', 'yes' if wins(b) else 'no'])
    rerun_rows.append(['V4-honest (reference)', *map(fmt4, ref['folds']), fmt4(ref['score']),
                       *map(fmt4, v4_rerun['folds']), fmt4(v4_rerun['score']),
                       fmt4((ref['score'] + v4_rerun['score']) / 2), '—', '—'])
    rerun = pd.DataFrame(rerun_rows, columns=['trial', 'fold1 (342)', 'fold2 (342)', 'fold3 (342)', 'OOF (342)',
                                              'fold1 (1342)', 'fold2 (1342)', 'fold3 (1342)', 'OOF (1342)',
                                              'mean OOF', 'beats V4-honest all folds (342)',
                                              'beats V4-honest all folds (1342)'])
    base_h = pd.read_csv(OUT / 'task3_baseline_honest' / 'task3_oof.csv')['prediction'].to_numpy(dtype=float) \
        if (OUT / 'task3_baseline_honest' / 'task3_oof.csv').exists() else None
    summary_rows = []
    for label, r in [('Baseline-honest', None if base_h is None else dict(folds=d.fold_scores(base_h),
                                                                          score=_score(d.y, base_h))),
                     ('V4-honest (champion reference)', ref), (f'V8 = trial {v8_trial.number}, seed 342 + fold', v8),
                     (f'V8 = trial {v8_trial.number}, seed 1342 + fold', v8b),
                     ('V4-honest, seed 1342 + fold (noise reference)', v4_rerun)]:
        if r is None:
            continue
        summary_rows.append([label, *map(fmt4, r['folds']), fmt4(r['score']), f'{r["score"] - ref["score"]:+.4f}',
                             per_fold(r['folds'], ref['folds']), 'yes' if wins(r) else 'no'])
    summary = pd.DataFrame(summary_rows, columns=['model', 'fold1', 'fold2', 'fold3', 'OOF', 'Δ vs V4-honest',
                                                  'Δ per fold vs V4-honest', 'wins all folds vs V4-honest?'])
    side = pd.DataFrame({'bucket': [*BUCKETS, 'total'],
                         'count': [*bucket_table(d.y, ref['oof'])['count'].map('{:,}'.format), f'{len(d.y):,}']})
    for label, pred in [('baseline-honest', base_h), ('V4-honest', ref['oof']), ('V8 (342)', v8['oof']),
                        ('V8 (1342)', v8b['oof'])]:
        if pred is not None:
            e = bucket_table(d.y, pred)['error_sum']
            side[f'{label} error (M THB)'] = [f'{v / 1e6:.2f}' for v in [*e, e.sum()]]
    e_ref, e_v8 = bucket_table(d.y, ref['oof'])['error_sum'], bucket_table(d.y, v8['oof'])['error_sum']
    side['Δ V8 (342) − V4-honest'] = [f'{(b - a) / 1e6:+.2f}' for a, b in zip([*e_ref, e_ref.sum()], [*e_v8, e_v8.sum()])]
    trial_minutes = [(t.datetime_complete - t.datetime_start).total_seconds() / 60 for t in done()]
    runtime = pd.DataFrame([['finished trials (complete + pruned)', f'{len(done())}'],
                            ['complete / pruned / failed', f'{counts["COMPLETE"]} / {counts["PRUNED"]} / {counts["FAIL"]}'],
                            ['trial time, total (all runs of the study)', f'{used() / 3600:.2f} h'],
                            ['trial time, mean / max', f'{np.mean(trial_minutes):.1f} / {np.max(trial_minutes):.1f} min'],
                            ['top-3 re-runs + V4 seed reference (this run, 0 if cached)', f'{rerun_seconds / 60:.1f} min'],
                            ['this invocation, wall time', f'{(time.time() - t_run) / 60:.1f} min']],
                           columns=['measure', 'value'])

    if eligible:
        decision = (f'V8 = trial {v8_trial.number} beats V4-honest on all 3 folds in both seed runs '
                    f'({per_fold(v8["folds"], ref["folds"])} with seed 342; {per_fold(v8b["folds"], ref["folds"])} '
                    f'with seed 1342): eligible.')
    else:
        decision = (f'STOP: V8 = trial {v8_trial.number} does not beat V4-honest on all 3 folds in both seed runs '
                    f'(seed 342: {per_fold(v8["folds"], ref["folds"])}; seed 1342: {per_fold(v8b["folds"], ref["folds"])}); '
                    'champion stays V4-honest, no refit, no Kaggle file.')
    print(f'\nTop 10 trials\n{best10.to_string(index=False)}\n\nParameter importances\n{importance.to_string(index=False)}'
          f'\n\nTop-3 re-run\n{rerun.to_string(index=False)}\n\nSummary\n{summary.to_string(index=False)}'
          f'\n\nError by bucket\n{side.to_string(index=False)}\n\nRuntimes\n{runtime.to_string(index=False)}\n\n{decision}')
    if args.quick:
        print('quick run: nothing tracked written')
        return

    # --- Promotion (protocol honest-es) ---
    save_predictions(d, V8_DIR, v8)
    promoted = False
    if eligible and args.promote:
        if champ['variant'] == 'v8':
            print('champion is already V8; not re-promoted')
        else:
            meta = {'name': f'V8: V4 + Optuna-tuned amount model (trial {v8_trial.number})', 'variant': 'v8',
                    'protocol': PROTOCOL, 'protocol_note': champ['protocol_note'],
                    'script': 'experiments/task3/tuning/tuning.py', 'branch': current_branch(), 'commit': current_commit(),
                    'commit_note': 'HEAD when the run started; the promotion itself lands in the next commit',
                    'fold_scores': [round(s, 6) for s in v8['folds']], 'oof': round(v8['score'], 6),
                    'fold_scores_seed1342': [round(s, 6) for s in v8b['folds']], 'oof_seed1342': round(v8b['score'], 6),
                    'date': datetime.date.today().isoformat(), 'previous_champion': champ['name'],
                    'previous_champion_fold_scores': champ['fold_scores'], 'previous_champion_oof': champ['oof'],
                    'rule': champ['rule'] + '; V8 also had to win every fold with a second seed scheme (1342 + fold)',
                    'previous_protocol': champ.get('previous_protocol'), 'lb_history': champ.get('lb_history', []),
                    **({'submission': champ['submission']} if 'submission' in champ else {})}
            oof_frame = pd.DataFrame({'player_id': d.train['player_id'], 'target': d.y, 'prediction': v8['oof']})
            write_champion(meta, oof_frame)
            promoted = True
        V8_JSON.write_text(json.dumps({
            'trial': v8_trial.number, 'study': STUDY, 'protocol': PROTOCOL,
            'amount_params': to_catboost(v8_trial.params), 'amount_iterations_limit': MAX_ITERS,
            'amount_best_iterations': v8['best'], 'amount_best_iterations_seed1342': v8b['best'],
            'classifier_best_iterations': clf['best'], 'fold_scores': [round(s, 6) for s in v8['folds']],
            'oof': round(v8['score'], 6), 'fold_scores_seed1342': [round(s, 6) for s in v8b['folds']],
            'oof_seed1342': round(v8b['score'], 6), 'date': datetime.date.today().isoformat()},
            indent=2) + '\n', encoding='utf-8')
        print(f'PROMOTED: {"V8" if promoted else "V8 (already)"} is the champion; wrote {V8_JSON.relative_to(ROOT)}')
    elif eligible:
        print('V8 is eligible; re-run with --promote to promote it.')

    # --- tuning_results.md (Optuna section) and LOG.md ---
    best = complete[0]
    write_section('optuna', [
        '## Part 2: Optuna on the amount model (V8)', '',
        f'Generated by `python experiments/task3/tuning/tuning.py{" --promote" if args.promote else ""}` '
        f'(budget {args.trials} trials or {args.hours:g} h of trial time). Tuned: stage 2 only (CatBoost `MultiQuantile` '
        f'on `log(y / prev)`, alphas {ALPHAS[0]:.2f}–{ALPHAS[-1]:.2f}); the honest-es classifier p of each fold is computed '
        'once and kept fixed. Objective: overall OOF score of the full V4 pipeline (quantile combine, s = 1, t = 0.5, cap) '
        f'under honest-es on the same 3 folds. Iterations up to {MAX_ITERS} with honest early stopping (80 rounds). '
        f'`TPESampler(seed={SEED})`, `MedianPruner(n_startup_trials=8)` on the fold-1 score. Every trial uses '
        '`bootstrap_type=Bayesian`, because `bagging_temperature` has no effect under CatBoost\'s CPU default (MVS); '
        'trial 0 is V4\'s settings with Bayesian bootstrap (temperature 1). `min_data_in_leaf` only exists for Depthwise. '
        'Reproducibility check: V4\'s own settings on fold 1 reproduce V4-honest '
        f'({study.user_attrs["repro_fold1"]:.6f}).', '',
        '> **CAUTION:** The best trial\'s OOF is the maximum over many trials on the same folds, so it is optimistic; '
        'that is why the top 3 are re-run with a second seed scheme and V8 must win every fold in both runs.', '',
        f'### Best 10 trials (of {counts["COMPLETE"]} complete; {counts["PRUNED"]} pruned, {counts["FAIL"]} failed)', '',
        to_markdown(best10), '',
        '### Parameter importances', '',
        'fANOVA (`optuna.importance`, seed 342) over complete trials; `min_data_in_leaf` is left out because it only '
        'exists in Depthwise trials.', '', to_markdown(importance), '',
        f'### Top-3 re-run (random_seed {SEED} + fold vs {RERUN_SEED} + fold, same folds)', '', to_markdown(rerun), '',
        '### Summary vs V4-honest', '', to_markdown(summary), '',
        '### Error by bucket (OOF)', '', to_markdown(side), '',
        '### Runtimes', '', to_markdown(runtime), '',
        '### Decision', '', decision + (' **Promoted to champion (protocol honest-es).**' if promoted else ''), ''])
    today, branch = datetime.date.today().isoformat(), current_branch()
    upsert_log([
        [today, branch, LABELS['optuna'], ' / '.join(fmt4(f) for f in best.user_attrs['folds']), fmt4(best.value), '—',
         f'{len(done())} finished trials ({counts["COMPLETE"]} complete, {counts["PRUNED"]} pruned, {counts["FAIL"]} failed), '
         f'{used() / 3600:.2f} h of trial time; best trial #{best.number} OOF {best.value:.4f} '
         f'({best.value - ref["score"]:+.4f} vs V4-honest; max over trials, optimistic): {describe(best.params)}; '
         'tunes stage 2 only, classifier p fixed; `tuning/`, outputs/optuna/task3_amount.db'],
        [today, branch, LABELS['v8'], ' / '.join(map(fmt4, v8['folds'])), fmt4(v8['score']), '—',
         f'trial #{v8_trial.number}: {describe(v8_trial.params)}; vs champion (V4-honest): '
         f'{v8["score"] - ref["score"]:+.4f} ({per_fold(v8["folds"], ref["folds"])}), wins all folds: '
         f'{"yes" if wins(v8) else "no"}; best iterations {"/".join(map(str, v8["best"]))}; '
         + ('**PROMOTED to champion** (protocol honest-es)' if promoted else decision)],
        [today, branch, LABELS['v8_rerun'], ' / '.join(map(fmt4, v8b['folds'])), fmt4(v8b['score']), '—',
         f'vs V4-honest: {v8b["score"] - ref["score"]:+.4f} ({per_fold(v8b["folds"], ref["folds"])}), wins all folds: '
         f'{"yes" if wins(v8b) else "no"}; mean of the 2 seed runs {(v8["score"] + v8b["score"]) / 2:.4f}; '
         f'V4-honest with the same seed scheme: {v4_rerun["score"]:.4f} ({v4_rerun["score"] - ref["score"]:+.4f}, seed noise)']])
    print(f'Wrote the Optuna section of tuning_results.md; updated experiments/task3/LOG.md')


if __name__ == '__main__':
    main()
