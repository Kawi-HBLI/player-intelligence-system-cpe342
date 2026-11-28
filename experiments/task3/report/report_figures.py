"""Draw the three Task 3 figures used in the technical report (chapter 4).

Run from the repo root:
    python experiments/task3/report/report_figures.py

Inputs
- experiments/task3/model/champion/task3_oof.csv  (V9a OOF: target, prediction)
- OOF scores per version: experiments/task3/LOG.md
- Error per actual-spend bucket: baseline from model/oracle/oracle_results.md,
  V9a from model/v9_impute/v9_impute_results.md (million THB)

Outputs (experiments/task3/report/figures/)
- path.png        OOF by version
- err_bucket.png  sum |error| by actual-spend bucket, baseline vs V9a
- rtm.png         median actual vs predicted, grouped by actual and by predicted spend
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OOF = HERE.parent / "model" / "champion" / "task3_oof.csv"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9aa0a6"


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    d = pd.read_csv(OOF)
    y, p = d["target"].to_numpy(), d["prediction"].to_numpy()
    err = np.abs(y - p)
    top = np.sort(err)[::-1]
    print(f"V9a OOF score {1 / (1 + err.sum() / y.sum()):.4f}")
    print(f"top 1% rows share of error {top[: len(top) // 100].sum() / err.sum():.1%}")
    print(f"sum(pred) / sum(actual) {p.sum() / y.sum():.4f}")

    # rtm.png: regression to the mean
    bins = [-1, 0, 1000, 5000, 20000, 1e9]
    labels = ["0", "1-1k", "1k-5k", "5k-20k", ">20k"]
    by_actual = d.groupby(pd.cut(y, bins, labels=labels), observed=False).agg(
        t=("target", "median"), q=("prediction", "median"))
    by_pred = d.groupby(pd.cut(p, bins, labels=labels), observed=False).agg(
        t=("target", "median"), q=("prediction", "median"), n=("target", "size"))
    print("grouped by actual\n", by_actual.round(0))
    print("grouped by predicted\n", by_pred.round(0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, r, title in [(axes[0], by_actual, "Grouped by ACTUAL spend"),
                         (axes[1], by_pred, "Grouped by PREDICTED spend")]:
        x = np.arange(len(r))
        ax.bar(x - 0.2, r["t"].clip(lower=1), 0.4, color=BLUE, label="median actual")
        ax.bar(x + 0.2, r["q"].clip(lower=1), 0.4, color=ORANGE, label="median predicted")
        ax.set_xticks(x, r.index)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("group (THB)")
        tidy(ax)
    axes[0].set_ylabel("THB (log, 0 shown as 1)")
    axes[0].legend(frameon=False)
    fig.suptitle("Task 3 V9a OOF: looks under-predicted by actual, but calibrated by prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rtm.png", dpi=150)

    # err_bucket.png: baseline copied from oracle_results.md (its OOF is not in the repo); V9a computed from the OOF
    buckets = ["0", ">0-1k", "1k-20k", ">20k"]
    base = [91.17, 4.90, 20.51, 199.80]
    v9a = (pd.Series(err).groupby(pd.cut(y, [-1, 0, 1000, 20000, 1e9], labels=buckets), observed=False).sum()
           / 1e6).round(2).tolist()
    print(f"V9a error by bucket (M THB) {v9a}")  # v9_impute_results.md: 92.60 / 4.62 / 15.52 / 169.23
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(buckets))
    for bars in (ax.bar(x - 0.2, base, 0.4, color=GREY, label=f"baseline (total {sum(base):.1f}M)"),
                 ax.bar(x + 0.2, v9a, 0.4, color=BLUE, label=f"V9a (total {sum(v9a):.1f}M)")):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, [f"actual {k} THB" for k in buckets])
    ax.set_ylabel("sum |error| (million THB)")
    ax.set_title("Where the error is: absolute error by actual-spend bucket (OOF)")
    tidy(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "err_bucket.png", dpi=150)

    # path.png: OOF by version (LOG.md); * = honest early stopping. The line breaks between the two
    # protocols (LOG: compare only within one); V4* is V4 re-measured under honest-es (step 14).
    versions = ["baseline", "V2", "V3", "V4", "V4*", "V8*", "V9a*"]
    scores = [0.7753, 0.7849, 0.7900, 0.7918, 0.7911, 0.7921, 0.7947]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(4), scores[:4], "o-", color=BLUE, label="early stop on the scored fold")
    ax.plot(range(4, 7), scores[4:], "o-", color=ORANGE, label="honest early stopping (*)")
    ax.set_xticks(range(len(versions)), versions)
    for i, s in enumerate(scores):
        ax.text(i, s + 0.0008, f"{s:.4f}", ha="center", fontsize=8)
    ax.axhline(0.7735, ls=":", color="#888")
    ax.text(6.05, 0.7737, "baseline (honest) 0.7735", fontsize=8, ha="right")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
    ax.set_ylabel("OOF score")
    ax.set_title("Task 3 OOF by version (* = honest early stopping)")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "path.png", dpi=150)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
