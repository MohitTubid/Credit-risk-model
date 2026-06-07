"""
Box-plot the DISTRIBUTION of per-fold CV scores across all models -- so you compare
not just the mean (the leaderboard) but the spread/stability across the identical
grouped CV folds.

This is the adaptation of the classic

    for name, clf in clfs:
        results[name] = cross_val_score(clf, X, y, cv=kf, scoring=scorer)
    sns.boxplot(pd.DataFrame(results))

pattern to THIS pipeline: instead of re-instantiating untuned models and re-running
CV (which would re-tune all 9 models, ~15 min), it reads the per-fold scores that
run_study already produced with the TUNED models on the SAME grouped folds.

Covers both tasks via the TASKS table below:
  * Classification -> cv_folds_classification.csv  (PR-AUC, ROC-AUC)
  * Regression     -> cv_folds_regression.csv      (R^2, -RMSE)  [auto-skipped until
                                                                   a regression harness
                                                                   writes that file]

Multi-page PDF (one metric per page) + a PNG per page, popped open in the viewer.

Run AFTER run_study_classification.py / run_study_regression.py.
Usage:  python plot_cv_distributions.py
Headless:  LEADERBOARD_SHOW=0 python plot_cv_distributions.py
"""
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = os.environ.get("LEADERBOARD_SHOW", "1") != "0"

# (task label, per-fold CSV, CV description, [(metric column, axis label), ...])
TASKS = [
    ("Classification", os.path.join(HERE, "cv_folds_classification.csv"), "grouped by-loan",
        [("PR_AUC", "PR-AUC (primary)"), ("ROC_AUC", "ROC-AUC (secondary)")]),
    ("Regression", os.path.join(HERE, "cv_folds_regression.csv"), "plain",
        [("R2", "R^2 (primary)"), ("neg_RMSE", "-RMSE (secondary)")]),
]


def _order_by_median(df, metric):
    """Order models best -> worst by median fold score (descending)."""
    return (df.groupby("model")[metric].median()
              .sort_values(ascending=False).index.tolist())


def make_box_fig(df, metric, label, task, cv_kind):
    order = _order_by_median(df, metric)
    n_folds = df["fold"].nunique()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df, x="model", y=metric, order=order, ax=ax,
                width=0.6, fliersize=0,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=6))
    # Overlay the actual fold points so you can see all n_folds values.
    sns.stripplot(data=df, x="model", y=metric, order=order, ax=ax,
                  color="#2c3e50", size=5, alpha=0.7, jitter=0.12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"CV {label}", fontsize=12)
    ax.set_title(f"{task} - per-fold CV {label} "
                 f"(identical {cv_kind} {n_folds}-fold CV; white diamond = mean)",
                 fontsize=13)
    ax.tick_params(axis="x", rotation=30)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    produced = False
    for task, csv, cv_kind, metrics in TASKS:
        if not os.path.exists(csv):
            print(f"[skip] {task}: {os.path.basename(csv)} not found.")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            print(f"[skip] {task}: {os.path.basename(csv)} is empty.")
            continue

        figs = [(make_box_fig(df, m, lbl, task, cv_kind), m)
                for m, lbl in metrics if m in df.columns]
        if not figs:
            print(f"[skip] {task}: none of {[m for m, _ in metrics]} in {os.path.basename(csv)}.")
            continue

        pdf_path = os.path.join(HERE, f"cv_distributions_{task.lower()}.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig, _ in figs:
                pdf.savefig(fig)
        for fig, metric in figs:
            fig.savefig(os.path.join(HERE, f"cv_dist_{task.lower()}_{metric}.png"), dpi=150)
        print(f"{task}: saved {len(figs)}-page PDF -> {pdf_path}")
        produced = True

        if SHOW:
            try:
                if sys.platform.startswith("win"):
                    os.startfile(pdf_path)              # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    os.system(f'open "{pdf_path}"')
                else:
                    os.system(f'xdg-open "{pdf_path}"')
            except Exception as e:
                print(f"(could not open {pdf_path}: {e})")

    if not produced:
        sys.exit("No per-fold CSVs found. Run run_study_classification.py "
                 "and/or run_study_regression.py first.")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
