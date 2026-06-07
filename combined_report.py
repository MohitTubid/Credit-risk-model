"""
Combined study report: ONE PDF that opens with a cover page showing BOTH leaderboards
(classification + regression) side by side, followed by the detail bar-chart pages.

  COVER     - both ranking tables (classification by PR-AUC, regression by R^2)
  CLF bars  - PR-AUC and ROC-AUC mean +/- std bar charts
  CLF dist  - per-fold PR-AUC / ROC-AUC box plots  (if cv_folds_classification.csv exists)
  REG bars  - R^2 and RMSE mean +/- std bar charts
  REG dist  - per-fold R^2 / -RMSE box plots        (if cv_folds_regression.csv exists)
  FI        - permutation feature importance for BOTH winners (if the FI CSVs exist)

The box-plot and feature-importance pages are included only when their source files are
present, so the report grows as you run more of the pipeline.

Reuses the chart builders from the two leaderboard modules so there is no duplicated
plotting logic. Run AFTER both studies have produced their cv_*.csv files (and, for the
importance page, after the two feature_importance_*.py scripts).

Usage: python combined_report.py
Headless: LEADERBOARD_SHOW=0 python combined_report.py
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("TkAgg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from leaderboard_classification import load_ranked as load_clf, make_bar_fig as clf_bar
from leaderboard_regression import load_ranked as load_reg, make_bar_fig as reg_bar
from plot_cv_distributions import make_box_fig   # per-fold distribution box plots

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "model_study_report.pdf")
COVER_PNG = os.path.join(HERE, "model_study_cover.png")
FI_CLF_CSV = os.path.join(HERE, "feature_importance_classification.csv")
FI_REG_CSV = os.path.join(HERE, "feature_importance_regression.csv")
FOLDS_CLF_CSV = os.path.join(HERE, "cv_folds_classification.csv")
FOLDS_REG_CSV = os.path.join(HERE, "cv_folds_regression.csv")
SHOW = os.environ.get("LEADERBOARD_SHOW", "1") != "0"


def _draw_table(ax, df, headers, rows, title):
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            rec = df.iloc[r - 1]
            if rec["rank"] == 1:
                cell.set_facecolor("#f9e79f")
            elif rec["tie"]:
                cell.set_facecolor("#fdebd0")
            elif r % 2 == 0:
                cell.set_facecolor("#f4f6f7")
        if c == 1:
            cell.set_text_props(ha="left")


def cover_fig(clf, reg):
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Credit Risk Model Comparison Study", fontsize=18, fontweight="bold", y=0.975)
    fig.text(0.5, 0.935,
             "Classification: default_time (by-loan grouped 5-fold CV, ranked by PR-AUC)   |   "
             "Regression: lgd_time (plain 5-fold CV, ranked by R^2)\n"
             "gold = leader   orange = statistical tie with leader (overlapping +/- std)",
             ha="center", fontsize=9.5, color="#444")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.28,
                          top=0.88, bottom=0.05, left=0.06, right=0.94)

    ax1 = fig.add_subplot(gs[0])
    clf_headers = ["#", "Model", "PR-AUC", "+/-std", "ROC-AUC"]
    clf_rows = [[f"{r.rank}", r.model, f"{r.CV_PR_AUC_mean:.4f}",
                 f"{r.CV_PR_AUC_std:.4f}", f"{r.CV_ROC_AUC_mean:.4f}"] for r in clf.itertuples()]
    _draw_table(ax1, clf, clf_headers, clf_rows,
                f"CLASSIFICATION - default_time   (winner: {clf.loc[0,'model']}, "
                f"PR-AUC {clf.loc[0,'CV_PR_AUC_mean']:.4f})")

    ax2 = fig.add_subplot(gs[1])
    reg_headers = ["#", "Model", "R^2", "+/-std", "RMSE"]
    reg_rows = [[f"{r.rank}", r.model, f"{r.CV_R2_mean:.4f}",
                 f"{r.CV_R2_std:.4f}", f"{r.CV_RMSE_mean:.4f}"] for r in reg.itertuples()]
    _draw_table(ax2, reg, reg_headers, reg_rows,
                f"REGRESSION - lgd_time   (winner: {reg.loc[0,'model']}, "
                f"R^2 {reg.loc[0,'CV_R2_mean']:.4f})")
    return fig


def fi_page():
    """Page with both winners' permutation importances. Returns None if neither CSV exists."""
    specs = [("Classification winner", FI_CLF_CSV, "drop in test PR-AUC", "#5dade2"),
             ("Regression winner", FI_REG_CSV, "drop in test R^2", "#58d68d")]
    if not any(os.path.exists(c) for _, c, _, _ in specs):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, csv, xlabel, color) in zip(axes, specs):
        if os.path.exists(csv):
            d = pd.read_csv(csv).sort_values("perm_mean")
            winner = str(d["winner"].iloc[0])
            ax.barh(d["feature"], d["perm_mean"], xerr=d["perm_std"],
                    color=color, edgecolor="#34495e",
                    error_kw=dict(ecolor="#7f8c8d", capsize=3))
            ax.set_title(f"{label}: {winner}")
            ax.set_xlabel(f"Permutation importance ({xlabel})")
            ax.grid(axis="x", alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{label}:\n(run its feature_importance script)",
                    ha="center", va="center", fontsize=11, color="#888")
    fig.suptitle("Permutation Feature Importance - study winners",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def dist_figs(csv, task, cv_kind, metrics):
    """Per-fold distribution box plots for one task (empty list if its folds CSV is absent)."""
    if not os.path.exists(csv):
        return []
    df = pd.read_csv(csv)
    return [make_box_fig(df, m, lbl, task, cv_kind) for m, lbl in metrics if m in df.columns]


def main():
    clf = load_clf()    # exits if cv_classification.csv missing
    reg = load_reg()    # exits if cv_regression.csv missing

    figs = [cover_fig(clf, reg)]

    # --- Classification: bars then per-fold distributions ---
    figs.append(clf_bar(clf, "CV_PR_AUC_mean", "CV_PR_AUC_std", "PR-AUC (primary)"))
    figs.append(clf_bar(clf, "CV_ROC_AUC_mean", "CV_ROC_AUC_std", "ROC-AUC (secondary)"))
    figs += dist_figs(FOLDS_CLF_CSV, "Classification", "grouped by-loan",
                      [("PR_AUC", "PR-AUC (primary)"), ("ROC_AUC", "ROC-AUC (secondary)")])

    # --- Regression: bars then per-fold distributions ---
    figs.append(reg_bar(reg, "CV_R2_mean", "CV_R2_std", "R^2 (primary)", lower_better=False))
    figs.append(reg_bar(reg, "CV_RMSE_mean", "CV_RMSE_std", "RMSE (secondary)", lower_better=True))
    figs += dist_figs(FOLDS_REG_CSV, "Regression", "plain",
                      [("R2", "R^2 (primary)"), ("neg_RMSE", "-RMSE (secondary)")])

    # --- Feature importance (both winners) ---
    fi = fi_page()
    if fi is not None:
        figs.append(fi)

    cover = figs[0]

    with PdfPages(PDF_PATH) as pdf:
        for f in figs:
            pdf.savefig(f)
    cover.savefig(COVER_PNG, dpi=150)
    print(f"Saved {len(figs)}-page combined report: {PDF_PATH}")
    print(f"Saved cover PNG: {COVER_PNG}")

    if SHOW:
        try:
            if sys.platform.startswith("win"):
                os.startfile(PDF_PATH)             # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{PDF_PATH}"')
            else:
                os.system(f'xdg-open "{PDF_PATH}"')
        except Exception as e:
            print(f"(Could not auto-open PDF: {e})")
        plt.show()


if __name__ == "__main__":
    main()
