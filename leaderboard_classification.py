"""
Re-read cv_classification.csv and render the ranked leaderboard VISUALLY as a
MULTI-PAGE PDF (one plot per page):

    page 1 - ranking table
    page 2 - PR-AUC bar chart   (primary metric)
    page 3 - ROC-AUC bar chart  (secondary metric)

Also writes a PNG per page and pops the PDF open in the system viewer.
Run AFTER run_study_classification.py (or any model module) has populated cv_classification.csv.

Usage:
    python leaderboard_classification.py                 # render + pop open the PDF + show windows
    LEADERBOARD_SHOW=0 python leaderboard_classification.py   # headless: just write the files

Dependencies: pandas + matplotlib only (does NOT import sklearn).
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("TkAgg", force=False)   # interactive backend for pop-up windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
CV_CSV = os.path.join(HERE, "cv_classification.csv")
PDF_PATH = os.path.join(HERE, "leaderboard.pdf")
PNG_TABLE = os.path.join(HERE, "leaderboard_table.png")
PNG_PR = os.path.join(HERE, "leaderboard_pr_auc.png")
PNG_ROC = os.path.join(HERE, "leaderboard_roc_auc.png")

SHOW = os.environ.get("LEADERBOARD_SHOW", "1") != "0"
SUBTITLE = ("Ranked by grouped-CV PR-AUC (by-loan split)  -  "
            "gold = leader, orange = statistical tie with leader")


def load_ranked():
    if not os.path.exists(CV_CSV):
        sys.exit(f"No results file found at {CV_CSV}\nRun run_study_classification.py first.")
    df = pd.read_csv(CV_CSV)
    if df.empty:
        sys.exit("cv_classification.csv is empty - run the study first.")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    else:
        df = df.drop_duplicates("model", keep="last")
    df = df.sort_values("CV_PR_AUC_mean", ascending=False).reset_index(drop=True)

    lead_lo = df.loc[0, "CV_PR_AUC_mean"] - df.loc[0, "CV_PR_AUC_std"]
    df["tie"] = (df["CV_PR_AUC_mean"] + df["CV_PR_AUC_std"]) >= lead_lo
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _row_color(rec, even):
    if rec["rank"] == 1:
        return "#f9e79f"          # gold leader
    if rec["tie"]:
        return "#fdebd0"          # orange tie
    return "#f4f6f7" if even else "white"


# ---------------- PAGE 1: ranking table ----------------
def make_table_fig(df):
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(df) + 2.2))
    ax.axis("off")
    fig.suptitle("Default Prediction - Model Comparison Leaderboard",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90, SUBTITLE, ha="center", fontsize=9.5, color="#444")

    headers = ["#", "Model", "PR-AUC", "+/-std", "ROC-AUC", "+/-std"]
    cell_text = [[f"{r.rank}", r.model,
                  f"{r.CV_PR_AUC_mean:.4f}", f"{r.CV_PR_AUC_std:.4f}",
                  f"{r.CV_ROC_AUC_mean:.4f}", f"{r.CV_ROC_AUC_std:.4f}"]
                 for r in df.itertuples()]
    table = ax.table(cellText=cell_text, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(_row_color(df.iloc[row - 1], row % 2 == 0))
        if col == 1:
            cell.set_text_props(ha="left")
    return fig


# ---------------- PAGE 2 & 3: bar charts ----------------
def make_bar_fig(df, mean_col, std_col, title):
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(df) + 2.5))
    d = df.sort_values(mean_col, ascending=True)     # best on top after barh
    colors = ["#f1c40f" if r["rank"] == 1 else "#e67e22" if r["tie"] else "#5dade2"
              for _, r in d.iterrows()]
    bars = ax.barh(d["model"], d[mean_col], xerr=d[std_col],
                   color=colors, edgecolor="#34495e",
                   error_kw=dict(ecolor="#7f8c8d", capsize=3))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel(title, fontsize=11)
    ax.set_xlim(0, max(0.001, (df[mean_col] + df[std_col]).max() * 1.22))
    ax.margins(y=0.02)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Default Prediction - {title}", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.93, SUBTITLE, ha="center", fontsize=9, color="#444")
    # Wide left margin so long model names are never clipped.
    fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.12)
    return fig


def main():
    df = load_ranked()
    print("Leaderboard (grouped-CV PR-AUC):")
    print(df[["rank", "model", "CV_PR_AUC_mean", "CV_PR_AUC_std", "tie"]]
          .to_string(index=False))

    figs = [
        (make_table_fig(df), PNG_TABLE),
        (make_bar_fig(df, "CV_PR_AUC_mean", "CV_PR_AUC_std", "PR-AUC (primary)"), PNG_PR),
        (make_bar_fig(df, "CV_ROC_AUC_mean", "CV_ROC_AUC_std", "ROC-AUC (secondary)"), PNG_ROC),
    ]

    # Multi-page PDF: one plot per page.
    with PdfPages(PDF_PATH) as pdf:
        for fig, _ in figs:
            pdf.savefig(fig)
    for fig, png in figs:
        fig.savefig(png, dpi=150)

    print(f"\nSaved {len(figs)}-page PDF: {PDF_PATH}")
    print(f"Saved PNGs: {PNG_TABLE}, {PNG_PR}, {PNG_ROC}")

    if SHOW:
        try:
            if sys.platform.startswith("win"):
                os.startfile(PDF_PATH)              # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{PDF_PATH}"')
            else:
                os.system(f'xdg-open "{PDF_PATH}"')
        except Exception as e:
            print(f"(Could not auto-open PDF: {e})")
        plt.show()


if __name__ == "__main__":
    main()
