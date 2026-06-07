"""
Regression leaderboard: re-read cv_regression.csv and render a MULTI-PAGE PDF
(one plot per page):

    page 1 - ranking table (R^2 and RMSE, mean +/- std)
    page 2 - R^2 bar chart    (primary metric; HIGHER is better)
    page 3 - RMSE bar chart   (secondary metric; LOWER is better)

Also writes a PNG per page and pops the PDF open in the system viewer.
Run AFTER run_study_regression.py.

Usage:
    python leaderboard_regression.py
    LEADERBOARD_SHOW=0 python leaderboard_regression.py   # headless

Dependencies: pandas + matplotlib only (does NOT import sklearn).
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("TkAgg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
CV_CSV = os.path.join(HERE, "cv_regression.csv")
PDF_PATH = os.path.join(HERE, "leaderboard_regression.pdf")
PNG_TABLE = os.path.join(HERE, "leaderboard_reg_table.png")
PNG_R2 = os.path.join(HERE, "leaderboard_reg_r2.png")
PNG_RMSE = os.path.join(HERE, "leaderboard_reg_rmse.png")

SHOW = os.environ.get("LEADERBOARD_SHOW", "1") != "0"
SUBTITLE = ("Ranked by 5-fold CV R^2 (Loss_Data, lgd_time)  -  "
            "gold = leader, orange = statistical tie with leader")


def load_ranked():
    if not os.path.exists(CV_CSV):
        sys.exit(f"No results file found at {CV_CSV}\nRun run_study_regression.py first.")
    df = pd.read_csv(CV_CSV)
    if df.empty:
        sys.exit("cv_regression.csv is empty - run the regression study first.")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    else:
        df = df.drop_duplicates("model", keep="last")
    df = df.sort_values("CV_R2_mean", ascending=False).reset_index(drop=True)

    lead_lo = df.loc[0, "CV_R2_mean"] - df.loc[0, "CV_R2_std"]
    df["tie"] = (df["CV_R2_mean"] + df["CV_R2_std"]) >= lead_lo
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _row_color(rec, even):
    if rec["rank"] == 1:
        return "#f9e79f"
    if rec["tie"]:
        return "#fdebd0"
    return "#f4f6f7" if even else "white"


def _bar_color(rec):
    if rec["rank"] == 1:
        return "#f1c40f"
    if rec["tie"]:
        return "#e67e22"
    return "#5dade2"


# ---------------- PAGE 1: ranking table ----------------
def make_table_fig(df):
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(df) + 2.2))
    ax.axis("off")
    fig.suptitle("LGD Regression - Model Comparison Leaderboard",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.90, SUBTITLE, ha="center", fontsize=9.5, color="#444")

    headers = ["#", "Model", "R^2", "+/-std", "RMSE", "+/-std"]
    cell_text = [[f"{r.rank}", r.model,
                  f"{r.CV_R2_mean:.4f}", f"{r.CV_R2_std:.4f}",
                  f"{r.CV_RMSE_mean:.4f}", f"{r.CV_RMSE_std:.4f}"]
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
def make_bar_fig(df, mean_col, std_col, title, lower_better):
    # barh draws the first row at the bottom, so the BEST model must be LAST in `d`.
    #   higher-is-better (R^2)  -> ascending  (largest ends on top)
    #   lower-is-better  (RMSE) -> descending (smallest ends on top)
    d = df.sort_values(mean_col, ascending=not lower_better)
    colors = [_bar_color(r) for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 0.7 * len(df) + 2.5))
    bars = ax.barh(d["model"], d[mean_col], xerr=d[std_col],
                   color=colors, edgecolor="#34495e",
                   error_kw=dict(ecolor="#7f8c8d", capsize=3))
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel(title + ("  (lower is better)" if lower_better else "  (higher is better)"),
                  fontsize=11)
    if lower_better:
        # Zoom so tiny RMSE differences are visible (values cluster ~0.31-0.32).
        lo = (df[mean_col] - df[std_col]).min()
        hi = (df[mean_col] + df[std_col]).max()
        pad = (hi - lo) * 0.25 + 1e-6
        ax.set_xlim(max(0, lo - pad), hi + pad)
    else:
        ax.set_xlim(0, max(0.001, (df[mean_col] + df[std_col]).max() * 1.22))
    ax.margins(y=0.02)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"LGD Regression - {title}", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.93, SUBTITLE, ha="center", fontsize=9, color="#444")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.88, bottom=0.12)
    return fig


def main():
    df = load_ranked()
    print("Regression leaderboard (5-fold CV R^2):")
    print(df[["rank", "model", "CV_R2_mean", "CV_R2_std", "CV_RMSE_mean", "tie"]]
          .to_string(index=False))

    figs = [
        (make_table_fig(df), PNG_TABLE),
        (make_bar_fig(df, "CV_R2_mean", "CV_R2_std", "R^2 (primary)", lower_better=False), PNG_R2),
        (make_bar_fig(df, "CV_RMSE_mean", "CV_RMSE_std", "RMSE (secondary)", lower_better=True), PNG_RMSE),
    ]

    with PdfPages(PDF_PATH) as pdf:
        for fig, _ in figs:
            pdf.savefig(fig)
    for fig, png in figs:
        fig.savefig(png, dpi=150)

    print(f"\nSaved {len(figs)}-page PDF: {PDF_PATH}")
    print(f"Saved PNGs: {PNG_TABLE}, {PNG_R2}, {PNG_RMSE}")

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
