import os
import streamlit as st
import pandas as pd
import altair as alt
from branding import header

# This file lives in model_study/pages/, so the project folder (with the CSVs) is TWO levels up:
#   __file__               -> .../model_study/pages/1_Leaderboards.py
#   dirname(__file__)      -> .../model_study/pages
#   dirname(dirname(...))  -> .../model_study      <- the CSVs are here
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

header("Leaderboards", "Model rankings by cross-validation score")
st.write("Models ranked by cross-validation score. Overlapping +/- std means a statistical tie.")

# Friendly display names for the raw CSV column headers.
COLUMN_LABELS = {
    "rank": "Rank", "model": "Model",
    "CV_PR_AUC_mean": "PR-AUC", "CV_PR_AUC_std": "PR-AUC ±std",
    "CV_ROC_AUC_mean": "ROC-AUC", "CV_ROC_AUC_std": "ROC-AUC ±std",
    "CV_R2_mean": "R²", "CV_R2_std": "R² ±std",
    "CV_RMSE_mean": "RMSE", "CV_RMSE_std": "RMSE ±std",
}


def ranked(csv_name, metric):
    """Read a results CSV, keep the latest run per model, and rank by `metric` (descending).
    Returns None if the file isn't present (e.g. a stale deployment)."""
    path = os.path.join(HERE, csv_name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df = df.drop(columns=["timestamp"], errors="ignore")   # used for sorting only, not shown
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.rename(columns=COLUMN_LABELS)


def databar_table(df, metric_col):
    """Table with an in-cell bar drawn across the primary-metric column (white cells)."""
    sty = (df.style
             .set_properties(**{"background-color": "white", "color": "#1A1A1A"})
             .bar(subset=[metric_col], color="#9db8e8"))
    st.dataframe(sty, hide_index=True, width="stretch")


def heatmap_table(df):
    """Table with metric cells colour-graded (darker = better)."""
    higher = [c for c in ("PR-AUC", "ROC-AUC", "R²") if c in df.columns]
    lower = [c for c in ("RMSE",) if c in df.columns]
    sty = df.style.set_properties(**{"background-color": "white", "color": "#1A1A1A"})
    if higher:
        sty = sty.background_gradient(cmap="Blues", subset=higher)
    if lower:
        sty = sty.background_gradient(cmap="Blues_r", subset=lower)   # lower RMSE = better = darker
    st.dataframe(sty, hide_index=True, width="stretch")


def caterpillar(csv_name, mean_col, std_col, axis_title):
    """Ranked interval (caterpillar) plot: dot = mean, whisker = +/-std. A dashed line marks
    the leader's lower bound, so overlapping intervals (statistical ties) are obvious."""
    df = pd.read_csv(os.path.join(HERE, csv_name))
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    df = df[["model", mean_col, std_col]].copy()
    df["lo"] = df[mean_col] - df[std_col]
    df["hi"] = df[mean_col] + df[std_col]
    leader_lo = float(df.loc[df[mean_col].idxmax(), "lo"])
    order = df.sort_values(mean_col, ascending=False)["model"].tolist()
    xs = alt.Scale(zero=False)
    base = alt.Chart(df).encode(y=alt.Y("model:N", sort=order, title=None))
    rule = base.mark_rule(strokeWidth=2, color="#9aa0a6").encode(
        x=alt.X("lo:Q", title=axis_title, scale=xs), x2="hi:Q")
    point = base.mark_point(filled=True, size=120, color="#003399").encode(
        x=alt.X(f"{mean_col}:Q", title=axis_title, scale=xs),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip(f"{mean_col}:Q", title=axis_title, format=".4f"),
                 alt.Tooltip(f"{std_col}:Q", title="std", format=".4f")])
    ref = alt.Chart(pd.DataFrame({"x": [leader_lo]})).mark_rule(
        strokeDash=[5, 4], color="#b30000").encode(x=alt.X("x:Q", scale=xs))
    return (rule + point + ref).properties(height=320, width="container")


def show(csv_name, metric, title):
    st.subheader(title)
    df = ranked(csv_name, metric)
    if df is None:
        st.error(f"`{csv_name}` wasn't found in this deployment. If you just pushed new files, "
                 f"reboot the app (Manage app -> Reboot) to pull the latest from GitHub.")
        return
    std_col = metric.replace("_mean", "_std")
    axis_title = COLUMN_LABELS.get(metric, metric)
    t1, t2, t3 = st.tabs(["Data bars", "Heatmap", "Caterpillar"])
    with t1:
        databar_table(df, axis_title)
    with t2:
        heatmap_table(df)
    with t3:
        st.altair_chart(caterpillar(csv_name, metric, std_col, axis_title))
        st.caption("Dot = mean CV score, whisker = +/-std. The dashed red line is the leader's "
                   "lower bound - any model reaching past it is a statistical tie with the leader.")


show("cv_classification.csv", "CV_PR_AUC_mean", "Classification - default_time (ranked by CV PR-AUC)")
show("cv_regression.csv", "CV_R2_mean", "Regression - lgd_time (ranked by CV R^2)")
