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


def white(df):
    """Paint the table cells white (the page background is grey)."""
    return df.style.set_properties(**{"background-color": "white", "color": "#1A1A1A"})


def altair_bar(csv_name, mean_col, std_col, axis_title):
    """Interactive horizontal bar chart with +/-std error bars and hover tooltips."""
    df = pd.read_csv(os.path.join(HERE, csv_name))
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    df = df[["model", mean_col, std_col]].copy()
    df["lo"] = df[mean_col] - df[std_col]
    df["hi"] = df[mean_col] + df[std_col]
    base = alt.Chart(df).encode(y=alt.Y("model:N", sort="-x", title=None))
    bars = base.mark_bar(color="#003399").encode(
        x=alt.X(f"{mean_col}:Q", title=axis_title),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip(f"{mean_col}:Q", title=axis_title, format=".4f"),
                 alt.Tooltip(f"{std_col}:Q", title="std", format=".4f")],
    )
    err = base.mark_errorbar(color="#444").encode(x=alt.X("lo:Q", title=axis_title), x2="hi:Q")
    return (bars + err).properties(height=300, width="container")


def show(csv_name, metric, title):
    st.subheader(title)
    df = ranked(csv_name, metric)
    if df is None:
        st.error(f"`{csv_name}` wasn't found in this deployment. If you just pushed new files, "
                 f"reboot the app (Manage app -> Reboot) to pull the latest from GitHub.")
        return
    st.dataframe(white(df), width="stretch")
    std_col = metric.replace("_mean", "_std")
    st.altair_chart(altair_bar(csv_name, metric, std_col, COLUMN_LABELS.get(metric, metric)))


show("cv_classification.csv", "CV_PR_AUC_mean", "Classification - default_time (ranked by CV PR-AUC)")
show("cv_regression.csv", "CV_R2_mean", "Regression - lgd_time (ranked by CV R^2)")
