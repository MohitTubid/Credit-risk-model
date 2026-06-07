import os
import streamlit as st
import pandas as pd
from branding import header

# This file lives in model_study/pages/, so the project folder (with the CSVs) is TWO levels up:
#   __file__               -> .../model_study/pages/1_Leaderboards.py
#   dirname(__file__)      -> .../model_study/pages
#   dirname(dirname(...))  -> .../model_study      <- the CSVs are here
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

header("Leaderboards", "Model rankings by cross-validation score")
st.write("Models ranked by cross-validation score. Overlapping +/- std means a statistical tie.")


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
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def white(df):
    """Paint the table cells white (the page background is grey)."""
    return df.style.set_properties(**{"background-color": "white", "color": "#1A1A1A"})


def show(csv_name, metric, title):
    st.subheader(title)
    df = ranked(csv_name, metric)
    if df is None:
        st.error(f"`{csv_name}` wasn't found in this deployment. If you just pushed new files, "
                 f"reboot the app (Manage app -> Reboot) to pull the latest from GitHub.")
    else:
        st.dataframe(white(df), width="stretch")


show("cv_classification.csv", "CV_PR_AUC_mean", "Classification - default_time (ranked by CV PR-AUC)")
show("cv_regression.csv", "CV_R2_mean", "Regression - lgd_time (ranked by CV R^2)")
