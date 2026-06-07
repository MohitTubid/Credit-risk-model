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
    """Read a results CSV, keep the latest run per model, and rank by `metric` (descending)."""
    df = pd.read_csv(os.path.join(HERE, csv_name))
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates("model", keep="last")
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


st.subheader("Classification - default_time (ranked by CV PR-AUC)")
st.dataframe(ranked("cv_classification.csv", "CV_PR_AUC_mean"), width="stretch")

st.subheader("Regression - lgd_time (ranked by CV R^2)")
st.dataframe(ranked("cv_regression.csv", "CV_R2_mean"), width="stretch")
