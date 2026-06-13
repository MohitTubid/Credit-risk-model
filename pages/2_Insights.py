import os
import streamlit as st
import pandas as pd
import pydeck as pdk
from branding import header

# Project folder (one level up from pages/), where the data files live.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

header("Insights", "Geographic risk, performance distributions & feature importance")

# ----------------------------------------------------------------------
# 3D map: observed default rate by US state (from precomputed state_stats.csv)
# ----------------------------------------------------------------------
st.subheader("Default risk by US state")
state_path = os.path.join(HERE, "state_stats.csv")
if os.path.exists(state_path):
    sdf = pd.read_csv(state_path)
    # Red gradient: higher default rate -> deeper red.
    mx = max(float(sdf["default_rate"].max()), 1e-9)
    sdf["r"] = 220
    sdf["g"] = (200 * (1 - sdf["default_rate"] / mx)).round().astype(int)
    sdf["b"] = 60

    layer = pdk.Layer(
        "ColumnLayer",
        data=sdf,
        get_position=["lon", "lat"],
        get_elevation="default_rate",
        elevation_scale=3_000_000,     # default_rate is ~0.01-0.06; scale up to visible columns
        radius=45000,
        get_fill_color=["r", "g", "b", 200],
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=39.5, longitude=-98, zoom=3.2, pitch=45)
    tooltip = {
        "html": "<b>{state}</b><br/>Default rate: {default_pct}%<br/>"
                "Avg LGD: {avg_lgd}<br/>Loans: {n}",
        "style": {"backgroundColor": "#003399", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
    st.caption("Column height & redness = observed default rate. Hover a state for details.")
else:
    st.info("`state_stats.csv` not found - run `make_state_stats.py` to generate it.")

st.divider()

# ----------------------------------------------------------------------
# Performance charts (PNGs produced by the study scripts)
# ----------------------------------------------------------------------
st.subheader("Performance charts")
st.write("Pick a task and a chart.")

CHARTS = {
    "Classification": {
        "Leaderboard - PR-AUC": "leaderboard_pr_auc.png",
        "Leaderboard - ROC-AUC": "leaderboard_roc_auc.png",
        "Score distribution - PR-AUC": "cv_dist_classification_PR_AUC.png",
        "Score distribution - ROC-AUC": "cv_dist_classification_ROC_AUC.png",
        "Feature importance": "feature_importance_classification.png",
    },
    "Regression": {
        "Leaderboard - R^2": "leaderboard_reg_r2.png",
        "Leaderboard - RMSE": "leaderboard_reg_rmse.png",
        "Score distribution - R^2": "cv_dist_regression_R2.png",
        "Score distribution - -RMSE": "cv_dist_regression_neg_RMSE.png",
        "Feature importance": "feature_importance_regression.png",
    },
}

task = st.radio("Task", list(CHARTS.keys()), horizontal=True)
chart = st.selectbox("Chart", list(CHARTS[task].keys()))

filename = CHARTS[task][chart]
path = os.path.join(HERE, filename)
if os.path.exists(path):
    st.image(path, caption=f"{task} - {chart}", width="stretch")
else:
    st.warning(
        f"'{filename}' hasn't been generated yet. Run the study and plot scripts "
        f"(e.g. run_study_*.py, leaderboard_*.py, plot_cv_distributions.py, "
        f"feature_importance_*.py) to create it."
    )
