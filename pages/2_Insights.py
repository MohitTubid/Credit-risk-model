import os
import streamlit as st
from branding import header

# Project folder (one level up from pages/), where the PNG charts live.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

header("Insights", "Performance distributions & feature importance")
st.write("Pick a task and a chart. These images are produced by the study scripts.")

# Friendly chart name -> PNG filename, grouped by task.
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

# WIDGET 1: a radio button. Returns the label the user picked (a string).
task = st.radio("Task", list(CHARTS.keys()), horizontal=True)

# WIDGET 2: a dropdown, whose options depend on the task chosen above.
chart = st.selectbox("Chart", list(CHARTS[task].keys()))

# Because the script re-runs top-to-bottom on every click, `task` and `chart`
# always hold the current selections by the time we reach here.
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
