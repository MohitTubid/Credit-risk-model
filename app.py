import streamlit as st
from branding import header

# set_page_config MUST be the first Streamlit command, and only goes in the main entry file.
# It sets the browser tab title/icon and uses the full page width.
st.set_page_config(page_title="Credit Risk Model Study", page_icon=":bar_chart:", layout="wide")

header("Credit Risk Model Comparison",
       "Default probability & loss severity across 9 ML models")

st.write(
    """
    Welcome! This app lets you explore a study comparing 9 machine-learning models on two
    credit-risk tasks, and make predictions from the trained winners.

    **Use the sidebar on the left to navigate:**
    - **Leaderboards** - model rankings for both tasks
    - **Charts** - performance distributions and feature importance
    - **Predict** - get a default probability and predicted loss for a loan
    - **Train** - *(owner only)* retrain the models
    """
)

st.caption("Classification: default_time (default_data.csv)  |  Regression: lgd_time (Loss_Data.csv)")
