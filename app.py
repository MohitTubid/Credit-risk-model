import streamlit as st

# set_page_config MUST be the first Streamlit command, and only goes in the main entry file.
# It sets the browser tab title/icon and uses the full page width.
st.set_page_config(page_title="Credit Risk Model Study", page_icon=":bar_chart:", layout="wide")

# A royal-blue brand banner. The theme's primaryColor tints *widgets*; a page header like
# this is drawn manually with inline HTML so the brand color is visible even where there
# are no widgets. unsafe_allow_html lets st.markdown render raw HTML.
st.markdown(
    "<div style='background-color:#003399; padding:16px 20px; border-radius:8px; "
    "margin-bottom:12px;'>"
    "<h1 style='color:#FFFFFF; margin:0; font-size:1.9rem;'>Credit Risk Model Comparison</h1>"
    "<p style='color:#DBDBDB; margin:4px 0 0 0;'>Default probability &amp; loss severity "
    "across 9 ML models</p>"
    "</div>",
    unsafe_allow_html=True,
)

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
