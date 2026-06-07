import streamlit as st
from branding import header

header("Credit Risk Model Comparison",
       "Where model performance becomes decision intelligence")

st.write(
    "Explore how leading machine learning models perform across credit risk scenarios "
    "and generate predictions using the strongest performers."
)

st.markdown("**Navigate using the sidebar:**")
st.markdown(
    """
- **Leaderboards** — compare model performance across credit risk tasks
- **Insights** — explore performance distributions and feature importance
- **Predict** — estimate default probability and potential loss
- **Train** — retrain and evaluate models (owner access)
    """
)
