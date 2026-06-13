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

st.divider()
st.subheader("How it works")

# A Graphviz DOT diagram of the pipeline (rendered in-browser; no graphviz binary needed).
# Raw string so the \n in node labels stay literal for Graphviz to interpret as line breaks.
DOT = r"""
digraph {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF",
        color="#003399", fontname="Helvetica", fontsize=11];
  edge [color="#003399"];
  data    [label="Loan data"];
  clean   [label="Clean +\nselect features"];
  split   [label="Group split\nby loan id"];
  tune    [label="Tune 9 models\n(per task)"];
  rank    [label="Grouped-CV rank\n(mean +/- std)"];
  winner  [label="Winner", fillcolor="#003399", fontcolor="white"];
  predict [label="Predict:\nExpected Loss = PD x LGD x Exposure", fillcolor="#003399", fontcolor="white"];
  data -> clean -> split -> tune -> rank -> winner -> predict;
}
"""
st.graphviz_chart(DOT, use_container_width=True)

