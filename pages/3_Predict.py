import os
import json
import joblib
import pandas as pd
import streamlit as st
from branding import header

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # model_study/
MODELS_DIR = os.path.join(HERE, "models")

# The 9 distinct inputs (union of both models' feature sets) + a friendly label and step.
FIELDS = [
    ("balance_time",       "Outstanding balance ($)",     1000.0),
    ("FICO_orig_time",     "FICO score (origination)",       1.0),
    ("interest_rate_time", "Interest rate (%)",              0.1),
    ("LTV_time",           "Loan-to-value now (%)",          1.0),
    ("orig_time",          "Origination time",               1.0),
    ("mat_time",           "Maturity time",                  1.0),
    ("hpi_time",           "House price index",              1.0),
    ("gdp_time",           "GDP growth (%)",                 0.1),
    ("uer_time",           "Unemployment rate (%)",          0.1),
]
FALLBACK = {f: 0.0 for f, _, _ in FIELDS}


# @st.cache_resource: load the models ONCE and reuse them for every interaction and every
# visitor. Without this, Streamlit would re-read the .joblib files on every single rerun.
@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(MODELS_DIR, "classification_winner.joblib"))
    reg = joblib.load(os.path.join(MODELS_DIR, "regression_winner.joblib"))
    return clf, reg


# @st.cache_data: cache plain DATA (the dict of default input values) so it's built once.
@st.cache_data
def feature_medians():
    # Prefer the precomputed JSON so the DEPLOYED app never needs the raw credit data.
    path = os.path.join(HERE, "feature_defaults.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback (local only): compute from the raw CSV if it happens to be available.
    try:
        from ml_eval import DATA_PATH
        df = pd.read_csv(DATA_PATH)
        return {f: float(df[f].median()) for f, _, _ in FIELDS}
    except Exception:
        return FALLBACK


header("Predict", "Default probability, loss given default, and expected loss")

try:
    clf, reg = load_models()
except FileNotFoundError:
    st.error("Trained models not found. The owner needs to run save_models.py "
             "(or use the Train page) to create them.")
    st.stop()

st.caption(f"Default-risk model: **{clf['winner']}**   |   Loss model: **{reg['winner']}**")
st.write("Enter the loan details below - predictions update live as you change them.")

# ---- Inputs: 9 number boxes laid out in 3 columns ----
defaults = feature_medians()
cols = st.columns(3)
vals = {}
for i, (feat, label, step) in enumerate(FIELDS):
    vals[feat] = cols[i % 3].number_input(label, value=round(defaults.get(feat, 0.0), 2), step=step)


def row_for(bundle):
    """Build a 1-row DataFrame with exactly the columns this model expects, in order."""
    feats = bundle["features"]
    return pd.DataFrame([[vals[f] for f in feats]], columns=feats)


# ---- Predictions ----
pd_prob = float(clf["model"].predict_proba(row_for(clf))[0, 1])   # P(default)
threshold = clf["threshold"]
lgd = float(reg["model"].predict(row_for(reg))[0])                # loss given default
lgd_clamped = max(0.0, lgd)
exp_loss_frac = pd_prob * lgd_clamped                            # PD x LGD
exp_loss_dollars = exp_loss_frac * vals["balance_time"]          # x exposure (balance)

st.subheader("Results")
c1, c2, c3 = st.columns(3)
c1.metric("Probability of default", f"{pd_prob:.1%}")
c1.write("Flag: **DEFAULT LIKELY**" if pd_prob >= threshold else "Flag: unlikely")
c1.caption(f"decision threshold = {threshold:.1%}")

c2.metric("Predicted LGD", f"{lgd:.3f}")
c2.caption("loss given default (fraction of balance, if it defaults)")

c3.metric("Expected loss", f"${exp_loss_dollars:,.0f}")
c3.caption(f"{exp_loss_frac:.2%} of balance")

with st.expander("How is Expected Loss computed?"):
    st.markdown(
        """
        **Expected Loss = PD x LGD x Exposure**, the standard credit-risk identity:
        - **PD** (probability of default) comes from the classification winner.
        - **LGD** (loss given default) comes from the regression winner - it is trained on
          *defaulted* loans, so it estimates loss *conditional on* a default happening.
        - **Exposure** is the outstanding balance.

        Multiplying them gives the loss you'd expect on average for this loan, blending
        *how likely* it is to default with *how much* you'd lose if it does.
        """
    )
