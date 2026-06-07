import os
import sys
import hmac
import subprocess
import streamlit as st

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # model_study/
PY = sys.executable   # the same Python that runs this app (the venv interpreter)

# Run child scripts headless so they never try to pop open a window / PDF on the server.
ENV = {**os.environ, "MPLBACKEND": "Agg", "LEADERBOARD_SHOW": "0"}

st.title("Train (owner only)")


# --------------------------------------------------------------------------
# Password gate. st.session_state remembers you're logged in across reruns.
# --------------------------------------------------------------------------
def check_password():
    if st.session_state.get("authed"):
        return True
    try:
        correct = st.secrets.get("admin_password", "")
    except Exception:
        correct = ""
    if not correct:
        st.warning(
            "No admin password is configured yet. Owner: set `admin_password` in "
            "`.streamlit/secrets.toml` (local) or the app's **Settings -> Secrets** (cloud)."
        )
        return False

    st.write("This page is restricted to the owner.")
    pw = st.text_input("Admin password", type="password")
    if pw:
        # compare_digest is a constant-time comparison (avoids timing attacks).
        if hmac.compare_digest(pw, correct):
            st.session_state["authed"] = True
            st.rerun()          # immediately re-run so the page shows the unlocked view
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()   # nothing below this line renders unless authenticated


# --------------------------------------------------------------------------
# Unlocked owner controls
# --------------------------------------------------------------------------
st.success("Unlocked. You can retrain / update the deployed models here.")


def run_and_stream(steps):
    """Run a list of (command, label) subprocesses, streaming their output live."""
    for cmd, label in steps:
        with st.status(label, expanded=True) as status:
            box = st.empty()
            logs = []
            proc = subprocess.Popen(cmd, cwd=HERE, env=ENV, text=True, bufsize=1,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                clean = line.rstrip().split("\r")[-1]   # drop the runtime-ticker prefix
                if clean:
                    logs.append(clean)
                    box.code("\n".join(logs[-25:]))
            code = proc.wait()
            if code == 0:
                status.update(label=f"{label} - done", state="complete")
            else:
                status.update(label=f"{label} - FAILED (exit {code})", state="error")
                return False
    return True


st.subheader("1. Refresh deployed models")
st.caption("Refits the current leaderboard winners on the latest data and updates the "
           "Predict page. ~5-8 minutes.")
if st.button("Refresh models now", type="primary"):
    ok = run_and_stream([([PY, "save_models.py"], "Refreshing models")])
    if ok:
        st.cache_resource.clear()   # force the Predict page to load the NEW models
        st.success("Models refreshed - the Predict page now uses the updated models.")

st.divider()

st.subheader("2. Full re-run (advanced, slow)")
st.caption("Re-runs BOTH studies (re-tunes all 9 models per task), then refreshes the "
           "saved models and all charts. Can take 20+ minutes - keep this tab open.")
if st.button("Re-run the full study"):
    ok = run_and_stream([
        ([PY, "run_study_classification.py"], "Classification study"),
        ([PY, "run_study_regression.py"], "Regression study"),
        ([PY, "save_models.py"], "Saving winners"),
        ([PY, "leaderboard_classification.py"], "Classification leaderboard"),
        ([PY, "leaderboard_regression.py"], "Regression leaderboard"),
        ([PY, "plot_cv_distributions.py"], "Distribution charts"),
        ([PY, "feature_importance_classification.py"], "Classification importance"),
        ([PY, "feature_importance_regression.py"], "Regression importance"),
    ])
    if ok:
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Full re-run complete - leaderboards, charts, and models are all updated.")
