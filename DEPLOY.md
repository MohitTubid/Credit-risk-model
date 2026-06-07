# Deploying the Credit Risk Model Study (Streamlit Community Cloud)

This guide hosts the app for free so **anyone can interact** with it, while **only you**
can unlock the Train page (password) and update the models.

---

## The one big idea about hosting + training

Free Streamlit Cloud is great for **serving** (viewing leaderboards/charts, making
predictions) but is too small (≈1 CPU, ≈1 GB RAM, app sleeps when idle) to run the full
14-minute study. So the practical model is:

- **Cloud app = serving.** Anyone can use Leaderboards / Charts / Predict.
- **You train LOCALLY**, then commit the updated `models/*.joblib` (and the cv_*.csv /
  PNGs) and `git push`. Streamlit Cloud auto-redeploys with your new models.
- The **password-gated Train page** still protects in-app actions, but treat heavy
  retraining as a local task.

So "only I can update the model" = only you have the password **and** only you can push
to the GitHub repo. Both gates are yours.

---

## Step A - Put the project on GitHub

1. Create a free account at https://github.com and a **new repository** (e.g. `credit-risk-app`).
2. The repo should contain (from this `model_study/` folder):
   - `app.py`, `pages/`, all the `*.py` scripts
   - `requirements.txt`
   - `models/` (the `.joblib` files - these get served on the cloud)
   - `cv_classification.csv`, `cv_regression.csv`, the `cv_folds_*.csv`, and the PNGs
     (so leaderboards/charts render without re-running anything)
   - the data CSVs **only if** you're comfortable publishing them (see Privacy below)
   - `.gitignore` (already excludes `secrets.toml`)
3. Do **NOT** commit `.streamlit/secrets.toml` (your password). It's already gitignored.

Basic commands (run in the `model_study` folder):
```bash
git init
git add .
git commit -m "Credit risk model study app"
git branch -M main
git remote add origin https://github.com/<you>/credit-risk-app.git
git push -u origin main
```

## Step B - Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** -> pick your repo, branch `main`, and **Main file path** = `app.py`.
3. Click **Deploy**. It installs `requirements.txt` and starts the app (first build = a few min).
4. You get a public URL like `https://<your-app>.streamlit.app` - share it with anyone.

## Step C - Set your password on the cloud

1. In the app's page on Streamlit Cloud: **Settings -> Secrets**.
2. Paste:
   ```toml
   admin_password = "your-real-strong-password"
   ```
3. Save. Now the Train page unlocks only with that password. (This replaces the local
   `secrets.toml`, which you never uploaded.)

## Step D - Updating models later (the normal workflow)

1. Locally: change data / retrain ->
   ```bash
   python run_study_classification.py
   python run_study_regression.py
   python save_models.py
   ```
2. Commit + push the updated `models/` and `cv_*.csv` / PNGs.
3. Streamlit Cloud redeploys automatically. Done.

---

## Privacy note (credit data)

The **Predict** page reads `default_data.csv` only to pick sensible default values
(feature medians). If you'd rather **not publish the raw data**, precompute those medians
into a small file and load that instead - ask and I'll wire it up. The trained `.joblib`
models do not contain raw rows, so serving predictions does not require shipping the data.

## Run locally (reminder)

```bash
cd model_study
streamlit run app.py
# opens http://localhost:8501
```
