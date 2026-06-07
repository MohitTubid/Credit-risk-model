# Credit Risk Model Comparison

> Where model performance becomes decision intelligence.

An end-to-end study comparing **9 machine-learning models** across two credit-risk tasks,
wrapped in an interactive **Streamlit** web app. Anyone can explore the results and generate
predictions; only the owner can retrain the models.

<!-- After deploying, paste your URL here: -->
**Live app:** `https://<your-app>.streamlit.app`

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)

---

## What it does

Two separate prediction problems on a mortgage-loan dataset:

| Task | Target | Data | Primary metric |
|------|--------|------|----------------|
| **Classification** | `default_time` — will the loan default? | `default_data.csv` (62k loan-months, ~2.45% default) | **PR-AUC** |
| **Regression** | `lgd_time` — loss given default | `Loss_Data.csv` (1.2k defaulted loans) | **R²** |

For each task, 9 models are tuned and ranked: **Logistic/Linear Regression, Random Forest,
Bagging, AdaBoost, HistGradientBoosting, XGBoost, LightGBM, CatBoost, SVM/SVR**.

The two winners combine into the classic credit-risk identity on the **Predict** page:

> **Expected Loss = PD × LGD × Exposure**
> (probability of default × loss given default × outstanding balance)

## Headline result

All 9 models land in a **statistical tie** on each task (overlapping ±std across folds) —
a clean *"the features are the ceiling, not the algorithm"* finding.

| Task | Top model | Score |
|------|-----------|-------|
| Classification | SVM (RBF approx) | PR-AUC ≈ 0.077 |
| Regression | CatBoost | R² ≈ 0.10 |

## Methodology highlights

- **Group-aware cross-validation** by loan `id` (`StratifiedGroupKFold`) so no borrower
  leaks across train/test — predictions measure generalization to *new* borrowers.
- **PR-AUC as the primary metric** for the imbalanced (~2.45%) default task; accuracy is
  useless and ROC-AUC is over-optimistic at that prevalence.
- **Rank by CV mean ± std, test once** — the held-out set is touched a single time, on the
  CV winner, to avoid winner's-curse bias across many models.
- **Decision-threshold tuning** from the precision-recall curve (the default 0.5 cutoff is
  meaningless at 2.45% prevalence).
- **Honest evaluation everywhere** — scaling/encoding inside CV folds, no target leakage.

## The app

| Page | What it shows |
|------|---------------|
| **App** | Overview + navigation |
| **Leaderboards** | Ranked tables for both tasks |
| **Insights** | Per-fold score distributions and feature importance |
| **Predict** | Enter loan details → default probability, predicted LGD, and expected loss |
| **Train** | *(password-protected)* retrain / refresh the deployed models |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py        # opens http://localhost:8501
```

Reproduce the study from scratch:

```bash
python run_study_classification.py    # tunes + ranks the 9 classifiers
python run_study_regression.py        # tunes + ranks the 9 regressors
python save_models.py                 # freezes the winners for the Predict page
```

## Deploy

Hosted free on **Streamlit Community Cloud**: push this folder to GitHub, then point
[share.streamlit.io](https://share.streamlit.io) at the repo with main file `app.py`. Set
`admin_password` in the app's **Settings → Secrets** to lock the Train page. Full guide in
[`DEPLOY.md`](DEPLOY.md).

## Privacy

The raw credit data is **not** in this repo. The app serves predictions from the saved
models plus precomputed feature medians (`feature_defaults.json`), so no raw rows are needed
to run it.

## Project structure

```
model_study/
├── app.py                       # Streamlit entry (st.navigation router)
├── home.py, pages/              # app pages: Leaderboards, Insights, Predict, Train
├── ml_eval.py                   # shared harness: split, CV, tuning, metrics
├── classification_models.py     # 9 classifier builders + registry
├── regression_models.py         # 9 regressor builders + registry
├── run_study_*.py               # run + rank each task
├── save_models.py               # freeze winners -> models/*.joblib
├── leaderboard_*.py             # leaderboard PDFs
├── plot_cv_distributions.py     # box-plot distributions
├── feature_importance_*.py      # permutation importance
├── combined_report.py           # single multi-page PDF report
├── models/                      # saved winner models (joblib)
└── requirements.txt
```
