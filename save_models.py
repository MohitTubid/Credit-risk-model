"""
Train the two leaderboard WINNERS and freeze them to disk for the app's Predict page.

Why: the Predict page must load ready-to-use models instantly; it cannot re-tune for
minutes on every visit. This script (owner-run) rebuilds each winner via its builder
(deterministic re-tune == the study model), fits it, and joblib-dumps a small "bundle":
the model + its feature list (+ the tuned decision threshold for the classifier).

Run:  python save_models.py
      (after run_study_classification.py and run_study_regression.py have produced the
       cv_*.csv leaderboards)
"""
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict

from ml_eval import (
    load_classification, load_regression, CV_CSV, CV_CSV_REG,
    CLF_FEATURES, REG_FEATURES, best_threshold_from_pr, sgkf, RuntimeTimer,
)
from classification_models import BUILDER_BY_NAME as CLF_BUILDERS
from regression_models import BUILDER_BY_NAME as REG_BUILDERS

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def winner(csv_path, metric):
    df = (pd.read_csv(csv_path)
            .sort_values("timestamp").drop_duplicates("model", keep="last")
            .sort_values(metric, ascending=False).reset_index(drop=True))
    return df.loc[0, "model"]


def save_classification():
    name = winner(CV_CSV, "CV_PR_AUC_mean")
    print(f"Classification winner: {name} -> rebuilding + fitting...")
    Xtr, Xte, ytr, yte, gtr = load_classification()
    _, model = CLF_BUILDERS[name](Xtr, ytr, gtr)
    model.fit(Xtr, ytr)
    # Tuned decision threshold from out-of-fold train probabilities (no leakage).
    oof = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=sgkf,
                            method="predict_proba", n_jobs=-1)[:, 1]
    thr, *_ = best_threshold_from_pr(ytr, oof)
    bundle = {"task": "classification", "winner": name,
              "features": CLF_FEATURES, "threshold": float(thr), "model": model}
    path = os.path.join(MODELS_DIR, "classification_winner.joblib")
    joblib.dump(bundle, path)
    print(f"  saved -> {path}  (threshold={thr:.4f})")


def save_regression():
    name = winner(CV_CSV_REG, "CV_R2_mean")
    print(f"Regression winner: {name} -> rebuilding + fitting...")
    Xtr, Xte, ytr, yte = load_regression()
    _, model = REG_BUILDERS[name](Xtr, ytr)
    model.fit(Xtr, ytr)
    bundle = {"task": "regression", "winner": name,
              "features": REG_FEATURES, "model": model}
    path = os.path.join(MODELS_DIR, "regression_winner.joblib")
    joblib.dump(bundle, path)
    print(f"  saved -> {path}")


def save_feature_defaults():
    """Precompute median feature values -> feature_defaults.json so the deployed Predict
    page can pick sensible defaults WITHOUT shipping the raw credit data."""
    from ml_eval import DATA_PATH
    fields = ["balance_time", "FICO_orig_time", "interest_rate_time", "LTV_time",
              "orig_time", "mat_time", "hpi_time", "gdp_time", "uer_time"]
    df = pd.read_csv(DATA_PATH)
    d = {f: round(float(df[f].median()), 4) for f in fields}
    path = os.path.join(HERE, "feature_defaults.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    with RuntimeTimer("save_models runtime"):
        save_classification()
        save_regression()
        save_feature_defaults()
    print("Done.")
