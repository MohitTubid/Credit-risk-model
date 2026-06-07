"""
Registry of the 9 classification model builders (mirrors regression_models.py).

Each builder: (Xtr, ytr, gtr) -> (name, tuned_estimator). One source of truth for the
model roster, shared by run_study_classification.py and feature_importance_classification.py.
"""
from logistic_regression import make_model as build_logistic
from random_forest import make_model as build_rf
from bagging import make_model as build_bagging
from adaboost import make_model as build_adaboost
from boosting_models import make_histgb, make_xgb
from lightgbm_model import make_model as build_lightgbm
from catboost_model import make_model as build_catboost
from svm import make_model as build_svm

ALL_BUILDERS = [build_logistic, build_rf, build_bagging, build_adaboost,
                make_histgb, make_xgb, build_lightgbm, build_catboost, build_svm]

# Lookup by the name each builder returns -> rebuild a chosen model (e.g. the leaderboard
# winner) without re-running the whole study. Re-tuning is deterministic (fixed seed).
BUILDER_BY_NAME = {
    "Logistic Regression": build_logistic,
    "Random Forest": build_rf,
    "Bagging": build_bagging,
    "AdaBoost": build_adaboost,
    "HistGradientBoosting": make_histgb,
    "XGBoost": make_xgb,
    "LightGBM": build_lightgbm,
    "CatBoost": build_catboost,
    "SVM (RBF approx)": build_svm,
}
