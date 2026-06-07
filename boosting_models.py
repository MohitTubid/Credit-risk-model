"""Gradient boosting family: Histogram Gradient Boosting (sklearn) + XGBoost.

HistGradientBoostingClassifier supports class_weight='balanced' (classic
GradientBoostingClassifier does not), and is fast/histogram-based.
XGBoost handles imbalance via scale_pos_weight = n_neg / n_pos.
"""
from scipy.stats import randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_histgb(Xtr, ytr, gtr):
    hgb = HistGradientBoostingClassifier(class_weight="balanced",
                                         random_state=RANDOM_STATE)
    dist = {
        "learning_rate":     uniform(0.01, 0.3),
        "max_iter":          randint(150, 600),
        "max_leaf_nodes":    randint(15, 63),
        "min_samples_leaf":  randint(20, 100),
        "l2_regularization": uniform(0.0, 1.0),
    }
    est, _ = tune(hgb, dist, Xtr, ytr, gtr, n_iter=25)
    return "HistGradientBoosting", est


def make_xgb(Xtr, ytr, gtr):
    from xgboost import XGBClassifier   # imported lazily so the study runs without xgboost
    spw = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                        tree_method="hist", scale_pos_weight=spw,
                        n_jobs=-1, random_state=RANDOM_STATE)
    dist = {
        "n_estimators":     randint(150, 600),
        "learning_rate":    uniform(0.01, 0.3),
        "max_depth":        randint(2, 7),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_lambda":       uniform(0.0, 2.0),
    }
    est, _ = tune(xgb, dist, Xtr, ytr, gtr, n_iter=25)
    return "XGBoost", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    for builder in (make_histgb, make_xgb):
        try:
            name, est = builder(Xtr, ytr, gtr)
            cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
        except Exception as e:
            print(f"[SKIP] {builder.__name__}: {type(e).__name__}: {e}")
