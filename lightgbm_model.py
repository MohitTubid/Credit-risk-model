"""LightGBM gradient boosting classifier, tuned on grouped CV by PR-AUC.

NOTE ON THE FILENAME: this file is `lightgbm_model.py`, NOT `lightgbm.py`.
A module named lightgbm.py would shadow the installed `lightgbm` package and make
`from lightgbm import LGBMClassifier` import itself (it breaks especially on the
case-insensitive Windows filesystem). Keep the `_model` suffix.

Imbalance is handled with scale_pos_weight = n_neg / n_pos (same lever as XGBoost),
so LightGBM and XGBoost are an apples-to-apples comparison.
"""
from scipy.stats import randint, uniform

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    from lightgbm import LGBMClassifier   # lazy import so the study runs without lightgbm
    spw = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
    lgbm = LGBMClassifier(
        objective="binary",
        scale_pos_weight=spw,       # imbalance handling
        subsample_freq=1,           # required for subsample (<1.0) to actually apply
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    dist = {
        "n_estimators":      randint(150, 600),
        "learning_rate":     uniform(0.01, 0.3),
        "num_leaves":        randint(15, 100),    # LightGBM's main complexity knob
        "max_depth":         [-1, 4, 8, 12],      # -1 = no limit
        "min_child_samples": randint(20, 100),
        "subsample":         uniform(0.6, 0.4),   # row subsampling [0.6, 1.0)
        "colsample_bytree":  uniform(0.6, 0.4),   # feature subsampling
        "reg_lambda":        uniform(0.0, 2.0),
    }
    est, _ = tune(lgbm, dist, Xtr, ytr, gtr, n_iter=25)
    return "LightGBM", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    try:
        name, est = make_model(Xtr, ytr, gtr)
        cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
    except Exception as e:
        print(f"[SKIP] LightGBM: {type(e).__name__}: {e}")
