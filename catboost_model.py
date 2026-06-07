"""CatBoost gradient boosting classifier, tuned on grouped CV by PR-AUC.

FILENAME: `catboost_model.py`, NOT `catboost.py` -- a module named catboost.py would
shadow the installed `catboost` package and break `from catboost import ...`
(especially on the case-insensitive Windows filesystem). Keep the `_model` suffix.

CatBoost differs from XGBoost/LightGBM: it builds symmetric (oblivious) trees and uses
ordered boosting, which often makes it robust on smaller/structured data. Imbalance is
handled with scale_pos_weight = n_neg / n_pos, matching XGBoost/LightGBM for a fair
comparison.
"""
from scipy.stats import randint, uniform

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    from catboost import CatBoostClassifier   # lazy import so the study runs without catboost
    spw = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
    cat = CatBoostClassifier(
        loss_function="Logloss",
        scale_pos_weight=spw,          # imbalance handling
        bootstrap_type="Bernoulli",    # enables row subsampling via `subsample`
        random_seed=RANDOM_STATE,
        thread_count=-1,
        allow_writing_files=False,     # don't litter a catboost_info/ dir during CV
        verbose=False,
    )
    dist = {
        "iterations":    randint(150, 600),     # = n_estimators
        "learning_rate": uniform(0.01, 0.3),
        "depth":         randint(4, 10),        # symmetric-tree depth
        "l2_leaf_reg":   uniform(1.0, 9.0),     # L2 regularization
        "subsample":     uniform(0.6, 0.4),     # row subsampling [0.6, 1.0)
        "rsm":           uniform(0.6, 0.4),     # feature subsampling (random subspace)
    }
    est, _ = tune(cat, dist, Xtr, ytr, gtr, n_iter=25)
    return "CatBoost", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    try:
        name, est = make_model(Xtr, ytr, gtr)
        cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
    except Exception as e:
        print(f"[SKIP] CatBoost: {type(e).__name__}: {e}")
