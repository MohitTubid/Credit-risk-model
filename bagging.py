"""Bagging of decision trees, tuned on grouped CV by PR-AUC.

Note: base trees are intentionally deep (high-variance) -- bagging reduces their variance.
This is the opposite of AdaBoost, which uses shallow weak learners.
"""
from scipy.stats import randint, uniform
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced",
                                         random_state=RANDOM_STATE),
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    dist = {
        "n_estimators":          randint(50, 200),
        "max_samples":           uniform(0.5, 0.5),   # [0.5, 1.0)
        "max_features":          uniform(0.5, 0.5),
        "estimator__max_depth":  [None, 10, 20],
    }
    est, _ = tune(bag, dist, Xtr, ytr, gtr, n_iter=15)
    return "Bagging", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    name, est = make_model(Xtr, ytr, gtr)
    cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
