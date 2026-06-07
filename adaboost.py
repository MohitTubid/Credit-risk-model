"""AdaBoost (shallow weak learners), tuned on grouped CV by PR-AUC.

sklearn 1.8: the `algorithm` parameter is gone (SAMME only) -- do not pass it.
Base learners are deliberately shallow (stumps to depth-3); boosting reduces bias.
"""
from scipy.stats import randint, uniform
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    # No class_weight here: AdaBoost reweights samples itself each round. Stacking
    # class_weight='balanced' on top makes the stump "worse than random" on the
    # reweighted distribution and aborts the fit (FitFailedWarning).
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    dist = {
        "n_estimators":         randint(100, 600),
        "learning_rate":        uniform(0.01, 1.0),
        "estimator__max_depth": [1, 2, 3],
    }
    est, _ = tune(ada, dist, Xtr, ytr, gtr, n_iter=25)
    return "AdaBoost", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    name, est = make_model(Xtr, ytr, gtr)
    cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
