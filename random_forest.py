"""Random Forest classifier, tuned on grouped CV by PR-AUC."""
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                random_state=RANDOM_STATE)
    dist = {
        "n_estimators":     randint(200, 500),
        "max_depth":        [8, 12, 16, None],
        "min_samples_leaf": randint(5, 50),
        "max_features":     ["sqrt", "log2"],
    }
    est, _ = tune(rf, dist, Xtr, ytr, gtr, n_iter=25)
    return "Random Forest", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    name, est = make_model(Xtr, ytr, gtr)
    cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
