"""Logistic Regression (with optional degree-2 features), tuned on grouped CV by PR-AUC."""
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from ml_eval import tune, load_classification, cv_evaluate_classifier


def make_model(Xtr, ytr, gtr):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),   # poly BEFORE scaling
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    dist = {"poly__degree": [1, 2], "logreg__C": loguniform(1e-3, 1e2)}
    est, _ = tune(pipe, dist, Xtr, ytr, gtr, n_iter=15)
    return "Logistic Regression", est


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    name, est = make_model(Xtr, ytr, gtr)
    cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
