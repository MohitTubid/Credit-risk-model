"""Scalable SVM for ~62k rows: RBF kernel approximated via Nystroem + LinearSVC.

Plain kernel SVC is O(n^2)-O(n^3) and infeasible at this size. Nystroem builds an
explicit RBF feature map, then a fast linear SVM is fit in that space. LinearSVC has
no predict_proba, so the tuned pipeline is wrapped in CalibratedClassifierCV (Platt
scaling) -- needed for PR-AUC / ROC-AUC and the final test's predict_proba.
"""
from scipy.stats import loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from ml_eval import tune, load_classification, cv_evaluate_classifier, RANDOM_STATE


def make_model(Xtr, ytr, gtr):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", random_state=RANDOM_STATE)),
        ("svc", LinearSVC(class_weight="balanced", dual=False, max_iter=5000,
                          random_state=RANDOM_STATE)),
    ])
    dist = {
        "nystroem__gamma":        loguniform(1e-4, 1e0),
        "nystroem__n_components": randint(200, 600),
        "svc__C":                 loguniform(1e-3, 1e2),
    }
    # Tune fast via LinearSVC.decision_function -> PR-AUC; calibrate only the winner.
    best, _ = tune(pipe, dist, Xtr, ytr, gtr, n_iter=15)
    return "SVM (RBF approx)", CalibratedClassifierCV(best, method="sigmoid", cv=3)


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, gtr = load_classification()
    name, est = make_model(Xtr, ytr, gtr)
    cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
