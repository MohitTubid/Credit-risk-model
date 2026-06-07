"""
Shared evaluation harness for the default-prediction model comparison study.

Design decisions (locked in with the user):
  * GROUP-AWARE SPLIT by loan `id` (GroupShuffleSplit holdout + StratifiedGroupKFold CV)
    -> no loan's rows straddle train/test. Measures generalization to NEW borrowers.
  * RANK BY GROUPED-CV PR-AUC (mean +/- std); the test set is touched ONCE, on the winner.
  * PR-AUC (average_precision) is the PRIMARY metric: default_time is only ~2.45% positive,
    so accuracy is useless and ROC-AUC is over-optimistic.

Scope caveat (document in any write-up): because macro features (hpi/gdp/uer) are shared
across loans in-sample, these estimates measure new-borrower generalization within the
observed period, NOT out-of-time forecasting through a future credit cycle.
"""
import os
import time
import threading
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, KFold, RandomizedSearchCV,
    train_test_split, cross_validate, cross_val_predict,
)
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
)

RANDOM_STATE, TEST_SIZE, GROUP_COL = 42, 0.2, "id"
HERE = os.path.dirname(os.path.abspath(__file__))
def _find_data(filename):
    """Locate a data CSV across likely folders (robust to project reorganisation)."""
    candidates = [
        os.path.join(os.path.dirname(HERE), "ml_models", filename),  # Quant Finance/ml_models/
        os.path.join(os.path.dirname(HERE), filename),               # Quant Finance/
        os.path.join(HERE, filename),                                # model_study/
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]   # fall back to first; a clear FileNotFoundError will follow


DATA_PATH = _find_data("default_data.csv")
CV_CSV = os.path.join(HERE, "cv_classification.csv")
FOLDS_CSV = os.path.join(HERE, "cv_folds_classification.csv")   # per-fold scores (long format)

CLF_FEATURES = ["mat_time", "balance_time", "interest_rate_time", "hpi_time",
                "gdp_time", "uer_time", "FICO_orig_time"]
TARGET = "default_time"

# ---- Regression track (Loss_Data.csv, target = lgd_time) ----
# Loss_Data is CROSS-SECTIONAL: 1 row per loan (1,221 unique ids), so NO grouping is
# needed -- a plain random split + KFold is already fair. lgd_time is loss-given-default
# on defaulted loans only; it is NOT bounded to [0, 1] (it reaches ~2.0), so no clipping.
# recovery_res and res_time are excluded (target leakage / missingness).
REG_DATA_PATH = _find_data("Loss_Data.csv")
REG_FEATURES = ["orig_time", "balance_time", "LTV_time", "interest_rate_time", "FICO_orig_time"]
REG_TARGET = "lgd_time"
CV_CSV_REG = os.path.join(HERE, "cv_regression.csv")
FOLDS_CSV_REG = os.path.join(HERE, "cv_folds_regression.csv")

# Group-aware CV splitter for classification (loans repeat -> must not straddle folds).
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# Plain CV splitter for regression (Loss_Data is 1 row per loan -> no grouping needed).
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Carriage-return line-clear: erase the live RuntimeTimer line before logging,
# so the ticking "Runtime: HH:MM:SS" never collides with [CV]/result output.
CLR = "\r" + " " * 52 + "\r"


class RuntimeTimer:
    """Live HH:MM:SS runtime counter that runs in a background thread.

    Use as a context manager so it always stops cleanly -- even if the wrapped
    code raises:

        with RuntimeTimer("Study runtime"):
            ...long-running work...

    Prints an updating line every second and the total elapsed time on exit.
    """

    def __init__(self, label="Runtime"):
        self.label = label
        self._running = False
        self._thread = None
        self._start = None

    def _loop(self):
        while self._running:
            h, rem = divmod(int(time.time() - self._start), 3600)
            m, s = divmod(rem, 60)
            print(f"\r{self.label}: {h:02}:{m:02}:{s:02}", end="", flush=True)
            time.sleep(1)

    def __enter__(self):
        self._running = True
        self._start = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        h, rem = divmod(int(time.time() - self._start), 3600)
        m, s = divmod(rem, 60)
        print(f"{CLR}Total runtime: {h:02}:{m:02}:{s:02}")
        return False   # never suppress exceptions from the wrapped block


def load_classification():
    """Load data and split by loan id so no borrower appears in both train and test."""
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET, *CLF_FEATURES])
    X, y, g = df[CLF_FEATURES], df[TARGET].astype(int), df[GROUP_COL]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                    random_state=RANDOM_STATE).split(X, y, groups=g))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[tr]


def tune(estimator, param_dist, Xtr, ytr, gtr, n_iter=25, scoring="average_precision"):
    """Group-aware randomized hyperparameter search on the TRAIN set only."""
    search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scoring,
                                cv=sgkf, n_jobs=-1, refit=True, random_state=RANDOM_STATE)
    search.fit(Xtr, ytr, groups=gtr)
    return search.best_estimator_, search.best_params_


def cv_evaluate_classifier(name, model, Xtr, ytr, groups_tr,
                           results_csv=CV_CSV, folds_csv=FOLDS_CSV):
    """PHASE 1 (every model): grouped-CV PR-AUC & ROC-AUC. NEVER touches the test set."""
    res = cross_validate(model, Xtr, ytr, groups=groups_tr, cv=sgkf,
                         scoring=["average_precision", "roc_auc"], n_jobs=-1)
    pr, auc = res["test_average_precision"], res["test_roc_auc"]

    # Persist the per-fold scores (long format) for distribution / box plots.
    pd.DataFrame({"model": name, "fold": range(1, len(pr) + 1),
                  "PR_AUC": pr, "ROC_AUC": auc}).to_csv(
        folds_csv, mode="a", header=not os.path.exists(folds_csv), index=False)

    row = {"model": name,
           "CV_PR_AUC_mean": pr.mean(),  "CV_PR_AUC_std": pr.std(),
           "CV_ROC_AUC_mean": auc.mean(), "CV_ROC_AUC_std": auc.std(),
           "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    pd.DataFrame([row]).to_csv(results_csv, mode="a",
                               header=not os.path.exists(results_csv), index=False)
    print(f"{CLR}[CV] {name:22s} PR-AUC {pr.mean():.4f} +/- {pr.std():.4f} | "
          f"ROC-AUC {auc.mean():.4f} +/- {auc.std():.4f}")
    return row


def best_threshold_from_pr(y_true, y_prob):
    """Pick the probability cutoff that maximizes F1 on a precision-recall sweep.

    At a 2.45% base rate the default 0.5 cutoff classifies everything as negative
    (F1 = 0), even though the model ranks risk well. This finds the operating point
    that actually balances precision and recall. Returns (threshold, precision,
    recall, f1) at that point.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one fewer threshold than prec/rec -> drop last.
    p, r = prec[:-1], rec[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.argmax(f1))
    return thr[i], p[i], r[i], f1[i]


def final_test_classifier(name, model, Xtr, Xte, ytr, yte, groups_tr):
    """PHASE 2 (winner only, ONCE): the single unbiased held-out performance number.

    The decision threshold is tuned on TRAIN out-of-fold probabilities (grouped CV,
    so no leakage), then applied once to the test set -- so F1/precision/recall are
    reported at a sensible operating point instead of the meaningless 0.5 cutoff.
    """
    # Out-of-fold train probabilities -> honest threshold (model never sees the row).
    oof = cross_val_predict(model, Xtr, ytr, groups=groups_tr, cv=sgkf,
                            method="predict_proba", n_jobs=-1)[:, 1]
    thr, p_tr, r_tr, f1_tr = best_threshold_from_pr(ytr, oof)

    # Fit on full train, evaluate once on test at the tuned threshold.
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= thr).astype(int)
    out = {"model": name,
           "threshold": thr,
           "test_PR_AUC": average_precision_score(yte, prob),
           "test_ROC_AUC": roc_auc_score(yte, prob),
           "test_F1": f1_score(yte, pred, zero_division=0),
           "test_precision": precision_score(yte, pred, zero_division=0),
           "test_recall": recall_score(yte, pred, zero_division=0)}
    print(f"{CLR}\n[FINAL TEST - winner] {name}")
    print(f"  Tuned threshold : {thr:.4f}  "
          f"(train OOF: P={p_tr:.3f} R={r_tr:.3f} F1={f1_tr:.3f})")
    print(f"  PR-AUC          : {out['test_PR_AUC']:.4f}")
    print(f"  ROC-AUC         : {out['test_ROC_AUC']:.4f}")
    print(f"  F1   @threshold : {out['test_F1']:.4f}")
    print(f"  Prec @threshold : {out['test_precision']:.4f}")
    print(f"  Rec  @threshold : {out['test_recall']:.4f}")
    return out


# ======================================================================
# REGRESSION TRACK  (Loss_Data.csv -> lgd_time)  -- no grouping needed
# ======================================================================
def load_regression():
    """Load Loss_Data and make a plain random holdout (1 row per loan -> no grouping)."""
    df = pd.read_csv(REG_DATA_PATH).dropna(subset=[REG_TARGET, *REG_FEATURES])
    X, y = df[REG_FEATURES], df[REG_TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def tune_regressor(estimator, param_dist, Xtr, ytr, n_iter=25, scoring="r2"):
    """Randomized hyperparameter search on the TRAIN set only, scored by R^2."""
    search = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, scoring=scoring,
                                cv=kf, n_jobs=-1, refit=True, random_state=RANDOM_STATE)
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


def cv_evaluate_regressor(name, model, Xtr, ytr, results_csv=CV_CSV_REG, folds_csv=FOLDS_CSV_REG):
    """PHASE 1 (every model): 5-fold CV R^2 & RMSE. NEVER touches the test set."""
    res = cross_validate(model, Xtr, ytr, cv=kf,
                         scoring=["r2", "neg_root_mean_squared_error"], n_jobs=-1)
    r2 = res["test_r2"]
    nrmse = res["test_neg_root_mean_squared_error"]   # negative RMSE (higher = better)

    # Per-fold long format for distribution / box plots (neg_RMSE so higher = better).
    pd.DataFrame({"model": name, "fold": range(1, len(r2) + 1),
                  "R2": r2, "neg_RMSE": nrmse}).to_csv(
        folds_csv, mode="a", header=not os.path.exists(folds_csv), index=False)

    row = {"model": name,
           "CV_R2_mean": r2.mean(),       "CV_R2_std": r2.std(),
           "CV_RMSE_mean": (-nrmse).mean(), "CV_RMSE_std": (-nrmse).std(),
           "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    pd.DataFrame([row]).to_csv(results_csv, mode="a",
                               header=not os.path.exists(results_csv), index=False)
    print(f"{CLR}[CV] {name:22s} R2 {r2.mean():.4f} +/- {r2.std():.4f} | "
          f"RMSE {(-nrmse).mean():.4f} +/- {(-nrmse).std():.4f}")
    return row


def final_test_regressor(name, model, Xtr, Xte, ytr, yte):
    """PHASE 2 (winner only, ONCE): the single unbiased held-out R^2 / RMSE / MAE."""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    out = {"model": name,
           "test_R2": r2_score(yte, pred),
           "test_RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
           "test_MAE": mean_absolute_error(yte, pred)}
    print(f"{CLR}\n[FINAL TEST - winner] {name}")
    print(f"  R^2  : {out['test_R2']:.4f}")
    print(f"  RMSE : {out['test_RMSE']:.4f}")
    print(f"  MAE  : {out['test_MAE']:.4f}")
    return out
