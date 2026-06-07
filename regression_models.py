"""
The 9 regression models for the lgd_time study (Loss_Data.csv), each tuned on the
shared 5-fold CV by R^2. Mirrors the classification roster:

  Linear Regression, Random Forest, Bagging, AdaBoost, HistGradientBoosting,
  XGBoost, LightGBM, CatBoost, SVR (RBF)

Each builder: (Xtr, ytr) -> (name, tuned_estimator).  No groups (cross-sectional data).
Small n (~977 train) so kernel SVR is fine (no Nystroem approximation needed), and tuning
budgets are modest to avoid overfitting.
"""
from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_eval import tune_regressor, load_regression, cv_evaluate_regressor, RANDOM_STATE


def build_linear(Xtr, ytr):
    # Plain OLS: nothing to tune (scale-invariant in its predictions).
    return "Linear Regression", LinearRegression()


def build_rf(Xtr, ytr):
    rf = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
    dist = {"n_estimators": randint(200, 500), "max_depth": [4, 6, 8, None],
            "min_samples_leaf": randint(2, 30), "max_features": ["sqrt", "log2", 1.0]}
    est, _ = tune_regressor(rf, dist, Xtr, ytr, n_iter=25)
    return "Random Forest", est


def build_bagging(Xtr, ytr):
    bag = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
                           n_jobs=-1, random_state=RANDOM_STATE)
    dist = {"n_estimators": randint(50, 200), "max_samples": uniform(0.5, 0.5),
            "max_features": uniform(0.5, 0.5), "estimator__max_depth": [None, 6, 12]}
    est, _ = tune_regressor(bag, dist, Xtr, ytr, n_iter=15)
    return "Bagging", est


def build_adaboost(Xtr, ytr):
    ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
                            random_state=RANDOM_STATE)
    dist = {"n_estimators": randint(100, 500), "learning_rate": uniform(0.01, 1.0),
            "loss": ["linear", "square", "exponential"],
            "estimator__max_depth": [2, 3, 4]}
    est, _ = tune_regressor(ada, dist, Xtr, ytr, n_iter=25)
    return "AdaBoost", est


def build_histgb(Xtr, ytr):
    hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    dist = {"learning_rate": uniform(0.01, 0.3), "max_iter": randint(150, 600),
            "max_leaf_nodes": randint(15, 63), "min_samples_leaf": randint(10, 60),
            "l2_regularization": uniform(0.0, 1.0)}
    est, _ = tune_regressor(hgb, dist, Xtr, ytr, n_iter=25)
    return "HistGradientBoosting", est


def build_xgb(Xtr, ytr):
    from xgboost import XGBRegressor
    xgb = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                       n_jobs=-1, random_state=RANDOM_STATE)
    dist = {"n_estimators": randint(150, 600), "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(2, 7), "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4), "min_child_weight": randint(1, 10),
            "reg_lambda": uniform(0.0, 2.0)}
    est, _ = tune_regressor(xgb, dist, Xtr, ytr, n_iter=25)
    return "XGBoost", est


def build_lgbm(Xtr, ytr):
    from lightgbm import LGBMRegressor
    lgbm = LGBMRegressor(objective="regression", subsample_freq=1,
                         n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    dist = {"n_estimators": randint(150, 600), "learning_rate": uniform(0.01, 0.3),
            "num_leaves": randint(15, 100), "max_depth": [-1, 4, 8, 12],
            "min_child_samples": randint(10, 60), "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": uniform(0.0, 2.0)}
    est, _ = tune_regressor(lgbm, dist, Xtr, ytr, n_iter=25)
    return "LightGBM", est


def build_catboost(Xtr, ytr):
    from catboost import CatBoostRegressor
    cat = CatBoostRegressor(loss_function="RMSE", bootstrap_type="Bernoulli",
                            random_seed=RANDOM_STATE, thread_count=-1,
                            allow_writing_files=False, verbose=False)
    dist = {"iterations": randint(150, 600), "learning_rate": uniform(0.01, 0.3),
            "depth": randint(4, 10), "l2_leaf_reg": uniform(1.0, 9.0),
            "subsample": uniform(0.6, 0.4), "rsm": uniform(0.6, 0.4)}
    est, _ = tune_regressor(cat, dist, Xtr, ytr, n_iter=25)
    return "CatBoost", est


def build_svr(Xtr, ytr):
    # Plain RBF SVR is feasible at n~977. Scaling matters -> pipeline.
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svr", SVR(kernel="rbf"))])
    dist = {"svr__C": loguniform(1e-1, 1e3), "svr__gamma": loguniform(1e-4, 1e0),
            "svr__epsilon": uniform(0.01, 0.3)}
    est, _ = tune_regressor(pipe, dist, Xtr, ytr, n_iter=20)
    return "SVR (RBF)", est


ALL_BUILDERS = [build_linear, build_rf, build_bagging, build_adaboost, build_histgb,
                build_xgb, build_lgbm, build_catboost, build_svr]

# Lookup by the name each builder returns -> lets other scripts rebuild a chosen model
# (e.g. the leaderboard winner) without re-running the whole study. Re-tuning is
# deterministic (fixed random_state), so this reproduces the exact study estimator.
BUILDER_BY_NAME = {
    "Linear Regression": build_linear,
    "Random Forest": build_rf,
    "Bagging": build_bagging,
    "AdaBoost": build_adaboost,
    "HistGradientBoosting": build_histgb,
    "XGBoost": build_xgb,
    "LightGBM": build_lgbm,
    "CatBoost": build_catboost,
    "SVR (RBF)": build_svr,
}


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = load_regression()
    for build in ALL_BUILDERS:
        try:
            name, est = build(Xtr, ytr)
            cv_evaluate_regressor(name, est, Xtr, ytr)
        except Exception as e:
            print(f"[SKIP] {build.__name__}: {type(e).__name__}: {e}")
