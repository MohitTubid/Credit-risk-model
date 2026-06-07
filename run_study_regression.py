"""
Run the full REGRESSION comparison study (Loss_Data.csv -> lgd_time) end to end:

  1. CV PHASE   - tune every model on 5-fold CV; record R^2 mean +/- std (and RMSE).
                  The test set is NEVER touched here.
  2. LEADERBOARD - rank by CV R^2; flag models whose +/- std interval overlaps the
                   leader's as statistical ties.
  3. FINAL TEST  - touch the held-out test set exactly ONCE, on the CV winner only.

Usage:  python run_study_regression.py
Then:   python plot_cv_distributions.py   (lights up the Regression page)
"""
import os
import pandas as pd

from ml_eval import (
    load_regression, cv_evaluate_regressor, final_test_regressor,
    CV_CSV_REG, FOLDS_CSV_REG, CLR, RuntimeTimer,
)
from regression_models import ALL_BUILDERS


def main():
    # Fresh run: start the regression CV files from scratch.
    for f in (CV_CSV_REG, FOLDS_CSV_REG):
        if os.path.exists(f):
            os.remove(f)

    Xtr, Xte, ytr, yte = load_regression()
    print(f"\nTrain rows: {len(Xtr):,} | Test rows: {len(Xte):,} | "
          f"lgd_time mean: train {ytr.mean():.4f} / test {yte.mean():.4f}")
    print(f"(Loss_Data is cross-sectional -> plain KFold, no grouping)\n")

    # ---------------- PHASE 1: CV every model (no test access) ----------------
    fitted = {}
    for build in ALL_BUILDERS:
        try:
            name, est = build(Xtr, ytr)
            cv_evaluate_regressor(name, est, Xtr, ytr)
            fitted[name] = est
        except Exception as e:
            print(f"[SKIP] {build.__name__}: {type(e).__name__}: {e}")

    # ---------------- PHASE 2: leaderboard (rank by CV R^2) ----------------
    df = (pd.read_csv(CV_CSV_REG)
            .sort_values("timestamp").drop_duplicates("model", keep="last")
            .sort_values("CV_R2_mean", ascending=False).reset_index(drop=True))
    lead_lo = df.loc[0, "CV_R2_mean"] - df.loc[0, "CV_R2_std"]
    df["tie_with_leader"] = (df["CV_R2_mean"] + df["CV_R2_std"]) >= lead_lo
    df.insert(0, "rank", range(1, len(df) + 1))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"{CLR}\n============== LEADERBOARD (5-fold CV R^2, mean +/- std) ==============")
    print(df[["rank", "model", "CV_R2_mean", "CV_R2_std",
              "CV_RMSE_mean", "tie_with_leader"]].to_string(index=False))
    ties = df.loc[df["tie_with_leader"], "model"].tolist()
    if len(ties) > 1:
        print(f"\nStatistical ties with the leader (overlapping +/- std): {ties}")

    # ---------------- PHASE 3: final test ONCE on the CV winner ----------------
    winner = df.loc[0, "model"]
    print(f"\nCV winner: {winner}")
    if winner in fitted:
        final_test_regressor(winner, fitted[winner], Xtr, Xte, ytr, yte)
    else:
        print("Winner estimator not in memory (was it skipped?). Re-run its module.")


if __name__ == "__main__":
    with RuntimeTimer("Regression study runtime"):
        main()
