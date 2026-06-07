"""
Run the full classification comparison study end to end:

  1. CV PHASE  - tune every model on grouped CV; record PR-AUC mean +/- std.
                 The test set is NEVER touched here.
  2. LEADERBOARD - rank by grouped-CV PR-AUC; flag models whose +/- std interval
                   overlaps the leader's as statistical ties.
  3. FINAL TEST  - touch the held-out test set exactly ONCE, on the CV winner only.

Usage:  python run_study_classification.py
"""
import os
import pandas as pd

from ml_eval import (
    load_classification, cv_evaluate_classifier, final_test_classifier,
    CV_CSV, FOLDS_CSV, CLR, RuntimeTimer,
)
from classification_models import ALL_BUILDERS as BUILDERS   # (Xtr, ytr, gtr) -> (name, est)


def main():
    # Fresh run: start the CV results files from scratch.
    for f in (CV_CSV, FOLDS_CSV):
        if os.path.exists(f):
            os.remove(f)

    Xtr, Xte, ytr, yte, gtr = load_classification()
    # Leading newline so the live "Study runtime: 00:00:00" tick stays on its own line.
    print(f"\nTrain rows: {len(Xtr):,} | Test rows: {len(Xte):,} | "
          f"train default rate: {ytr.mean():.4f} | test default rate: {yte.mean():.4f}")
    print(f"Train loans: {gtr.nunique():,}\n")

    # ---------------- PHASE 1: CV every model (no test access) ----------------
    fitted = {}
    for build in BUILDERS:
        try:
            name, est = build(Xtr, ytr, gtr)
            cv_evaluate_classifier(name, est, Xtr, ytr, gtr)
            fitted[name] = est
        except Exception as e:
            print(f"[SKIP] {build.__name__}: {type(e).__name__}: {e}")

    # ---------------- PHASE 2: leaderboard (rank by CV PR-AUC) ----------------
    df = (pd.read_csv(CV_CSV)
            .sort_values("timestamp").drop_duplicates("model", keep="last")
            .sort_values("CV_PR_AUC_mean", ascending=False).reset_index(drop=True))
    lead_lo = df.loc[0, "CV_PR_AUC_mean"] - df.loc[0, "CV_PR_AUC_std"]
    df["tie_with_leader"] = (df["CV_PR_AUC_mean"] + df["CV_PR_AUC_std"]) >= lead_lo
    df.insert(0, "rank", range(1, len(df) + 1))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"{CLR}\n============== LEADERBOARD (grouped-CV PR-AUC, mean +/- std) ==============")
    print(df[["rank", "model", "CV_PR_AUC_mean", "CV_PR_AUC_std",
              "CV_ROC_AUC_mean", "tie_with_leader"]].to_string(index=False))
    ties = df.loc[df["tie_with_leader"], "model"].tolist()
    if len(ties) > 1:
        print(f"\nStatistical ties with the leader (overlapping +/- std): {ties}")

    # ---------------- PHASE 3: final test ONCE on the CV winner ----------------
    winner = df.loc[0, "model"]
    print(f"\nCV winner: {winner}")
    if winner in fitted:
        final_test_classifier(winner, fitted[winner], Xtr, Xte, ytr, yte, gtr)
    else:
        print("Winner estimator not in memory (was it skipped?). Re-run its module.")


if __name__ == "__main__":
    # Live HH:MM:SS counter ticks throughout the run and prints the total at the end
    # (and still stops cleanly if a model raises).
    with RuntimeTimer("Study runtime"):
        main()
