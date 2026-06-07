"""
Feature-importance plots for the CLASSIFICATION winner (whichever model tops
cv_classification.csv).

Primary method is PERMUTATION IMPORTANCE -- model-agnostic, which matters here because
the usual winner (SVM via Nystroem + calibration) has no native feature_importances_.
It measures the drop in held-out test PR-AUC (average_precision) when each feature is
shuffled. If the winner DOES expose a clean native importance over the 5 raw features
(tree models), that is shown as a 2nd panel.

The winner is rebuilt via its builder (deterministic re-tune), reproducing the study model.
Also writes feature_importance_classification.csv for combined_report.py.

Run AFTER run_study_classification.py.   Usage: python feature_importance_classification.py
Headless: LEADERBOARD_SHOW=0 python feature_importance_classification.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml_eval import load_classification, CV_CSV, RANDOM_STATE
from classification_models import BUILDER_BY_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "feature_importance_classification.pdf")
PNG_PATH = os.path.join(HERE, "feature_importance_classification.png")
FI_CSV = os.path.join(HERE, "feature_importance_classification.csv")
SHOW = os.environ.get("LEADERBOARD_SHOW", "1") != "0"
SCORING = "average_precision"   # PR-AUC: the study's primary classification metric


def winner_name():
    if not os.path.exists(CV_CSV):
        sys.exit("cv_classification.csv not found. Run run_study_classification.py first.")
    df = (pd.read_csv(CV_CSV)
            .sort_values("timestamp").drop_duplicates("model", keep="last")
            .sort_values("CV_PR_AUC_mean", ascending=False).reset_index(drop=True))
    return df.loc[0, "model"]


def native_importance(model, n_features):
    """Native importance ONLY if it maps cleanly to the n_features raw inputs."""
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    vals = label = None
    if hasattr(est, "feature_importances_"):
        vals, label = np.asarray(est.feature_importances_), "Native feature_importances_"
    elif hasattr(est, "coef_"):
        vals, label = np.abs(np.ravel(est.coef_)), "|coef|"
    if vals is None or len(vals) != n_features:   # e.g. poly-expanded / pipeline transforms
        return None, None
    return vals, label


def main():
    name = winner_name()
    print(f"Classification winner: {name}  ->  refitting and computing importances...")
    Xtr, Xte, ytr, yte, gtr = load_classification()
    _, model = BUILDER_BY_NAME[name](Xtr, ytr, gtr)   # deterministic re-tune == study model
    model.fit(Xtr, ytr)
    features = list(Xtr.columns)

    r = permutation_importance(model, Xte, yte, scoring=SCORING,
                               n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    perm_mean, perm_std = r.importances_mean, r.importances_std

    # Persist for the combined report.
    pd.DataFrame({"winner": name, "feature": features,
                  "perm_mean": perm_mean, "perm_std": perm_std}).to_csv(FI_CSV, index=False)

    nat, nat_label = native_importance(model, len(features))
    npanels = 2 if nat is not None else 1
    fig, axes = plt.subplots(1, npanels, figsize=(max(9, 7 * npanels), 6), squeeze=False)
    axes = axes[0]

    order = np.argsort(perm_mean)
    axes[0].barh(np.array(features)[order], perm_mean[order], xerr=perm_std[order],
                 color="#5dade2", edgecolor="#34495e",
                 error_kw=dict(ecolor="#7f8c8d", capsize=3))
    axes[0].set_xlabel("Permutation importance (drop in test PR-AUC)")
    axes[0].set_title("Permutation importance (held-out test)")
    axes[0].grid(axis="x", alpha=0.3)

    if nat is not None:
        o2 = np.argsort(nat)
        axes[1].barh(np.array(features)[o2], nat[o2], color="#58d68d", edgecolor="#34495e")
        axes[1].set_xlabel(nat_label)
        axes[1].set_title("Native importance")
        axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(f"Default Classification - Feature Importance  (winner: {name})",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(fig)
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Saved: {PDF_PATH}\nSaved: {PNG_PATH}\nSaved: {FI_CSV}\n")
    print("Permutation importance (mean +/- std, drop in test PR-AUC):")
    for i in order[::-1]:
        print(f"  {features[i]:20s} {perm_mean[i]:+.4f} +/- {perm_std[i]:.4f}")

    if SHOW:
        try:
            if sys.platform.startswith("win"):
                os.startfile(PDF_PATH)             # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{PDF_PATH}"')
            else:
                os.system(f'xdg-open "{PDF_PATH}"')
        except Exception as e:
            print(f"(Could not auto-open PDF: {e})")
        plt.show()


if __name__ == "__main__":
    main()
