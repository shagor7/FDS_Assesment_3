"""Task 2.2: grouped cross-validation for team goals scored prediction."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "regression_2_2.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FEATURES = [
    "recent_gf_avg", "recent_ga_avg", "recent_win_rate",
    "recent_goal_diff_avg", "recent_points_avg", "rest_days",
    "host_team_indicator", "knockout_stage",
]
TARGET = "goals_scored"
GROUP = "match_id"


def main():
    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET, GROUP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 208:
        warnings.warn(f"Expected 208 rows; found {len(df)}")
    if df[required].isna().any().any():
        raise ValueError("Missing values found in predictors, target, or match_id. Resolve before modelling.")
    if len(FEATURES) != 8:
        raise AssertionError("Task 2.2 must use exactly eight predictors.")
    if df.groupby(GROUP).size().ne(2).any():
        raise ValueError("Expected exactly two team observations per match.")

    X = df[FEATURES]
    y = df[TARGET].astype(float)
    groups = df[GROUP]
    cv = GroupKFold(n_splits=5)
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "Ridge (alpha=1)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Ridge (alpha=10)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Ridge (alpha=100)": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
    }
    rows = []
    prediction_store = {}
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=None)
        prediction_store[name] = preds
        rows.append({
            "model": name,
            "mae": mean_absolute_error(y, preds),
            "rmse": mean_squared_error(y, preds) ** 0.5,
            "r2": r2_score(y, preds),
            "rows": len(df),
            "unique_matches": groups.nunique(),
            "cv": "5-fold GroupKFold by match_id",
        })
    result = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    result.to_csv(RESULTS / "model_comparison.csv", index=False)
    with (RESULTS / "model_comparison.txt").open("w", encoding="utf-8") as f:
        f.write("Task 2.2 — Goals scored model comparison\n")
        f.write("Evaluation: out-of-fold predictions from 5-fold GroupKFold grouped by match_id.\n")
        f.write("Metrics are computed across all out-of-fold predictions; values may differ from mean fold metrics.\n\n")
        f.write(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        f.write("\n\nPredictors: " + ", ".join(FEATURES) + "\n")
        f.write("Target: goals_scored\n")
        f.write("Caution: cross-validation performance is exploratory; verify data provenance and predictor restrictions.\n")

    best_name = result.iloc[0]["model"]
    best_pred = prediction_store[best_name]
    pred_df = df[[GROUP, "date", "team", "opponent", TARGET]].copy()
    pred_df["predicted_goals_best_model"] = best_pred
    pred_df["residual"] = pred_df[TARGET] - pred_df["predicted_goals_best_model"]
    pred_df.to_csv(RESULTS / "out_of_fold_predictions.csv", index=False)

    # Goals distribution
    counts = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(7, 4.5))
    plt.bar(counts.index.astype(int), counts.values)
    plt.title("Distribution of Goals Scored per Team")
    plt.xlabel("Goals scored")
    plt.ylabel("Number of team-match observations")
    plt.xticks(counts.index.astype(int))
    plt.tight_layout()
    plt.savefig(FIGURES / "goals_distribution.png", dpi=180)
    plt.close()

    # Actual vs predicted for the best model
    plt.figure(figsize=(6, 5))
    plt.scatter(y, best_pred, alpha=0.75)
    lo = min(float(y.min()), float(np.min(best_pred)))
    hi = max(float(y.max()), float(np.max(best_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.title(f"Actual vs Predicted Goals ({best_name})")
    plt.xlabel("Actual goals")
    plt.ylabel("Out-of-fold predicted goals")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=180)
    plt.close()

    # Residual plot
    residuals = y.to_numpy() - best_pred
    plt.figure(figsize=(6, 4.5))
    plt.scatter(best_pred, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.title(f"Residuals vs Predicted Goals ({best_name})")
    plt.xlabel("Out-of-fold predicted goals")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(FIGURES / "residual_plot.png", dpi=180)
    plt.close()

    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results to {RESULTS}")
    print(f"Saved figures to {FIGURES}")

if __name__ == "__main__":
    main()
