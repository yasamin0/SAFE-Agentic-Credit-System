# src/compliance.py

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.rga import compute_rga_curve
from src.rgr import compute_rgr_curve
from src.rge import compute_rge_feature_importance, compute_rge_curve


def _compute_topsis(df, metric_cols):
    """Compute TOPSIS scores. Higher metric values are better."""
    values = df[metric_cols].astype(float).values

    denom = np.sqrt((values ** 2).sum(axis=0))
    denom[denom == 0] = 1.0

    normalized = values / denom

    ideal = normalized.max(axis=0)
    anti_ideal = normalized.min(axis=0)

    dist_to_ideal = np.sqrt(((normalized - ideal) ** 2).sum(axis=1))
    dist_to_anti = np.sqrt(((normalized - anti_ideal) ** 2).sum(axis=1))

    return dist_to_anti / (dist_to_ideal + dist_to_anti + 1e-12)


def compute_compliance_scores(metrics_df):
    """
    Compute compliance scores from AURGA, AURGR, and AURGE.

    Important:
    A weak or random model can sometimes look artificially strong in AURGR/AURGE
    because bad rankings may still remain stable under perturbation/removal.
    To avoid rewarding stable-but-useless models, compliance uses adjusted
    robustness/explainability values when AURGA is very weak.
    """
    df = metrics_df.copy()

    # Keep original paper metrics unchanged for reporting.
    # These columns still show the raw measured AURGA/AURGR/AURGE values.
    df["AURGR_for_compliance"] = df["AURGR"]
    df["AURGE_for_compliance"] = df["AURGE"]

    # Penalize models with weak rank-based accuracy.
    # This prevents Random Baseline or very weak models from getting high compliance
    # only because their predictions are stable.
    weak_mask = (df["Model"] == "Random Baseline") | (df["AURGA"] <= 0.55)

    df.loc[weak_mask, "AURGR_for_compliance"] = df.loc[
        weak_mask, ["AURGR", "AURGA"]
    ].min(axis=1)

    df.loc[weak_mask, "AURGE_for_compliance"] = df.loc[
        weak_mask, ["AURGE", "AURGA"]
    ].min(axis=1)

    metric_cols = [
        "AURGA",
        "AURGR_for_compliance",
        "AURGE_for_compliance",
    ]

    df["Compliance_Arithmetic"] = df[metric_cols].mean(axis=1)

    df["Compliance_Geometric"] = (
        df[metric_cols].clip(lower=1e-12).prod(axis=1) ** (1.0 / len(metric_cols))
    )

    df["Compliance_RMS"] = np.sqrt((df[metric_cols] ** 2).mean(axis=1))

    df["Compliance_TOPSIS"] = _compute_topsis(df, metric_cols)

    return df.sort_values("Compliance_TOPSIS", ascending=False).reset_index(drop=True)
def run_model_metric_comparison(
    all_model_paths,
    X_test,
    y_test,
    rgr_columns,
    random_state,
):
    """Compute AURGA, AURGR, and AURGE for all saved model candidates."""
    rows = []

    for model_name, model_path in all_model_paths.items():
        model = joblib.load(model_path)

        _, aurga = compute_rga_curve(model, X_test, y_test)

        _, aurgr_gaussian = compute_rgr_curve(
            model=model,
            X_test=X_test,
            perturbation_type="gaussian",
            columns=rgr_columns,
            random_state=random_state,
        )

        _, aurgr_swapping = compute_rgr_curve(
            model=model,
            X_test=X_test,
            perturbation_type="swapping",
            columns=rgr_columns,
            random_state=random_state,
        )

        rge_importance_df = compute_rge_feature_importance(model, X_test)
        _, aurge = compute_rge_curve(model, X_test, rge_importance_df)

        rows.append({
            "Model": model_name,
            "AURGA": float(aurga),
            "AURGR_Gaussian": float(aurgr_gaussian),
            "AURGR_Swapping": float(aurgr_swapping),
            "AURGR": float(np.mean([aurgr_gaussian, aurgr_swapping])),
            "AURGE": float(aurge),
        })

    return pd.DataFrame(rows)


def save_compliance_plot(compliance_df, output_path):
    """Save a horizontal bar chart of TOPSIS compliance scores."""
    plot_df = compliance_df.sort_values("Compliance_TOPSIS", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Model"], plot_df["Compliance_TOPSIS"])
    plt.xlabel("Compliance TOPSIS Score")
    plt.ylabel("Model")
    plt.title("SAFE AI Compliance Score Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()