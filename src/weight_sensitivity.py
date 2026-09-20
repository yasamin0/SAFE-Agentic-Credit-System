# src/weight_sensitivity.py

import numpy as np
import pandas as pd

from src.config import APPROVAL_THRESHOLD

from src.paths import (
    CROSSVAL_SAFE_FOLDS_CSV_PATH,
    TAIWAN_EXTERNAL_FOLDS_CSV_PATH,
    WEIGHT_SENSITIVITY_CSV_PATH,
    WEIGHT_SENSITIVITY_SUMMARY_CSV_PATH,
    WEIGHT_SENSITIVITY_REPORT_PATH,
)


# ------------------------------------------------------------
# Weight scenarios
# ------------------------------------------------------------

WEIGHT_SCENARIOS = {
    "Equal weights": {
        "RGA": 0.25,
        "RGR": 0.25,
        "RGE": 0.25,
        "Fairness": 0.25,
    },

    "RGA priority": {
        "RGA": 0.40,
        "RGR": 0.20,
        "RGE": 0.20,
        "Fairness": 0.20,
    },

    "RGR priority": {
        "RGA": 0.20,
        "RGR": 0.40,
        "RGE": 0.20,
        "Fairness": 0.20,
    },

    "RGE priority": {
        "RGA": 0.20,
        "RGR": 0.20,
        "RGE": 0.40,
        "Fairness": 0.20,
    },

    "Fairness priority": {
        "RGA": 0.20,
        "RGR": 0.20,
        "RGE": 0.20,
        "Fairness": 0.40,
    },
}


# ------------------------------------------------------------
# SAFE score
# ------------------------------------------------------------

def _calculate_safe_score(
    aurga,
    rgr_aggregate,
    aurge,
    fairness_aggregate,
    weights,
):
    """
    Calculate SAFE using a specified policy-weight scenario.
    """

    return float(
        weights["RGA"] * aurga
        + weights["RGR"] * rgr_aggregate
        + weights["RGE"] * aurge
        + weights["Fairness"] * fairness_aggregate
    )


# ------------------------------------------------------------
# Validate input
# ------------------------------------------------------------

def _validate_fold_dataframe(
    df,
    dataset_name,
):
    required_columns = [
        "fold",
        "aurga",
        "rgr_aggregate",
        "aurge",
        "fairness_aggregate",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} fold file is missing columns: "
            f"{missing}"
        )


# ------------------------------------------------------------
# Dataset sensitivity
# ------------------------------------------------------------

def _evaluate_dataset(
    df,
    dataset_name,
):
    """
    Recalculate SAFE for every fold under each weighting scenario.
    """

    _validate_fold_dataframe(
        df=df,
        dataset_name=dataset_name,
    )

    rows = []

    for _, fold_row in df.iterrows():

        for scenario_name, weights in WEIGHT_SCENARIOS.items():

            safe_score = _calculate_safe_score(
                aurga=float(
                    fold_row["aurga"]
                ),
                rgr_aggregate=float(
                    fold_row["rgr_aggregate"]
                ),
                aurge=float(
                    fold_row["aurge"]
                ),
                fairness_aggregate=float(
                    fold_row["fairness_aggregate"]
                ),
                weights=weights,
            )

            rows.append({
                "dataset": dataset_name,
                "fold": int(
                    fold_row["fold"]
                ),
                "scenario": scenario_name,

                "w_rga": weights["RGA"],
                "w_rgr": weights["RGR"],
                "w_rge": weights["RGE"],
                "w_fairness": weights["Fairness"],

                "aurga": float(
                    fold_row["aurga"]
                ),

                "rgr_aggregate": float(
                    fold_row["rgr_aggregate"]
                ),

                "aurge": float(
                    fold_row["aurge"]
                ),

                "fairness_aggregate": float(
                    fold_row["fairness_aggregate"]
                ),

                "safe_score": safe_score,

                "decision": (
                    "APPROVED"
                    if safe_score >= APPROVAL_THRESHOLD
                    else "REJECTED"
                ),
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def _build_summary(
    results_df,
):
    summary_rows = []

    datasets = (
        results_df["dataset"]
        .unique()
        .tolist()
    )

    for dataset_name in datasets:

        dataset_df = results_df[
            results_df["dataset"]
            == dataset_name
        ]

        baseline_df = dataset_df[
            dataset_df["scenario"]
            == "Equal weights"
        ]

        baseline_mean = float(
            baseline_df[
                "safe_score"
            ].mean()
        )

        for scenario_name in WEIGHT_SCENARIOS:

            scenario_df = dataset_df[
                dataset_df["scenario"]
                == scenario_name
            ]

            scores = scenario_df[
                "safe_score"
            ].astype(float)

            mean_score = float(
                scores.mean()
            )

            std_score = float(
                scores.std(ddof=1)
            )

            min_score = float(
                scores.min()
            )

            max_score = float(
                scores.max()
            )

            approved_folds = int(
                (
                    scenario_df["decision"]
                    == "APPROVED"
                ).sum()
            )

            rejected_folds = int(
                (
                    scenario_df["decision"]
                    == "REJECTED"
                ).sum()
            )

            summary_rows.append({
                "dataset": dataset_name,
                "scenario": scenario_name,
                "mean_safe": mean_score,
                "std_safe": std_score,
                "min_safe": min_score,
                "max_safe": max_score,
                "delta_vs_equal": (
                    mean_score
                    - baseline_mean
                ),
                "approved_folds": approved_folds,
                "rejected_folds": rejected_folds,
            })

    return pd.DataFrame(
        summary_rows
    )


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def run_weight_sensitivity():
    """
    Evaluate the effect of alternative SAFE policy weights
    on both German Credit and Taiwan Credit results.
    """

    if not CROSSVAL_SAFE_FOLDS_CSV_PATH.exists():
        raise FileNotFoundError(
            "German 5-fold SAFE results not found at: "
            f"{CROSSVAL_SAFE_FOLDS_CSV_PATH}"
        )

    if not TAIWAN_EXTERNAL_FOLDS_CSV_PATH.exists():
        raise FileNotFoundError(
            "Taiwan 5-fold SAFE results not found at: "
            f"{TAIWAN_EXTERNAL_FOLDS_CSV_PATH}"
        )

    german_df = pd.read_csv(
        CROSSVAL_SAFE_FOLDS_CSV_PATH
    )

    taiwan_df = pd.read_csv(
        TAIWAN_EXTERNAL_FOLDS_CSV_PATH
    )

    german_results = _evaluate_dataset(
        df=german_df,
        dataset_name="German Credit",
    )

    taiwan_results = _evaluate_dataset(
        df=taiwan_df,
        dataset_name="Taiwan Credit",
    )

    results_df = pd.concat(
        [
            german_results,
            taiwan_results,
        ],
        ignore_index=True,
    )

    summary_df = _build_summary(
        results_df
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    results_df.to_csv(
        WEIGHT_SENSITIVITY_CSV_PATH,
        index=False,
    )

    summary_df.to_csv(
        WEIGHT_SENSITIVITY_SUMMARY_CSV_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Additional stability statistics
    # --------------------------------------------------------

    stability_rows = []

    for dataset_name in [
        "German Credit",
        "Taiwan Credit",
    ]:

        dataset_summary = summary_df[
            summary_df["dataset"]
            == dataset_name
        ].copy()

        best_row = dataset_summary.loc[
            dataset_summary[
                "mean_safe"
            ].idxmax()
        ]

        worst_row = dataset_summary.loc[
            dataset_summary[
                "mean_safe"
            ].idxmin()
        ]

        safe_range = float(
            best_row["mean_safe"]
            - worst_row["mean_safe"]
        )

        stability_rows.append({
            "dataset": dataset_name,
            "highest_scenario": (
                best_row["scenario"]
            ),
            "highest_mean_safe": float(
                best_row["mean_safe"]
            ),
            "lowest_scenario": (
                worst_row["scenario"]
            ),
            "lowest_mean_safe": float(
                worst_row["mean_safe"]
            ),
            "range_across_weights": (
                safe_range
            ),
        })

    stability_df = pd.DataFrame(
        stability_rows
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    report = f"""
# SAFE Weight Sensitivity Analysis

The SAFE score was recalculated under five policy-weighting scenarios
without retraining the predictive models.

The purpose of this analysis is to evaluate whether the composite SAFE
score is highly dependent on the original equal-weight assumption.

## Weighting Scenarios

| Scenario | RGA | RGR | RGE | Fairness |
|---|---:|---:|---:|---:|
| Equal weights | 0.25 | 0.25 | 0.25 | 0.25 |
| RGA priority | 0.40 | 0.20 | 0.20 | 0.20 |
| RGR priority | 0.20 | 0.40 | 0.20 | 0.20 |
| RGE priority | 0.20 | 0.20 | 0.40 | 0.20 |
| Fairness priority | 0.20 | 0.20 | 0.20 | 0.40 |

## Summary Results

{summary_df.to_markdown(index=False)}

## Stability Across Weighting Scenarios

{stability_df.to_markdown(index=False)}

The `delta_vs_equal` column reports the change in mean SAFE score
relative to the original equal-weight specification.

The range between the highest and lowest mean SAFE scores provides a
simple measure of sensitivity to alternative governance priorities.

Approval decisions were evaluated using the same SAFE approval
threshold used in the main framework.
"""

    WEIGHT_SENSITIVITY_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAFE WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)

    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nStability:")
    print(
        stability_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        WEIGHT_SENSITIVITY_CSV_PATH
    )

    print(
        WEIGHT_SENSITIVITY_SUMMARY_CSV_PATH
    )

    print(
        WEIGHT_SENSITIVITY_REPORT_PATH
    )

    return (
        results_df,
        summary_df,
        stability_df,
    )


if __name__ == "__main__":
    run_weight_sensitivity()