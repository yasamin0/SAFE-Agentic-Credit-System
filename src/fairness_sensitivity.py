# src/fairness_sensitivity.py

import pandas as pd

from src.paths import (
    CROSSVAL_SAFE_FOLDS_CSV_PATH,
    TAIWAN_EXTERNAL_FOLDS_CSV_PATH,
)


FAIRNESS_WEIGHT_SCENARIOS = {
    "Equal weights": {
        "SPD": 0.25,
        "EOD": 0.25,
        "AOD": 0.25,
        "DIR": 0.25,
    },
    "SPD priority": {
        "SPD": 0.40,
        "EOD": 0.20,
        "AOD": 0.20,
        "DIR": 0.20,
    },
    "EOD priority": {
        "SPD": 0.20,
        "EOD": 0.40,
        "AOD": 0.20,
        "DIR": 0.20,
    },
    "AOD priority": {
        "SPD": 0.20,
        "EOD": 0.20,
        "AOD": 0.40,
        "DIR": 0.20,
    },
    "DIR priority": {
        "SPD": 0.20,
        "EOD": 0.20,
        "AOD": 0.20,
        "DIR": 0.40,
    },
}


def _calculate_fairness(
    spd,
    eod,
    aod,
    dir_score,
    weights,
):
    return float(
        weights["SPD"] * spd
        + weights["EOD"] * eod
        + weights["AOD"] * aod
        + weights["DIR"] * dir_score
    )


def _analyse_dataset(
    df,
    dataset_name,
):
    required = [
        "fold",
        "fairness_score_spd",
        "fairness_score_eod",
        "fairness_score_aod",
        "fairness_score_dir",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} fold file does not contain "
            f"the fairness components required for sensitivity analysis: "
            f"{missing}"
        )

    rows = []

    for _, fold in df.iterrows():

        for scenario, weights in FAIRNESS_WEIGHT_SCENARIOS.items():

            score = _calculate_fairness(
                spd=float(
                    fold["fairness_score_spd"]
                ),
                eod=float(
                    fold["fairness_score_eod"]
                ),
                aod=float(
                    fold["fairness_score_aod"]
                ),
                dir_score=float(
                    fold["fairness_score_dir"]
                ),
                weights=weights,
            )

            rows.append({
                "dataset": dataset_name,
                "fold": int(
                    fold["fold"]
                ),
                "scenario": scenario,
                "w_spd": weights["SPD"],
                "w_eod": weights["EOD"],
                "w_aod": weights["AOD"],
                "w_dir": weights["DIR"],
                "fairness_aggregate": score,
            })

    return pd.DataFrame(rows)


def run_fairness_sensitivity():

    german = pd.read_csv(
        CROSSVAL_SAFE_FOLDS_CSV_PATH
    )

    taiwan = pd.read_csv(
        TAIWAN_EXTERNAL_FOLDS_CSV_PATH
    )

    german_results = _analyse_dataset(
        german,
        "German Credit",
    )

    taiwan_results = _analyse_dataset(
        taiwan,
        "Taiwan Credit",
    )

    results = pd.concat(
        [
            german_results,
            taiwan_results,
        ],
        ignore_index=True,
    )

    summary = (
        results
        .groupby(
            ["dataset", "scenario"],
            as_index=False,
        )
        .agg(
            mean_fairness=(
                "fairness_aggregate",
                "mean",
            ),
            std_fairness=(
                "fairness_aggregate",
                "std",
            ),
            min_fairness=(
                "fairness_aggregate",
                "min",
            ),
            max_fairness=(
                "fairness_aggregate",
                "max",
            ),
        )
    )

    equal_scores = (
        summary[
            summary["scenario"]
            == "Equal weights"
        ][
            ["dataset", "mean_fairness"]
        ]
        .rename(
            columns={
                "mean_fairness":
                "equal_mean_fairness"
            }
        )
    )

    summary = summary.merge(
        equal_scores,
        on="dataset",
        how="left",
    )

    summary["delta_vs_equal"] = (
        summary["mean_fairness"]
        - summary["equal_mean_fairness"]
    )

    summary = summary.drop(
        columns=[
            "equal_mean_fairness"
        ]
    )

    # --------------------------------------------------------
    # Stability range
    # --------------------------------------------------------

    stability_rows = []

    for dataset_name in (
        summary["dataset"]
        .unique()
    ):

        subset = summary[
            summary["dataset"]
            == dataset_name
        ]

        highest = subset.loc[
            subset[
                "mean_fairness"
            ].idxmax()
        ]

        lowest = subset.loc[
            subset[
                "mean_fairness"
            ].idxmin()
        ]

        stability_rows.append({
            "dataset": dataset_name,
            "highest_scenario": (
                highest["scenario"]
            ),
            "highest_mean_fairness": float(
                highest["mean_fairness"]
            ),
            "lowest_scenario": (
                lowest["scenario"]
            ),
            "lowest_mean_fairness": float(
                lowest["mean_fairness"]
            ),
            "range_across_weights": float(
                highest["mean_fairness"]
                - lowest["mean_fairness"]
            ),
        })

    stability = pd.DataFrame(
        stability_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_path = (
        CROSSVAL_SAFE_FOLDS_CSV_PATH
        .parent
        / "fairness_weight_sensitivity.csv"
    )

    summary_path = (
        CROSSVAL_SAFE_FOLDS_CSV_PATH
        .parent
        / "fairness_weight_sensitivity_summary.csv"
    )

    report_path = (
        CROSSVAL_SAFE_FOLDS_CSV_PATH
        .parent
        / "fairness_weight_sensitivity_report.md"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    report = f"""
# Fairness Aggregate Weight Sensitivity

The Fairness Aggregate was recalculated under five alternative
weighting policies for SPD, EOD, AOD, and DIR.

No predictive model was retrained. The analysis isolates the effect
of the fairness aggregation policy.

## Summary

{summary.to_markdown(index=False)}

## Stability

{stability.to_markdown(index=False)}

The `delta_vs_equal` column measures the change relative to the
equal-weight fairness specification.

The range across weighting scenarios quantifies the sensitivity of
the Fairness Aggregate to alternative fairness priorities.
"""

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FAIRNESS WEIGHT SENSITIVITY")
    print("=" * 70)

    print("\nSummary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nStability:")
    print(
        stability.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(report_path)

    return results, summary, stability


if __name__ == "__main__":
    run_fairness_sensitivity()