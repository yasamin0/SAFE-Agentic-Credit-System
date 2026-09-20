# src/statistical_tests.py

import numpy as np
import pandas as pd

from scipy.stats import cramervonmises_2samp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def bootstrap_auc_ci(
    y_true,
    y_probs,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
):
    """
    Compute a bootstrap confidence interval for ROC-AUC.

    Stratification is not forced inside each resample, so bootstrap samples
    containing only one outcome class are skipped.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have the same length.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        y_sample = y_true[indices]
        probs_sample = y_probs[indices]

        # ROC-AUC cannot be computed when only one class is present.
        if len(np.unique(y_sample)) < 2:
            continue

        score = roc_auc_score(
            y_sample,
            probs_sample,
        )

        bootstrap_scores.append(float(score))

    if not bootstrap_scores:
        raise ValueError(
            "No valid bootstrap samples were available for AUC estimation."
        )

    bootstrap_scores = np.asarray(bootstrap_scores)

    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            bootstrap_scores,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_scores,
            1.0 - alpha / 2.0,
        )
    )

    return {
        "estimate": float(roc_auc_score(y_true, y_probs)),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": float(confidence_level),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(len(bootstrap_scores)),
        "bootstrap_std": float(np.std(bootstrap_scores, ddof=1)),
    }


def bootstrap_metric_ci(
    values,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
):
    """
    Compute a bootstrap confidence interval for the mean of a metric.

    This is useful when the metric is available across folds, repeated runs,
    perturbation levels, or other repeated evaluations.
    """
    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError("No valid values were provided.")

    rng = np.random.default_rng(random_state)

    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means.append(
            float(np.mean(sample))
        )

    bootstrap_means = np.asarray(bootstrap_means)

    alpha = 1.0 - confidence_level

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return {
        "estimate": float(np.mean(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": float(confidence_level),
        "bootstrap_std": float(
            np.std(
                bootstrap_means,
                ddof=1,
            )
        ),
        "n_values": int(len(values)),
        "n_bootstrap": int(n_bootstrap),
    }


def cramervonmises_prediction_test(
    reference_probs,
    comparison_probs,
):
    """
    Compare two empirical predicted-probability distributions using the
    two-sample Cramér-von Mises test.

    H0:
    The two samples come from the same continuous distribution.

    A small p-value provides evidence that the prediction distributions differ.
    """
    reference_probs = np.asarray(
        reference_probs,
        dtype=float,
    )

    comparison_probs = np.asarray(
        comparison_probs,
        dtype=float,
    )

    reference_probs = reference_probs[
        np.isfinite(reference_probs)
    ]

    comparison_probs = comparison_probs[
        np.isfinite(comparison_probs)
    ]

    if len(reference_probs) == 0 or len(comparison_probs) == 0:
        raise ValueError(
            "Both probability samples must contain valid observations."
        )

    result = cramervonmises_2samp(
        reference_probs,
        comparison_probs,
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_0_05": bool(result.pvalue < 0.05),
        "n_reference": int(len(reference_probs)),
        "n_comparison": int(len(comparison_probs)),
    }

def cramervonmises_outcome_separation_test(
    y_true,
    y_probs,
):
    """
    Test whether predicted-probability distributions differ between
    the two observed outcome classes using the two-sample
    Cramer-von Mises test.

    H0:
    The predicted probabilities for y=0 and y=1 come from the
    same continuous distribution.

    H1:
    The two predicted-probability distributions differ.

    A small p-value indicates statistically significant
    distributional separation between the two outcome classes.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=float)

    if len(y_true) != len(y_probs):
        raise ValueError(
            "y_true and y_probs must have the same length."
        )

    valid_mask = np.isfinite(y_probs)
    y_true = y_true[valid_mask]
    y_probs = y_probs[valid_mask]

    probs_class_0 = y_probs[y_true == 0]
    probs_class_1 = y_probs[y_true == 1]

    if len(probs_class_0) == 0 or len(probs_class_1) == 0:
        raise ValueError(
            "Both outcome classes must contain valid observations."
        )

    result = cramervonmises_2samp(
        probs_class_0,
        probs_class_1,
        method="auto",
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_0_05": bool(result.pvalue < 0.05),
        "n_class_0": int(len(probs_class_0)),
        "n_class_1": int(len(probs_class_1)),
    }

def compute_calibration_intercept_slope(
    y_true,
    y_probs,
):
    """
    Estimate calibration intercept and slope.

    Ideal calibration:
    intercept = 0
    slope = 1
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=float)

    if len(y_true) != len(y_probs):
        raise ValueError(
            "y_true and y_probs must have the same length."
        )

    eps = 1e-6

    probs = np.clip(
        y_probs,
        eps,
        1.0 - eps,
    )

    logit_probs = np.log(
        probs / (1.0 - probs)
    ).reshape(-1, 1)

    calibration_model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )

    calibration_model.fit(
        logit_probs,
        y_true,
    )

    intercept = float(
        calibration_model.intercept_[0]
    )

    slope = float(
        calibration_model.coef_[0][0]
    )

    return intercept, slope

def build_statistical_summary(
    auc_ci,
    cvm_result=None,
):
    """
    Convert statistical results into a compact table for reporting.
    """
    rows = [
        {
            "analysis": "ROC-AUC Bootstrap CI",
            "estimate": auc_ci["estimate"],
            "ci_lower": auc_ci["ci_lower"],
            "ci_upper": auc_ci["ci_upper"],
            "statistic": np.nan,
            "p_value": np.nan,
            "significant_0_05": np.nan,
        }
    ]

    if cvm_result is not None:
        rows.append({
            "analysis": "Cramer-von Mises Outcome Separation",
            "estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "statistic": cvm_result["statistic"],
            "p_value": cvm_result["p_value"],
            "significant_0_05": cvm_result["significant_0_05"],
        })

    return pd.DataFrame(rows)