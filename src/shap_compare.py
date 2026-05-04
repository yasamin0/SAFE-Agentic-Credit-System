# src/shap_compare.py

import numpy as np
import pandas as pd


def _prepare_shap_sample(X_test, sample_size=200, random_state=42):
    """Prepare a numeric test sample for XGBoost SHAP contributions."""
    X_sample = X_test.copy()

    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(sample_size, random_state=random_state)

    return X_sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _compute_xgboost_shap_importance(model, X_sample):
    """Compute mean absolute SHAP values using XGBoost built-in TreeSHAP."""
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_sample)

    # Last contribution column is the bias term, so it is removed.
    shap_contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_values = shap_contribs[:, :-1]

    return pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    })


def _merge_rge_and_shap(rge_importance_df, shap_df):
    """Merge RGE and SHAP scores and compute both rankings."""
    merged = rge_importance_df.merge(shap_df, on="feature", how="inner")

    merged["rge_rank"] = merged["rge_importance"].rank(
        ascending=False,
        method="average",
    )
    merged["shap_rank"] = merged["mean_abs_shap"].rank(
        ascending=False,
        method="average",
    )

    return merged.sort_values(
        "rge_importance",
        ascending=False,
    ).reset_index(drop=True)


def _write_success_report(merged, spearman_corr, output_report_path):
    """Write the RGE-vs-SHAP comparison report."""
    columns = [
        "feature",
        "rge_importance",
        "mean_abs_shap",
        "rge_rank",
        "shap_rank",
    ]

    top_rge = merged[columns].head(15)
    top_shap = merged.sort_values("mean_abs_shap", ascending=False)[columns].head(15)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# RGE vs SHAP Comparison Report\n\n")
        f.write("This report compares RGE feature ranking with XGBoost TreeSHAP ranking.\n\n")
        f.write(f"- Spearman correlation: {spearman_corr:.4f}\n")
        f.write(f"- Compared features: {len(merged)}\n\n")

        f.write("## Top Features by RGE\n\n")
        f.write(top_rge.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Top Features by SHAP\n\n")
        f.write(top_shap.to_markdown(index=False))
        f.write("\n")


def _write_failure_report(output_report_path, error):
    """Write a short failure report if SHAP comparison fails."""
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# RGE vs SHAP Comparison Report\n\n")
        f.write("SHAP comparison could not be completed.\n\n")
        f.write(f"Reason: {error}\n")


def compare_rge_with_shap(
    model,
    X_test,
    rge_importance_df,
    output_csv_path,
    output_report_path,
    sample_size=200,
):
    """
    Compare RGE feature ranking with XGBoost TreeSHAP ranking.

    Uses XGBoost built-in SHAP contributions for stability.
    """
    try:
        X_sample = _prepare_shap_sample(X_test, sample_size=sample_size)
        shap_df = _compute_xgboost_shap_importance(model, X_sample)
        merged = _merge_rge_and_shap(rge_importance_df, shap_df)

        spearman_corr = float(
            merged["rge_importance"].corr(
                merged["mean_abs_shap"],
                method="spearman",
            )
        )

        merged.to_csv(output_csv_path, index=False)
        _write_success_report(merged, spearman_corr, output_report_path)

        return {
            "status": "completed",
            "spearman_corr": spearman_corr,
            "compared_features": int(len(merged)),
        }

    except Exception as e:
        _write_failure_report(output_report_path, e)

        return {
            "status": "failed",
            "spearman_corr": None,
            "compared_features": 0,
            "error": str(e),
        }