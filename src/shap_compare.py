import numpy as np
import pandas as pd


def _prepare_shap_sample(X_test, sample_size=100, random_state=42):
    """Prepare a numeric test sample for SHAP computation."""
    X_sample = X_test.copy()

    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(sample_size, random_state=random_state)

    return X_sample.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _prepare_background_sample(X_sample, background_size=30, random_state=42):
    """Prepare a small background sample for model-agnostic SHAP."""
    if len(X_sample) > background_size:
        return X_sample.sample(background_size, random_state=random_state)

    return X_sample.copy()


def _extract_class_one_shap_values(shap_values):
    """
    Extract SHAP values for the positive class.

    Different SHAP explainers return different shapes:
    - list[class_0, class_1]
    - array with shape (n_samples, n_features)
    - array with shape (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def _compute_xgboost_shap_importance(model, X_sample):
    """Compute mean absolute SHAP values using XGBoost built-in TreeSHAP."""
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_sample)

    shap_contribs = booster.predict(dmatrix, pred_contribs=True)
    shap_values = shap_contribs[:, :-1]

    return pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "shap_method": "xgboost_treeshap",
    })


def _compute_tree_shap_importance(model, X_sample):
    """Compute SHAP importance for tree-based sklearn models such as Random Forest."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_values = _extract_class_one_shap_values(shap_values)

    return pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "shap_method": "tree_explainer",
    })


def _compute_model_agnostic_shap_importance(
    model,
    X_sample,
    background_size=30,
    random_state=42,
    nsamples=100,
):
    """
    Compute model-agnostic SHAP importance for models such as VotingClassifier
    and StackingClassifier.

    This is slower than TreeSHAP, so sample sizes are intentionally small.
    """
    import shap

    background = _prepare_background_sample(
        X_sample,
        background_size=background_size,
        random_state=random_state,
    )

    def predict_positive(data):
        data_df = pd.DataFrame(data, columns=X_sample.columns)
        return model.predict_proba(data_df)[:, 1]

    explainer = shap.KernelExplainer(predict_positive, background)
    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)

    shap_values = _extract_class_one_shap_values(shap_values)

    return pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "shap_method": "kernel_explainer",
    })


def compute_general_shap_importance(
    model,
    X_test,
    model_name="model",
    sample_size=100,
    background_size=30,
    random_state=42,
):
    """
    Compute SHAP importance for different model types.

    Priority:
    1. XGBoost built-in TreeSHAP if model has get_booster()
    2. TreeExplainer for tree models such as Random Forest
    3. KernelExplainer for ensemble/meta models such as Voting/Stacking
    """
    X_sample = _prepare_shap_sample(
        X_test,
        sample_size=sample_size,
        random_state=random_state,
    )

    try:
        if hasattr(model, "get_booster"):
            shap_df = _compute_xgboost_shap_importance(model, X_sample)

        elif model.__class__.__name__ in [
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
        ]:
            shap_df = _compute_tree_shap_importance(model, X_sample)

        else:
            shap_df = _compute_model_agnostic_shap_importance(
                model=model,
                X_sample=X_sample,
                background_size=background_size,
                random_state=random_state,
            )

        shap_df["model"] = model_name
        return shap_df, {
            "status": "completed",
            "model": model_name,
            "shap_method": shap_df["shap_method"].iloc[0],
            "sample_size": int(len(X_sample)),
            "error": None,
        }

    except Exception as e:
        return pd.DataFrame({
            "model": [model_name],
            "feature": ["ERROR"],
            "mean_abs_shap": [np.nan],
            "shap_method": ["failed"],
        }), {
            "status": "failed",
            "model": model_name,
            "shap_method": "failed",
            "sample_size": int(len(X_sample)),
            "error": str(e),
        }


def merge_rge_and_shap(rge_importance_df, shap_df, model_name):
    """Merge RGE and SHAP scores and compute rankings."""
    merged = rge_importance_df.merge(shap_df, on="feature", how="inner")

    merged["model"] = model_name

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


def compute_rge_shap_spearman(merged):
    """Compute Spearman correlation between RGE importance and SHAP importance."""
    if merged.empty:
        return None

    if "ERROR" in merged["feature"].values:
        return None

    corr = merged["rge_importance"].corr(
        merged["mean_abs_shap"],
        method="spearman",
    )

    if pd.isna(corr):
        return None

    return float(corr)


def write_top_models_shap_rge_report(summary_df, output_report_path):
    """Write a report for SHAP-RGE comparison across selected top models."""
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("# Top Models SHAP-RGE Comparison Report\n\n")
        f.write(
            "This report compares RGE feature ranking with SHAP feature ranking "
            "for the top selected candidate models.\n\n"
        )

        f.write("## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write(
            "A higher Spearman correlation means that the RGE-based ranking and "
            "the SHAP-based ranking agree more strongly. XGBoost uses built-in "
            "TreeSHAP, Random Forest uses SHAP TreeExplainer, and meta-models "
            "such as Voting or Stacking use model-agnostic KernelExplainer.\n"
        )