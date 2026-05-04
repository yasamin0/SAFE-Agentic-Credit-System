# src/reporting.py

import re

from crewai.tools import tool

from src.config import (
    APPROVAL_THRESHOLD,
    DROP_SENSITIVE_FROM_MODEL,
    PRED_THRESHOLD,
    RANDOM_STATE,
    SENSITIVE_FEATURE,
    W_AUC,
    W_FAIR,
    W_ROB,
)
from src.paths import (
    EVALUATION_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
    SYSTEM_CARD_PATH,
)


def _extract_float(text, label):
    """Extract a numeric metric from markdown text."""
    pattern = rf"\*\*{re.escape(label)}\*\*:\s*([0-9]*\.?[0-9]+)"
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _fmt(value, digits=4):
    """Format optional numeric values."""
    return f"{value:.{digits}f}" if value is not None else "N/A"


def _read_sensitivity_excerpt(max_lines=18):
    """Read a short sensitivity-report excerpt for the system card."""
    try:
        with open(SENSITIVITY_REPORT_PATH, "r", encoding="utf-8") as f:
            return "\n".join(f.read().splitlines()[:max_lines])
    except Exception:
        return "Sensitivity report was not available."


def _parse_evaluation_metrics(report_text):
    """Parse governance and auxiliary metrics from evaluation_report.md."""
    return {
        "auc": _extract_float(report_text, "Accuracy (AUC)"),
        "fair": _extract_float(report_text, "Fairness Aggregate"),
        "rob": _extract_float(report_text, "Robustness Aggregate"),

        "pr_auc": _extract_float(report_text, "PR-AUC"),
        "precision": _extract_float(report_text, "Precision"),
        "recall": _extract_float(report_text, "Recall"),
        "f1": _extract_float(report_text, "F1 Score"),
        "brier": _extract_float(report_text, "Brier Score"),

        "aurga": _extract_float(report_text, "AURGA"),
        "aurge": _extract_float(report_text, "AURGE"),
        "rgr": _extract_float(report_text, "RGR Aggregate"),
        "shap_corr": _extract_float(report_text, "SHAP-RGE Spearman Correlation"),

        "mitigated_safe": _extract_float(report_text, "Mitigated SAFE Score"),
        "mitigated_auc": _extract_float(report_text, "Mitigated AUC"),
        "mitigated_fair": _extract_float(report_text, "Mitigated Fairness Aggregate"),
    }


def _compute_safe_decision(metrics):
    """Compute final SAFE score and decision."""
    final_score = (
        W_AUC * metrics["auc"]
        + W_FAIR * metrics["fair"]
        + W_ROB * metrics["rob"]
    )

    decision = "APPROVED" if final_score >= APPROVAL_THRESHOLD else "REJECTED"

    return final_score, decision


def _get_weakest_dimension(metrics):
    """Return the weakest core SAFE dimension."""
    core_scores = {
        "AUC": metrics["auc"],
        "Fairness Aggregate": metrics["fair"],
        "Robustness Aggregate": metrics["rob"],
    }

    return min(core_scores, key=core_scores.get)


def _build_system_card(metrics, final_score, decision):
    """Build the markdown system card."""
    mitigated_safe = metrics["mitigated_safe"]

    mitigated_decision = "N/A"
    if mitigated_safe is not None:
        mitigated_decision = (
            "APPROVED" if mitigated_safe >= APPROVAL_THRESHOLD else "REJECTED"
        )

    weakest_metric = _get_weakest_dimension(metrics)
    sensitivity_excerpt = _read_sensitivity_excerpt()

    return f"""# System Card — SAFE Agentic Credit Scoring

## Final Governance Decision
**Decision:** {decision}

**Final SAFE Score:** {final_score:.3f}

**Approval Threshold:** {APPROVAL_THRESHOLD:.3f}

## Decision Logic
The model is approved only if:

`SAFE Score >= Approval Threshold`

Current result:

`{final_score:.3f} >= {APPROVAL_THRESHOLD:.3f}` → **{decision}**

## SAFE Score Formula
`SAFE Score = W_AUC*AUC + W_FAIR*Fairness_Aggregate + W_ROB*Robustness_Aggregate`

Current weights:
- W_AUC = {W_AUC:.3f}
- W_FAIR = {W_FAIR:.3f}
- W_ROB = {W_ROB:.3f}

Current computation:
- AUC = {metrics["auc"]:.4f}
- Fairness Aggregate = {metrics["fair"]:.4f}
- Robustness Aggregate = {metrics["rob"]:.4f}
- Final SAFE Score = {final_score:.4f}

## Main Reason for Decision
The weakest core dimension is **{weakest_metric}**.

In this run, the model is rejected because the weighted SAFE score is below the approval threshold.

## Additional Performance Metrics
- PR-AUC: {_fmt(metrics["pr_auc"])}
- Precision: {_fmt(metrics["precision"])}
- Recall: {_fmt(metrics["recall"])}
- F1 Score: {_fmt(metrics["f1"])}
- Brier Score: {_fmt(metrics["brier"])}

## Fairness Extension
Fairness is kept as an extension for credit lending.

The system evaluates:
- SPD
- EOD
- AOD
- DIR
- Fairness Aggregate
- Group-aware mitigation result

Fairness Aggregate: {metrics["fair"]:.4f}

## Mitigation Result
- Mitigated AUC: {_fmt(metrics["mitigated_auc"])}
- Mitigated Fairness Aggregate: {_fmt(metrics["mitigated_fair"])}
- Mitigated SAFE Score: {_fmt(metrics["mitigated_safe"])}
- Mitigated Decision: {mitigated_decision}

## SAFE AI Paper Metrics
- AURGA: {_fmt(metrics["aurga"])}
- RGR Aggregate: {_fmt(metrics["rgr"])}
- AURGE: {_fmt(metrics["aurge"])}
- SHAP-RGE Spearman Correlation: {_fmt(metrics["shap_corr"])}

## Configuration
- Prediction threshold: {PRED_THRESHOLD}
- Sensitive feature: {SENSITIVE_FEATURE}
- Drop sensitive from model: {DROP_SENSITIVE_FROM_MODEL}
- Random state: {RANDOM_STATE}

## Sensitivity Snapshot
{sensitivity_excerpt}

## Governance Note
This card separates two concepts:
1. **SAFE Score**, which is the project governance score using AUC, fairness, and robustness.
2. **Compliance Score**, which is the SAFE AI paper-style score using AURGA, AURGR, AURGE, and TOPSIS.
"""


@tool
def governance_scoring_tool(description: str):
    """Read evaluation_report.md and write the final system card."""
    try:
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            report_text = f.read()

        if not report_text.strip():
            return "REJECTED: evaluation_report.md is empty."

        metrics = _parse_evaluation_metrics(report_text)

        if metrics["auc"] is None or metrics["fair"] is None or metrics["rob"] is None:
            return (
                "REJECTED: Could not parse AUC/Fairness Aggregate/"
                "Robustness Aggregate from evaluation_report.md."
            )

        final_score, decision = _compute_safe_decision(metrics)
        system_card = _build_system_card(metrics, final_score, decision)

        with open(SYSTEM_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(system_card)

        return (
            f"{decision}: SAFE Score={final_score:.3f}. "
            f"System Card saved to {SYSTEM_CARD_PATH.name}."
        )

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"