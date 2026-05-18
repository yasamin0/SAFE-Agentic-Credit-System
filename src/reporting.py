# src/reporting.py

import re

from crewai.tools import tool

from src.config import (
    APPROVAL_THRESHOLD,
    DROP_SENSITIVE_FROM_MODEL,
    PRED_THRESHOLD,
    RANDOM_STATE,
    SENSITIVE_FEATURE,
    W_RGA,
    W_RGR,
    W_RGE,
    W_FAIR,
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
        "selected_mitigation_delta": _extract_float(report_text, "Selected Mitigation Delta"),
        "selected_adjusted_threshold": _extract_float(report_text, "Selected Adjusted Threshold"),
    }


def _compute_safe_decision(metrics):
    """Compute final SAFE score and decision using equal SAFE paper-based metrics."""
    final_score = (
        W_RGA * metrics["aurga"]
        + W_RGR * metrics["rgr"]
        + W_RGE * metrics["aurge"]
        + W_FAIR * metrics["fair"]
    )

    decision = "APPROVED" if final_score >= APPROVAL_THRESHOLD else "REJECTED"

    return final_score, decision


def _get_weakest_dimension(metrics):
    """Return the weakest final SAFE dimension."""
    core_scores = {
        "RGA / AURGA": metrics["aurga"],
        "RGR Aggregate": metrics["rgr"],
        "RGE / AURGE": metrics["aurge"],
        "Fairness Aggregate": metrics["fair"],
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
    baseline_decision = decision

    mitigation_delta = None
    if mitigated_safe is not None:
        mitigation_delta = mitigated_safe - final_score

    mitigation_summary = "Mitigation result was not available."
    if mitigated_safe is not None:
        if mitigation_delta >= 0:
            direction = "improved"
        else:
            direction = "reduced"

        mitigation_summary = (
            f"Mitigation {direction} the SAFE Score from "
            f"{final_score:.4f} to {mitigated_safe:.4f}. "
            f"The mitigated decision is {mitigated_decision}."
        )

    return f"""# System Card — SAFE Agentic Credit Scoring

## Final Governance Decision
**Baseline Decision:** {baseline_decision}

**Baseline SAFE Score:** {final_score:.3f}

**Mitigated Decision:** {mitigated_decision}

**Mitigated SAFE Score:** {_fmt(mitigated_safe, 3)}

**Approval Threshold:** {APPROVAL_THRESHOLD:.3f}

**Governance Conclusion:** The deployment decision remains **{baseline_decision}** under the baseline governance rule. The mitigation result is reported separately as post-processing evidence.

## Decision Logic
The baseline governance decision is approved only if:

`Baseline SAFE Score >= Approval Threshold`

Baseline result:

`{final_score:.3f} >= {APPROVAL_THRESHOLD:.3f}` → **{baseline_decision}**

Mitigated result:

`{_fmt(mitigated_safe, 3)} >= {APPROVAL_THRESHOLD:.3f}` → **{mitigated_decision}**

Mitigation interpretation:

{mitigation_summary}
## SAFE Score Formula
`SAFE Score = W_RGA*AURGA + W_RGR*RGR_Aggregate + W_RGE*AURGE + W_FAIR*Fairness_Aggregate`

Current weights:
- W_RGA = {W_RGA:.3f}
- W_RGR = {W_RGR:.3f}
- W_RGE = {W_RGE:.3f}
- W_FAIR = {W_FAIR:.3f}

Current computation:
- AURGA = {metrics["aurga"]:.4f}
- RGR Aggregate = {metrics["rgr"]:.4f}
- AURGE = {metrics["aurge"]:.4f}
- Fairness Aggregate = {metrics["fair"]:.4f}
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
- Mitigation type: group-aware threshold search
- Selected threshold delta: {_fmt(metrics["selected_mitigation_delta"])}
- Selected adjusted threshold: {_fmt(metrics["selected_adjusted_threshold"])}
- Baseline SAFE Score: {final_score:.4f}
- Baseline Decision: {baseline_decision}
- Mitigated AUC: {_fmt(metrics["mitigated_auc"])}
- Mitigated Fairness Aggregate: {_fmt(metrics["mitigated_fair"])}
- Mitigated SAFE Score: {_fmt(metrics["mitigated_safe"])}
- Mitigated Decision: {mitigated_decision}
- Mitigation summary: {mitigation_summary}

## SAFE AI Paper Metrics
- AURGA: {_fmt(metrics["aurga"])}
- RGR Aggregate: {_fmt(metrics["rgr"])}
- AURGE: {_fmt(metrics["aurge"])}
- SHAP-RGE Spearman Correlation: {_fmt(metrics["shap_corr"])}

## Configuration
- Prediction threshold from configuration: {PRED_THRESHOLD}
- Approval threshold from configuration: {APPROVAL_THRESHOLD}
- Sensitive feature: {SENSITIVE_FEATURE}
- Drop sensitive from model: {DROP_SENSITIVE_FROM_MODEL}
- Random state: {RANDOM_STATE}

## Sensitivity Snapshot
{sensitivity_excerpt}

## Governance Note
This card uses the final SAFE score requested for this project:
1. **RGA / AURGA** for rank-based accuracy.
2. **RGR Aggregate** for rank-based robustness.
3. **RGE / AURGE** for rank-based explainability.
4. **Fairness Aggregate** for credit-lending fairness.

All four dimensions use equal weights.
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

        if (
            metrics["aurga"] is None
            or metrics["rgr"] is None
            or metrics["aurge"] is None
            or metrics["fair"] is None
        ):
            return (
                "REJECTED: Could not parse AURGA/RGR Aggregate/"
                "AURGE/Fairness Aggregate from evaluation_report.md."
            )

        final_score, decision = _compute_safe_decision(metrics)
        system_card = _build_system_card(metrics, final_score, decision)

        with open(SYSTEM_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(system_card)

        mitigated_safe = metrics.get("mitigated_safe")
        mitigated_decision = "N/A"
        if mitigated_safe is not None:
            mitigated_decision = (
                "APPROVED" if mitigated_safe >= APPROVAL_THRESHOLD else "REJECTED"
            )

        return (
            f"{decision}: Baseline SAFE Score={final_score:.3f}. "
            f"Mitigated Decision={mitigated_decision}; "
            f"Mitigated SAFE Score={_fmt(mitigated_safe, 3)}. "
            f"System Card saved to {SYSTEM_CARD_PATH.name}."
        )

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"