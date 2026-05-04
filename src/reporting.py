# src/reporting.py

# Regular expressions are used to parse numeric metrics from the evaluation report
import re

# CrewAI tool decorator so this governance step can be called by the Governance Agent
from crewai.tools import tool

# Configuration values used in the final SAFE decision rule
from src.config import (
    W_AUC,
    W_FAIR,
    W_ROB,
    APPROVAL_THRESHOLD,
    PRED_THRESHOLD,
    SENSITIVE_FEATURE,
    DROP_SENSITIVE_FROM_MODEL,
    RANDOM_STATE,
)

# Paths to the evaluation input and system-card output artifacts
from src.paths import (
    EVALUATION_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
    SYSTEM_CARD_PATH,
)


@tool
def governance_scoring_tool(description: str):
    """
    Read evaluation_report.md, compute the final SAFE score,
    and write a clear governance system card.
    """
    try:
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            rep = f.read()

        if not rep.strip():
            return "REJECTED: evaluation_report.md is empty."

        def extract_float(label):
            pattern = rf"\*\*{re.escape(label)}\*\*:\s*([0-9]*\.?[0-9]+)"
            m = re.search(pattern, rep, re.MULTILINE)
            return float(m.group(1)) if m else None

        auc = extract_float("Accuracy (AUC)")
        fair = extract_float("Fairness Aggregate")
        rob = extract_float("Robustness Aggregate")

        pr_auc = extract_float("PR-AUC")
        precision = extract_float("Precision")
        recall = extract_float("Recall")
        f1 = extract_float("F1 Score")
        brier = extract_float("Brier Score")

        aurga = extract_float("AURGA")
        aurge = extract_float("AURGE")
        rgr = extract_float("RGR Aggregate")
        shap_corr = extract_float("SHAP-RGE Spearman Correlation")

        mitigated_safe = extract_float("Mitigated SAFE Score")
        mitigated_auc = extract_float("Mitigated AUC")
        mitigated_fair = extract_float("Mitigated Fairness Aggregate")

        if auc is None or fair is None or rob is None:
            return "REJECTED: Could not parse AUC/Fairness Aggregate/Robustness Aggregate from evaluation_report.md."

        final_score = (W_AUC * auc) + (W_FAIR * fair) + (W_ROB * rob)
        decision = "APPROVED" if final_score >= APPROVAL_THRESHOLD else "REJECTED"

        mitigated_decision = "N/A"
        if mitigated_safe is not None:
            mitigated_decision = "APPROVED" if mitigated_safe >= APPROVAL_THRESHOLD else "REJECTED"

        weakest_metric = min(
            {
                "AUC": auc,
                "Fairness Aggregate": fair,
                "Robustness Aggregate": rob,
            },
            key={
                "AUC": auc,
                "Fairness Aggregate": fair,
                "Robustness Aggregate": rob,
            }.get
        )

        try:
            with open(SENSITIVITY_REPORT_PATH, "r", encoding="utf-8") as f:
                sensitivity_excerpt = "\n".join(f.read().splitlines()[:18])
        except Exception:
            sensitivity_excerpt = "Sensitivity report was not available."

        system_card = f"""# System Card — SAFE Agentic Credit Scoring

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
- AUC = {auc:.4f}
- Fairness Aggregate = {fair:.4f}
- Robustness Aggregate = {rob:.4f}
- Final SAFE Score = {final_score:.4f}

## Main Reason for Decision
The weakest core dimension is **{weakest_metric}**.

In this run, the model is rejected because the weighted SAFE score is below the approval threshold.

## Additional Performance Metrics
- PR-AUC: {pr_auc if pr_auc is not None else "N/A"}
- Precision: {precision if precision is not None else "N/A"}
- Recall: {recall if recall is not None else "N/A"}
- F1 Score: {f1 if f1 is not None else "N/A"}
- Brier Score: {brier if brier is not None else "N/A"}

## Fairness Extension
Fairness is kept as an extension for credit lending.

The system evaluates:
- SPD
- EOD
- AOD
- DIR
- Fairness Aggregate
- Group-aware mitigation result

Fairness Aggregate: {fair:.4f}

## Mitigation Result
- Mitigated AUC: {mitigated_auc if mitigated_auc is not None else "N/A"}
- Mitigated Fairness Aggregate: {mitigated_fair if mitigated_fair is not None else "N/A"}
- Mitigated SAFE Score: {mitigated_safe if mitigated_safe is not None else "N/A"}
- Mitigated Decision: {mitigated_decision}

## SAFE AI Paper Metrics
- AURGA: {aurga if aurga is not None else "N/A"}
- RGR Aggregate: {rgr if rgr is not None else "N/A"}
- AURGE: {aurge if aurge is not None else "N/A"}
- SHAP-RGE Spearman Correlation: {shap_corr if shap_corr is not None else "N/A"}

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

        with open(SYSTEM_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(system_card)

        return f"{decision}: SAFE Score={final_score:.3f}. System Card saved to {SYSTEM_CARD_PATH.name}."

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"
