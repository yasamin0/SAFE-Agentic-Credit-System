import re
from crewai.tools import tool

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
from src.paths import (
    EVALUATION_REPORT_PATH,
    SENSITIVITY_REPORT_PATH,
    SYSTEM_CARD_PATH,
)


@tool
def governance_scoring_tool(description: str):
    """Computes weighted SAFE score from evaluation_report.md and writes system_card.md."""
    try:
        with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            rep = f.read()

        if not rep.strip():
            return "REJECTED: evaluation_report.md is empty."

        def extract_float(pattern, text):
            m = re.search(pattern, text, re.MULTILINE)
            return float(m.group(1)) if m else None

        auc = extract_float(r"\*\*Accuracy \(AUC\)\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        fair = extract_float(r"\*\*Fairness Aggregate\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        rob = extract_float(r"\*\*Robustness Aggregate\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        mitigated_safe = extract_float(r"\*\*Mitigated SAFE Score\*\*:\s*([0-9]*\.?[0-9]+)", rep)

        if auc is None or fair is None or rob is None:
            return "REJECTED: Could not parse AUC/Fairness Aggregate/Robustness Aggregate from evaluation_report.md."

        final_score = (W_AUC * auc) + (W_FAIR * fair) + (W_ROB * rob)
        decision = "APPROVED" if final_score >= APPROVAL_THRESHOLD else "REJECTED"

        mitigated_decision = None
        if mitigated_safe is not None:
            mitigated_decision = "APPROVED" if mitigated_safe >= APPROVAL_THRESHOLD else "REJECTED"

        mitigated_safe_text = f"{mitigated_safe:.3f}" if mitigated_safe is not None else "N/A"
        mitigated_decision_text = mitigated_decision if mitigated_decision is not None else "N/A"

        with open(SENSITIVITY_REPORT_PATH, "r", encoding="utf-8") as f:
            sensitivity_excerpt = "\n".join(f.read().splitlines()[:18])

        system_card = f"""# System Card — SAFE Agentic Credit Scoring

## Decision
**{decision}**

## Final SAFE Score
**{final_score:.3f}**

## Decision Rule
- SAFE Score = {W_AUC:.3f}*AUC + {W_FAIR:.3f}*Fairness_Aggregate + {W_ROB:.3f}*Robustness_Aggregate
- Approval threshold = {APPROVAL_THRESHOLD:.2f}
- Policy = APPROVED if SAFE Score >= threshold else REJECTED

## Metrics Used
- Baseline AUC: {auc:.3f}
- Baseline Fairness Aggregate: {fair:.3f}
- Baseline Robustness Aggregate: {rob:.3f}
- Mitigated SAFE Score: {mitigated_safe_text}
- Mitigated Decision: {mitigated_decision_text}

## Reproducibility Controls
- Prediction threshold: {PRED_THRESHOLD}
- Sensitive feature: {SENSITIVE_FEATURE}
- Drop sensitive from model: {DROP_SENSITIVE_FROM_MODEL}
- Random state: {RANDOM_STATE}

## Rationale
The final SAFE score balances predictive performance with multi-metric fairness and multi-scenario robustness.

## Mitigation Result
- Baseline decision: {decision}
- Mitigated decision: {mitigated_decision_text}
- Interpretation: mitigation is reported separately to show whether fairness-aware post-processing improves the governance outcome under a fixed policy.

## Sensitivity Snapshot
{sensitivity_excerpt}
"""
        with open(SYSTEM_CARD_PATH, "w", encoding="utf-8") as f:
            f.write(system_card)

        return f"{decision}: SAFE Score={final_score:.3f}. System Card saved to {SYSTEM_CARD_PATH.name}."

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"