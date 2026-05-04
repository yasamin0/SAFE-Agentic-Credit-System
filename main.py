# main.py

import os
import subprocess
import sys

# Use non-GUI backend for plot generation inside CrewAI tools.
os.environ["MPLBACKEND"] = "Agg"

from crewai import Agent, Crew, Process, Task

from src.config import crew_llm
from src.data_loader import data_preprocessing_tool, get_credit_data
from src.evaluate import evaluation_and_risk_tool
from src.paths import ensure_directories
from src.reporting import governance_scoring_tool
from src.train import model_training_tool


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

ensure_directories()
RAW_DATA_PATH = get_credit_data()


# ------------------------------------------------------------
# Agents
# ------------------------------------------------------------
data_agent = Agent(
    role="Data Preprocessor and Feature Engineer",
    goal=(
        "Clean and transform the German Credit dataset, encode categorical variables, "
        "scale numerical features, create train/test splits, and generate the Data Card."
    ),
    backstory=(
        "An expert in financial data preparation focused on data quality, leakage prevention, "
        "fairness, and robustness before modeling."
    ),
    tools=[data_preprocessing_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True,
)

modeling_agent = Agent(
    role="Machine Learning Model Builder and Validator",
    goal=(
        "Train model candidates using model_training_tool and report only tool-generated artifacts. "
        "Never fabricate metrics, file paths, or model-card content."
    ),
    backstory=(
        "A strict ML engineering agent for credit-risk modeling. "
        "It only reports results produced by the tool and saved project artifacts."
    ),
    tools=[model_training_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True,
)


eval_agent = Agent(
    role="Risk and Performance Auditor (SAFE AI Focus)",
    goal="Evaluate the trained model across accuracy, fairness, robustness, and SAFE paper metrics.",
    backstory=(
        "A SAFE AI auditor that stress-tests credit-risk models and generates grounded "
        "evaluation reports."
    ),
    tools=[evaluation_and_risk_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True,
)


safety_agent = Agent(
    role="SAFE AI Governance & Compliance Officer",
    goal=(
        "Compute the final SAFE score, apply approval/rejection logic, "
        "and generate a clear governance System Card."
    ),
    backstory=(
        "The governance gatekeeper of the pipeline. It consolidates evaluation artifacts, "
        "checks reproducibility and reporting quality, and writes the final decision."
    ),
    tools=[governance_scoring_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True,
)


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

task_data_prep = Task(
    description=(
        f"Load data from {RAW_DATA_PATH}, clean it using data_preprocessing_tool, "
        "and confirm creation of clean_train_features.csv, clean_train_target.csv, "
        "clean_test_features.csv, and clean_test_target.csv."
    ),
    expected_output=(
        "A short confirmation that preprocessing, encoding, scaling, train/test split, "
        "EDA artifacts, outlier analysis, and Data Card generation were completed."
    ),
    agent=data_agent,
)


task_model_train = Task(
    description=(
        "Using the cleaned data from Task 1, call model_training_tool exactly once. "
        "The tool trains multiple model candidates, performs cross-validation and hyperparameter search, "
        "saves the governance model as best_model.pkl, and generates the real model card.\n\n"
        "Rules:\n"
        "- Do not invent model metrics.\n"
        "- Do not write or summarize a fake model card.\n"
        "- Do not mention generated_model_card.txt or /model_card/best_model_card.html.\n"
        "- Do not create example accuracy, precision, recall, or F1 values.\n"
        "- Only confirm the real artifacts created by the tool.\n"
        "- Real model card path: docs/model_card.md.\n"
        "- Real CV results path: reports/cv_results.csv."
    ),
    expected_output=(
        "A short confirmation only: model candidates trained, cross-validation and hyperparameter "
        "search completed, best_model.pkl saved, docs/model_card.md generated, and "
        "reports/cv_results.csv generated. No fabricated metrics."
    ),
    agent=modeling_agent,
    context=[task_data_prep],
)


task_full_eval = Task(
    description=(
        "Call evaluation_and_risk_tool exactly once. "
        "Use the trained model and test data to generate evaluation_report.md, "
        "final_report.md, and sensitivity_report.md. "
        "The final answer must summarize only the metrics returned by the tool."
    ),
    expected_output=(
        "A grounded summary of AUC, PR-AUC, fairness aggregate, robustness aggregate, "
        "SAFE score, RGA/RGR/RGE results, compliance outputs, and generated report files."
    ),
    agent=eval_agent,
    context=[task_model_train],
)


task_governance = Task(
    description=(
        "Call governance_scoring_tool exactly once. "
        "Return ONLY the exact text returned by governance_scoring_tool. "
        "Do not summarize, expand, explain, rewrite, or add extra governance interpretation. "
        "Do not create a new System Card in the final answer. "
        "The actual System Card is already written by the tool to docs/system_card.md."
    ),
    expected_output=(
        "Exactly the governance_scoring_tool return string, for example: "
        "REJECTED: SAFE Score=0.692. System Card saved to system_card.md."
    ),
    agent=safety_agent,
    context=[task_full_eval],
)

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------

def build_crew():
    return Crew(
        agents=[
            data_agent,
            modeling_agent,
            eval_agent,
            safety_agent,
        ],
        tasks=[
            task_data_prep,
            task_model_train,
            task_full_eval,
            task_governance,
        ],
        process=Process.sequential,
        verbose=True,
    )


def main():
    print("--- Starting SAFE Agentic Credit System ---")

    safe_agent_crew = build_crew()
    final_result = safe_agent_crew.kickoff()

    print("\n\n################################################")
    print("## FINISHED! FINAL GOVERNANCE DECISION ##")
    print("################################################")
    print(final_result)
    print("\n[SUCCESS] System Card saved.")
    print("[SUCCESS] SAFE pipeline completed.")

    subprocess.run([sys.executable, "-m", "src.chat_cli"])

if __name__ == "__main__":
    main()