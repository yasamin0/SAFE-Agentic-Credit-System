from crewai import Agent, Task, Crew, Process

from src.config import crew_llm
from src.paths import ensure_directories
from src.data_loader import get_credit_data, data_preprocessing_tool
from src.train import model_training_tool
from src.evaluate import evaluation_and_risk_tool
from src.reporting import governance_scoring_tool
from src.chatbot import safe_chatbot_tool, run_safe_chatbot_cli
from src.paths import CHATBOT_LOG_PATH

import subprocess
import sys

ensure_directories()
RAW_DATA_PATH = get_credit_data()

data_agent = Agent(
    role="Data Preprocessor and Feature Engineer",
    goal="Rigorously clean and transform the German Credit dataset, handling categorical variables, scaling numerical features, and creating a balanced train/test split. Produce a detailed Data Card artifact.",
    backstory=(
        "An expert in financial data preparation who specializes in preventing "
        "data leakage and ensuring data quality before any modeling starts. "
        "Their primary focus is on maximizing fairness and robustness in the downstream model."
    ),
    tools=[data_preprocessing_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

modeling_agent = Agent(
    role="Machine Learning Model Builder and Validator",
    goal="Train highly optimized predictive models (e.g., Logistic Regression and XGBoost) and select the best model based on cross-validation metrics. Output a comprehensive Model Card.",
    backstory=(
        "A seasoned ML engineer focused on credit risk assessment. "
        "They utilize state-of-the-art libraries to perform hyperparameter tuning "
        "and establish strong performance baselines."
    ),
    tools=[model_training_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

eval_agent = Agent(
    role="Risk and Performance Auditor (SAFE AI Focus)",
    goal="Evaluate the model against Accuracy, Robustness, and Fairness metrics.",
    backstory="A specialized auditor using the SAFE AI framework to stress test models.",
    tools=[evaluation_and_risk_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

safety_agent = Agent(
    role="SAFE AI Governance & Compliance Officer",
    goal=(
        "1) Compute a weighted SAFE Score from the evaluation report. "
        "2) Perform governance/compliance checks (no PII leakage, artifacts exist, reproducibility). "
        "3) Approve/Reject with a detailed, human-readable System Card."
    ),
    backstory=(
        "You are the gatekeeper of the pipeline. You consolidate artifacts from Data/Model/Eval, "
        "verify basic compliance (no raw PII, files exist, clear reporting), then compute the FINAL SAFE SCORE."
    ),
    tools=[governance_scoring_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

chatbot_agent = Agent(
    role="SAFE Results Chatbot",
    goal="Answer user questions about the finished SAFE pipeline run using generated reports and configuration artifacts.",
    backstory=(
        "A lightweight assistant that only answers from the generated SAFE artifacts. "
        "It explains results, configuration, mitigation, and feature-importance findings in plain language."
    ),
    tools=[safe_chatbot_tool],
    llm=crew_llm,
    allow_delegation=False,
    verbose=True
)

task_data_prep = Task(
    description=(
        f"Load data from {RAW_DATA_PATH}, clean it using data_preprocessing_tool, and confirm creation of "
        "clean_train_features.csv, clean_train_target.csv, clean_test_features.csv, clean_test_target.csv."
    ),
    expected_output="A summary confirming preprocessing, encoding, scaling, and the paths to the split datasets.",
    agent=data_agent,
)

task_model_train = Task(
    description="Using the cleaned data from Task 1, execute the model_training_tool to train and select the best classification model and save the 'best_model.pkl' artifact.",
    expected_output="A confirmation that the best model has been saved and the path to the generated Model Card.",
    agent=modeling_agent,
    context=[task_data_prep]
)

task_full_eval = Task(
    description=(
        "You MUST call the evaluation_and_risk_tool tool exactly once. "
        "Do not just describe what should be done. "
        "Use the trained model and test data to generate evaluation_report.md, final_report.md, and sensitivity_report.md. "
        "Your final answer must summarize the metrics returned by the tool."
    ),
    expected_output="A summary of AUC, fairness aggregate, robustness aggregate, and confirmation that evaluation_report.md was created.",
    agent=eval_agent,
    context=[task_model_train]
)

task_governance = Task(
    description=(
        "You MUST call governance_scoring_tool exactly once. "
        "Do not explain the process. "
        "Read evaluation_report.md and generate system_card.md with the final SAFE score and decision."
    ),
    expected_output="system_card.md containing decision + final SAFE score.",
    agent=safety_agent,
    context=[task_full_eval]
)

task_chatbot = Task(
    description=(
        "You MUST call safe_chatbot_tool exactly once with a short readiness query such as 'help'. "
        "Do not invent results. Confirm the chatbot is ready to answer grounded questions from the generated artifacts."
    ),
    expected_output="A short readiness message from the chatbot confirming supported question types.",
    agent=chatbot_agent,
    context=[task_governance]
)

if __name__ == "__main__":
    safe_agent_crew = Crew(
        agents=[data_agent, modeling_agent, eval_agent, safety_agent, chatbot_agent],
        tasks=[task_data_prep, task_model_train, task_full_eval, task_governance, task_chatbot],
        process=Process.sequential,
        verbose=True
    )

    print("--- Starting SAFE Agentic Credit System ---")

    final_result = safe_agent_crew.kickoff()

    print("\n\n################################################")
    print("## FINISHED! FINAL GOVERNANCE DECISION: ##")
    print("################################################")
    print(final_result)
    print("\n[SUCCESS] System Card saved.")
    print("\n[SUCCESS] Launching standalone SAFE chatbot...")

    subprocess.run([sys.executable, "-m", "src.chat_cli"])