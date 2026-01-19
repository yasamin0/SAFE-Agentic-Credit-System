# main.py

import os
import pandas as pd
import openml
import numpy as np 
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import joblib 
from xgboost import XGBClassifier
# --- SCALER AND PIPELINE IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

# --- CONFIGURATION & SETUP ---

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv() 

# --- USER CONFIG (via .env or environment variables) ---
DATA_PATH = os.getenv("DATA_PATH", "").strip() or None   # e.g. raw_credit_data.csv
OPENML_ID = int(os.getenv("OPENML_ID", "31"))            # default German Credit (31)

PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.50"))         # for y_pred
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.75")) # for SAFE decision

W_AUC = float(os.getenv("W_AUC", "0.4"))
W_FAIR = float(os.getenv("W_FAIR", "0.4"))
W_ROB = float(os.getenv("W_ROB", "0.2"))

# normalize weights so sum=1
w_sum = W_AUC + W_FAIR + W_ROB
if w_sum <= 0:
    raise ValueError("Weights must sum to > 0")
W_AUC, W_FAIR, W_ROB = W_AUC / w_sum, W_FAIR / w_sum, W_ROB / w_sum

SENSITIVE_FEATURE = os.getenv("SENSITIVE_FEATURE", "personal_status")  # or foreign_worker
DROP_SENSITIVE_FROM_MODEL = os.getenv("DROP_SENSITIVE_FROM_MODEL", "0") == "1"

# --- DATA LOADER ---

def get_credit_data():
    """
    If DATA_PATH is set, use that CSV.
    Otherwise fetch OpenML dataset OPENML_ID and save to raw_credit_data.csv.
    """
    try:
        if DATA_PATH:
            # Use user-provided CSV
            df = pd.read_csv(DATA_PATH)
            df.to_csv("raw_credit_data.csv", index=False)
            return "raw_credit_data.csv"

        dataset = openml.datasets.get_dataset(OPENML_ID)
        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        # German Credit mapping (good/bad)
        y = y.apply(lambda x: 1 if x == 'bad' else 0)

        data = pd.concat([X, y.rename('CreditRisk')], axis=1)
        data.to_csv("raw_credit_data.csv", index=False)
        return "raw_credit_data.csv"

    except Exception as e:
        return f"Error loading data: {e}"
RAW_DATA_PATH = get_credit_data()

@tool
def data_preprocessing_tool(file_path: str):
    """Processes, cleans, encodes, scales, and splits the raw credit data. Saves results as 'clean_train_features.csv', 'clean_test_features.csv' and their target files."""
    try:
        # 1. Load the data
        df = pd.read_csv(file_path)
        
        # 2. Separate features and target
        X = df.drop('CreditRisk', axis=1)
        y = df['CreditRisk']
        
        # 3. Detect column types
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # 4. Build the data transformer
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', ohe, categorical_features),
                ('num', StandardScaler(), numerical_features)
            ],
            remainder='passthrough'
        )
        
        # 5. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # ---- Save sensitive feature for fairness evaluation ----
        if SENSITIVE_FEATURE in X_train.columns:
            X_train[[SENSITIVE_FEATURE]].to_csv("sensitive_train.csv", index=False)
            X_test[[SENSITIVE_FEATURE]].to_csv("sensitive_test.csv", index=False)
        else:
            # fallback so evaluation doesn't crash
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_train)}).to_csv("sensitive_train.csv", index=False)
            pd.DataFrame({"group": ["UNKNOWN"] * len(X_test)}).to_csv("sensitive_test.csv", index=False)

        # ---- Optionally drop sensitive from model features ----
        if DROP_SENSITIVE_FROM_MODEL and (SENSITIVE_FEATURE in X_train.columns):
            X_train = X_train.drop(columns=[SENSITIVE_FEATURE])
            X_test = X_test.drop(columns=[SENSITIVE_FEATURE])

        # 6. Apply preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # 7. Fix column names (critical step to avoid XGBoost errors)
        raw_feature_names = (
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
            list(numerical_features)
        )
        
        # Replace illegal characters [ ] < with safe text
        clean_feature_names = [
            name.replace("[", "").replace("]", "").replace("<", "less_than_") 
            for name in raw_feature_names
        ]
        
        # 8. Save the files
        pd.DataFrame(X_train_processed, columns=clean_feature_names).to_csv("clean_train_features.csv", index=False)
        y_train.to_csv("clean_train_target.csv", index=False, header=['CreditRisk'])
        pd.DataFrame(X_test_processed, columns=clean_feature_names).to_csv("clean_test_features.csv", index=False)
        y_test.to_csv("clean_test_target.csv", index=False, header=['CreditRisk'])

        # 9. Update the datacard (without extra characters to prevent Permission Error)
        import json
        datacard = {
            "status": "CLEANED",
            "features_after_encoding": int(len(clean_feature_names)),
            "numeric_features_raw": list(map(str, numerical_features)),
            "sensitive_feature": SENSITIVE_FEATURE,
            "drop_sensitive_from_model": bool(DROP_SENSITIVE_FROM_MODEL)
        }
        with open("datacard.json", "w", encoding="utf-8") as f:
            json.dump(datacard, f, indent=2)

        return "Data successfully processed, feature names cleaned for XGBoost, and datasets saved."
        
    except Exception as e:
        return f"DATA PREPROCESSING FAILED: {str(e)}"

# --- AGENTS DEFINITION ---

@tool
def model_training_tool(description: str):
    """Trains an XGBoost model on the cleaned training data. Saves the model as 'best_model.pkl' and creates a 'model_card.md'."""
    try:
        # Load the processed training data
        X_train = pd.read_csv("clean_train_features.csv")
        y_train = pd.read_csv("clean_train_target.csv").values.ravel()

        # Initialize and train the XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Save the model artifact
        joblib.dump(model, "best_model.pkl")

        # Create the Model Card
        with open('model_card.md', 'w') as f:
            f.write("## XGBoost Model Card\n\n- Model Type: XGBoost Classifier\n- Status: Trained\n- Features: " + str(X_train.shape[1]))

        return "SUCCESS: Model trained and saved as 'best_model.pkl'. Model Card generated."
    except Exception as e:
        return f"MODEL TRAINING FAILED: {e}"

@tool
def evaluation_and_risk_tool(description: str):
    """Evaluates the model and calculates SAFE metrics: Accuracy, Robustness, and Fairness."""
    try:
        import sklearn.metrics as metrics
        model = joblib.load("best_model.pkl")
        X_test = pd.read_csv("clean_test_features.csv")
        y_test = pd.read_csv("clean_test_target.csv").values.ravel()

        # 1. Calculate accuracy (Accuracy/AUC)
        y_probs = model.predict_proba(X_test)[:, 1]
        auc_score = metrics.roc_auc_score(y_test, y_probs)
        
      # ---- REAL FAIRNESS (simple gaps) ----
        s_df = pd.read_csv("sensitive_test.csv")
        group = s_df.iloc[:, 0].astype(str).fillna("NA")

        # predictions
        pred_thr = PRED_THRESHOLD
        y_pred = (y_probs >= pred_thr).astype(int)

        tmp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "group": group})

        # Statistical parity (positive rate gap)
        pos_rates = tmp.groupby("group")["y_pred"].mean()
        spd_gap = float(pos_rates.max() - pos_rates.min()) if len(pos_rates) > 1 else 0.0

        # Equal opportunity (TPR gap)
        def tpr(df):
            tp = ((df.y_pred == 1) & (df.y_true == 1)).sum()
            fn = ((df.y_pred == 0) & (df.y_true == 1)).sum()
            return tp / (tp + fn) if (tp + fn) else 0.0

        tprs = tmp.groupby("group").apply(tpr)
        eod_gap = float(tprs.max() - tprs.min()) if len(tprs) > 1 else 0.0

        fairness_gap = 0.5 * (spd_gap + eod_gap)
        fairness_score = float(np.clip(1.0 - fairness_gap, 0.0, 1.0))

        # ---- REAL ROBUSTNESS (noise test on numeric processed columns) ----
        # numeric columns in processed data are the original numeric feature names
        import json
        with open("datacard.json", "r", encoding="utf-8") as f:
            dc = json.load(f)

        num_cols = [c for c in dc.get("numeric_features_raw", []) if c in X_test.columns]

        X_noisy = X_test.copy()
        if num_cols:
            noise = np.random.normal(0.0, 0.10, size=(len(X_noisy), len(num_cols)))
            X_noisy.loc[:, num_cols] = X_noisy.loc[:, num_cols].values + noise

        auc_noisy = metrics.roc_auc_score(y_test, model.predict_proba(X_noisy)[:, 1])
        robustness_score = float(np.clip(auc_noisy / auc_score, 0.0, 1.0)) if auc_score > 0 else 0.0

        
        report_content = f"""### Detailed SAFE AI Evaluation Report
- **Accuracy (AUC)**: {auc_score:.2f}
- **Robustness Score**: {robustness_score}
- **Fairness Score**: {fairness_score}
- **Status**: Metrics extracted for weighting.
"""

        with open('evaluation_report.md', 'w') as f:
            f.write(report_content)

        # write final report with details
        group_table = tmp.groupby("group").agg(
            n=("y_true", "count"),
            positive_rate=("y_pred", "mean"),
            tpr=("y_true", lambda s: tpr(tmp.loc[s.index]))
        ).reset_index()

        final_report = f"""# Final SAFE Agentic Credit Scoring Report

## User Controls
- Data source: {"CSV (" + str(DATA_PATH) + ")" if DATA_PATH else "OpenML (" + str(OPENML_ID) + ")"}
- Prediction threshold: {PRED_THRESHOLD}
- Approval threshold: {APPROVAL_THRESHOLD}
- Weights: AUC={W_AUC:.3f}, Fairness={W_FAIR:.3f}, Robustness={W_ROB:.3f}
- Sensitive feature: {SENSITIVE_FEATURE}

## Accuracy
- AUC: {auc_score:.4f}

## Robustness (Noise Stress Test)
- Robustness score (AUC_noisy / AUC_base): {robustness_score:.4f}

## Fairness (Group Gaps)
- SPD gap (max-min positive rate): {spd_gap:.4f}
- EOD gap (max-min TPR): {eod_gap:.4f}
- Fairness score (1 - avg gap): {fairness_score:.4f}

### Group Table
{group_table.to_markdown(index=False)}

## Auditor Notes
- If fairness score is low: consider threshold tuning or mitigation techniques.
- If robustness score is low: test more perturbations/shifts.
"""

        with open("final_report.md", "w", encoding="utf-8") as f:
            f.write(final_report)

        return report_content

    except Exception as e:
        return f"EVALUATION FAILED: {e}"
    
@tool
def governance_scoring_tool(description: str):
    """Computes weighted SAFE score from evaluation_report.md and writes system_card.md."""
    try:
        import re

        with open("evaluation_report.md", "r", encoding="utf-8") as f:
            rep = f.read()

        def extract_float(pattern, text):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else None

        auc  = extract_float(r"\*\*Accuracy \(AUC\)\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        rob  = extract_float(r"\*\*Robustness Score\*\*:\s*([0-9]*\.?[0-9]+)", rep)
        fair = extract_float(r"\*\*Fairness Score\*\*:\s*([0-9]*\.?[0-9]+)", rep)

        if auc is None or rob is None or fair is None:
            return "REJECTED: Could not parse AUC/Robustness/Fairness from evaluation_report.md."

        w_auc, w_fair, w_rob = W_AUC, W_FAIR, W_ROB
        final_score = (w_auc * auc) + (w_fair * fair) + (w_rob * rob)
        decision = "APPROVED" if final_score > APPROVAL_THRESHOLD else "REJECTED"

        system_card = f"""# System Card — SAFE Agentic Credit Scoring

## Decision
**{decision}**

## Final SAFE Score
**{final_score:.3f}**  
(Weights: AUC={w_auc}, Fairness={w_fair}, Robustness={w_rob})

## Metrics Used
- AUC: {auc:.3f}
- Fairness: {fair:.3f}
- Robustness: {rob:.3f}

## Rationale
The final SAFE score balances predictive performance with fairness and robustness.
"""

        with open("system_card.md", "w", encoding="utf-8") as f:
            f.write(system_card)

        return f"{decision}: SAFE Score={final_score:.3f}. System Card saved to system_card.md."

    except Exception as e:
        return f"GOVERNANCE FAILED: {e}"

# 1. Data Agent
data_agent = Agent(
    role='Data Preprocessor and Feature Engineer',
    goal='Rigorously clean and transform the German Credit dataset, handling categorical variables, scaling numerical features, and creating a balanced train/test split. Produce a detailed Data Card artifact.',
    backstory=(
        "An expert in financial data preparation who specializes in preventing "
        "data leakage and ensuring data quality before any modeling starts. "
        "Their primary focus is on maximizing fairness and robustness in the downstream model."
    ),
    tools=[data_preprocessing_tool], # DataAgent uses the preprocessing tool
    allow_delegation=False,
    verbose=True
)

# 2. Modeling Agent
modeling_agent = Agent(
    role='Machine Learning Model Builder and Validator',
    goal='Train highly optimized predictive models (e.g., Logistic Regression and XGBoost) and select the best model based on cross-validation metrics. Output a comprehensive Model Card.',
    backstory=(
        "A seasoned ML engineer focused on credit risk assessment. "
        "They utilize state-of-the-art libraries to perform hyperparameter tuning "
        "and establish strong performance baselines."
    ),
    tools=[model_training_tool], # ModelingAgent uses the training tool
    allow_delegation=False,
    verbose=True
)

# 3. Evaluation Agent 
eval_agent = Agent(
    role='Risk and Performance Auditor (SAFE AI Focus)',
    goal='Evaluate the model against Accuracy, Robustness, and Fairness metrics.',
    backstory='A specialized auditor using the SAFE AI framework to stress test models.',
    tools=[evaluation_and_risk_tool],
    verbose=True
)

# 4. Safety Agent- Unified Safety Agent (SAFE Score + Governance/Compliance) ---
safety_agent = Agent(
    role='SAFE AI Governance & Compliance Officer',
    goal=(
        "1) Compute a weighted SAFE Score from the evaluation report. "
        "2) Perform governance/compliance checks (no PII leakage, artifacts exist, reproducibility). "
        "3) Approve/Reject with a detailed, human-readable System Card."
    ),
    backstory=(
        "You are the gatekeeper of the pipeline. You consolidate artifacts from Data/Model/Eval, "
        "verify basic compliance (no raw PII, files exist, clear reporting), then compute the FINAL SAFE SCORE "
        "using weights selected by the user/policy: Score = 0.4*AUC + 0.4*Fairness + 0.2*Robustness. "
        "If Score > 0.75 approve; otherwise reject and recommend mitigations."
    ),
    tools=[governance_scoring_tool],
    allow_delegation=True,
    verbose=True
)


# --- TASKS DEFINITION (The Execution Flow) ---

# T1: Data Preparation
task_data_prep = Task(
    description=(
        f"Load data from {RAW_DATA_PATH}, clean it using data_preprocessing_tool, and confirm creation of "
        "clean_train_features.csv, clean_train_target.csv, clean_test_features.csv, clean_test_target.csv."
    ),
    expected_output="A summary confirming preprocessing, encoding, scaling, and the paths to the split datasets.",
    agent=data_agent,
)

# T2: Model Training
task_model_train = Task(
    description="Using the cleaned data from Task 1, execute the model_training_tool to train and select the best classification model and save the 'best_model.pkl' artifact.",
    expected_output="A confirmation that the best model has been saved and the path to the generated Model Card.",
    agent=modeling_agent,
    context=[task_data_prep] # Needs the DataAgent's output to proceed
)

# T3: Full Evaluation
task_full_eval = Task(
    description="Using the trained model and the test data, execute the evaluation_and_risk_tool. Focus on reporting the model's AUC, Robustness to noise, and any Fairness gaps detected in sensitive features.",
    expected_output="A summary of the model's performance and risk assessment, including the generated 'evaluation_report.md'.",
    agent=eval_agent,
    context=[task_model_train] # Needs the ModelingAgent's output
)

# T4: Governance Sign-off
task_governance = Task(
    description="Run governance_scoring_tool to compute the weighted SAFE Score from evaluation_report.md and generate system_card.md with Approved/Rejected decision and reasoning.",
    expected_output="system_card.md containing decision + final SAFE score.",
    agent=safety_agent,
    context=[task_full_eval]
)

# --- CREW LAUNCHER ---

if __name__ == "__main__":
    # Create the Crew
    safe_agent_crew = Crew(
        agents=[data_agent, modeling_agent, eval_agent, safety_agent],
        tasks=[task_data_prep, task_model_train, task_full_eval, task_governance],
        process=Process.sequential, # Tasks run in sequence: T1 -> T2 -> T3 -> T4
        verbose=True 
    )

    print("--- Starting SAFE Agentic Credit System ---")
    
    # Kick off the process!
    final_result = safe_agent_crew.kickoff()
    
    print("\n\n################################################")
    print("## FINISHED! FINAL GOVERNANCE DECISION: ##")
    print("################################################")
    print(final_result)
    print("\n[SUCCESS] System Card saved to 'system_card.md'")

    