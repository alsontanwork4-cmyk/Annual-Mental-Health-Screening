"""
train_export.py

Reproducible training script for the Student Depression Prediction model.
This script:
1. Loads the REAL student depression dataset (student_depression_dataset.csv)
2. Applies the same feature engineering as the notebook
3. Trains and tunes a Logistic Regression pipeline
4. Exports the best model and metadata for the Streamlit dashboard
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Dataset path (real dataset)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "student_depression_dataset.csv")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "depression_model_v1.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
SAMPLE_SURVEY_PATH = os.path.join(ARTIFACTS_DIR, "sample_survey.csv")


# ---------------------------------------------------------------------------
# 1. Load and Preprocess Real Dataset (mirroring notebook logic)
# ---------------------------------------------------------------------------

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load the real student depression dataset and apply the same
    preprocessing as the notebook:
    - Drop high-cardinality columns (City, Profession, Degree)
    - Drop sparse columns (Work Pressure, Job Satisfaction)
    - Map Sleep Duration and Dietary Habits to numeric
    - Create engineered features (Total Stress Score, Lifestyle Score, Education Level)
    - Map Yes/No to 1/0 for binary columns
    - One-hot encode Gender -> Gender_Male
    - Drop Dietary Habits after computing Lifestyle Score
    """
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"Raw dataset shape: {df.shape}")

    # Make a copy for modeling
    df_model = df.copy()

    # Drop ID column if exists
    if "id" in df_model.columns:
        df_model = df_model.drop(columns=["id"])

    # ---------------------------------------------------------------------------
    # Drop high-cardinality and sparse columns (same as notebook)
    # ---------------------------------------------------------------------------
    cols_to_drop = ["Profession", "City", "Degree", "Work Pressure", "Job Satisfaction"]
    df_model = df_model.drop(columns=[c for c in cols_to_drop if c in df_model.columns])
    print(f"Dropped columns: {cols_to_drop}")

    # ---------------------------------------------------------------------------
    # Create Education Level from Degree (before dropping Degree - already dropped)
    # We'll derive it from the data or use a default
    # Since Degree is dropped, we'll skip this or use a placeholder
    # Based on notebook: group_degree_refined function maps Degree -> Education Level
    # Since Degree is gone, we'll set Education Level = 2 (Undergrad) as default
    # ---------------------------------------------------------------------------
    # Actually, let's check if we need to derive this from original df before dropping
    if "Degree" in df.columns:
        def group_degree_refined(degree):
            if pd.isna(degree):
                return 2  # Default to Undergrad
            degree = str(degree).strip().lower()
            school_level = ["class 12", "class 10", "high school", "secondary", "ssc", "hsc"]
            postgrad_level = ["m.", "mba", "mca", "msc", "mtech", "phd", "doctorate", "master", "pg"]
            
            for term in school_level:
                if term in degree:
                    return 1  # School
            for term in postgrad_level:
                if term in degree:
                    return 3  # Postgraduate
            return 2  # Undergraduate (default)
        
        df_model["Education Level"] = df["Degree"].apply(group_degree_refined)
    else:
        df_model["Education Level"] = 2  # Default

    # ---------------------------------------------------------------------------
    # Map binary Yes/No columns to 1/0
    # ---------------------------------------------------------------------------
    binary_map = {"Yes": 1, "No": 0, 1.0: 1, 0.0: 0, 1: 1, 0: 0}
    
    if "Depression" in df_model.columns:
        df_model["Depression"] = df_model["Depression"].replace(binary_map).astype(int)
    
    if "Family History of Mental Illness" in df_model.columns:
        df_model["Family History of Mental Illness"] = df_model["Family History of Mental Illness"].replace(binary_map).astype(int)
    
    if "Have you ever had suicidal thoughts ?" in df_model.columns:
        df_model["Have you ever had suicidal thoughts ?"] = df_model["Have you ever had suicidal thoughts ?"].replace(binary_map).astype(int)

    # ---------------------------------------------------------------------------
    # Process Sleep Duration (map to numeric hours)
    # ---------------------------------------------------------------------------
    sleep_mapping = {
        "Less than 5 hours": 4,
        "'Less than 5 hours'": 4,
        "5-6 hours": 5.5,
        "'5-6 hours'": 5.5,
        "7-8 hours": 7.5,
        "'7-8 hours'": 7.5,
        "More than 8 hours": 9,
        "'More than 8 hours'": 9,
        "Others": np.nan,
        "'Others'": np.nan,
    }
    
    if "Sleep Duration" in df_model.columns:
        # Clean the column
        df_model["Sleep Duration"] = df_model["Sleep Duration"].astype(str).str.replace("'", "").str.strip()
        df_model["Sleep Duration"] = df_model["Sleep Duration"].replace(["nan", "Others"], np.nan)
        
        # Fill missing with mode
        mode_sleep = df_model["Sleep Duration"].mode()[0] if not df_model["Sleep Duration"].mode().empty else "7-8 hours"
        df_model["Sleep Duration"] = df_model["Sleep Duration"].fillna(mode_sleep)
        
        # Map to numeric
        sleep_mapping_clean = {
            "Less than 5 hours": 4,
            "5-6 hours": 5.5,
            "7-8 hours": 7.5,
            "More than 8 hours": 9,
        }
        df_model["Sleep Duration"] = df_model["Sleep Duration"].map(sleep_mapping_clean)
        df_model["Sleep Duration"] = df_model["Sleep Duration"].fillna(7.5)  # Default to 7-8 hours

    # ---------------------------------------------------------------------------
    # Process Dietary Habits (map to numeric points)
    # ---------------------------------------------------------------------------
    diet_mapping = {
        "Unhealthy": 0,
        "Moderate": 1,
        "Healthy": 2,
    }
    
    if "Dietary Habits" in df_model.columns:
        df_model["Dietary Habits"] = df_model["Dietary Habits"].astype(str).str.replace("'", "").str.strip()
        df_model["Dietary Habits"] = df_model["Dietary Habits"].map(diet_mapping).fillna(1)  # Default to Moderate

    # ---------------------------------------------------------------------------
    # Convert Financial Stress to numeric (it might be stored as string)
    # ---------------------------------------------------------------------------
    if "Financial Stress" in df_model.columns:
        df_model["Financial Stress"] = pd.to_numeric(df_model["Financial Stress"], errors="coerce").fillna(3)

    # ---------------------------------------------------------------------------
    # Create Total Stress Score (Academic Pressure + Financial Stress)
    # Note: Work Pressure is dropped, so we only use Academic Pressure + Financial Stress
    # ---------------------------------------------------------------------------
    df_model["Total Stress Score"] = (
        df_model["Academic Pressure"].fillna(0) + 
        df_model["Financial Stress"].fillna(0)
    )

    # ---------------------------------------------------------------------------
    # Create Lifestyle Score (based on Sleep Duration and Dietary Habits)
    # ---------------------------------------------------------------------------
    df_model["Lifestyle Score"] = df_model["Dietary Habits"] + (df_model["Sleep Duration"] / 4)

    # ---------------------------------------------------------------------------
    # One-hot encode Gender -> Gender_Male
    # ---------------------------------------------------------------------------
    if "Gender" in df_model.columns:
        df_model["Gender_Male"] = (df_model["Gender"] == "Male").astype(int)
        df_model = df_model.drop(columns=["Gender"])

    # ---------------------------------------------------------------------------
    # Drop Dietary Habits (after computing Lifestyle Score, per notebook)
    # ---------------------------------------------------------------------------
    if "Dietary Habits" in df_model.columns:
        df_model = df_model.drop(columns=["Dietary Habits"])

    # ---------------------------------------------------------------------------
    # Fill any remaining NaN values
    # ---------------------------------------------------------------------------
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    print(f"Preprocessed dataset shape: {df_model.shape}")
    print(f"Columns: {list(df_model.columns)}")
    
    return df_model


# ---------------------------------------------------------------------------
# 2. Preprocessing + Model Pipeline
# ---------------------------------------------------------------------------

# Final feature list (must match notebook)
NUMERIC_FEATURES = [
    "Age",
    "Academic Pressure",
    "CGPA",
    "Study Satisfaction",
    "Sleep Duration",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Education Level",
    "Total Stress Score",
    "Lifestyle Score",
    "Gender_Male",
]

# Survey question mapping for the dashboard
SURVEY_QUESTION_MAP = {
    "Age": "What is your age?",
    "Academic Pressure": "Rate your academic pressure (1-5)",
    "CGPA": "What is your current CGPA?",
    "Study Satisfaction": "Rate your study satisfaction (1-5)",
    "Sleep Duration": "Average sleep hours per night",
    "Have you ever had suicidal thoughts ?": "Have you ever had suicidal thoughts? (0=No, 1=Yes)",
    "Work/Study Hours": "Daily work/study hours",
    "Financial Stress": "Rate your financial stress (1-5)",
    "Family History of Mental Illness": "Family history of mental illness? (0=No, 1=Yes)",
    "Education Level": "Education level (1=School, 2=Undergrad, 3=Postgrad)",
    "Gender_Male": "Gender (0=Female, 1=Male)",
}


def build_pipeline():
    """Build the preprocessing + classifier pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    log_reg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_SEED,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", log_reg),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# 3. Training + Hyperparameter Tuning
# ---------------------------------------------------------------------------

def train_and_tune(X_train, y_train):
    """Train and tune the pipeline using GridSearchCV."""
    pipeline = build_pipeline()

    param_grid = {
        "classifier__solver": ["liblinear", "saga"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.01, 0.1, 1, 10],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    print("Best CV Recall:", round(grid.best_score_, 4))
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = round(roc_auc_score(y_test, y_prob), 4)
    acc = round(accuracy_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)

    print("\nConfusion Matrix:\n", cm)
    print("ROC-AUC:", roc_auc)
    print("Accuracy:", acc)
    print("Recall:", recall)

    return {
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "accuracy": acc,
        "recall": recall,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# 5. Export Model + Metadata
# ---------------------------------------------------------------------------

def export_artifacts(model, metrics, features, question_map):
    """Save model and metadata to artifacts folder."""
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    metadata = {
        "model_name": "Student Depression Prediction (Logistic Regression)",
        "version": "1.0",
        "training_date": datetime.now().isoformat(),
        "features": features,
        "survey_question_map": question_map,
        "target": {"name": "Depression", "positive_label": 1, "negative_label": 0},
        "metrics": metrics,
        "governance": {
            "data_source": "Annual Student Mental Health Survey",
            "intended_use": "Educational demonstration only. Not for clinical diagnosis.",
            "privacy_note": "No real PII is stored in this demo.",
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_PATH}")


# ---------------------------------------------------------------------------
# 6. Generate Sample Survey CSV
# ---------------------------------------------------------------------------

def generate_sample_survey(df: pd.DataFrame, n_samples: int = 10):
    """Create a small sample CSV for batch upload demo."""
    sample = df.drop(columns=["Depression"]).sample(n=n_samples, random_state=RANDOM_SEED)
    sample["Student_ID"] = [f"STU{1001 + i}" for i in range(n_samples)]
    cols = ["Student_ID"] + [c for c in sample.columns if c != "Student_ID"]
    sample = sample[cols]
    sample.to_csv(SAMPLE_SURVEY_PATH, index=False)
    print(f"Sample survey CSV saved to: {SAMPLE_SURVEY_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Training Student Depression Prediction Model")
    print("=" * 60)

    # 1. Load and preprocess real data
    print("\n[1/5] Loading and preprocessing real dataset...")
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Depression distribution:\n{df['Depression'].value_counts(normalize=True)}")

    # 2. Train/test split
    print("\n[2/5] Splitting data (80/20 stratified)...")
    X = df.drop(columns=["Depression"])
    y = df["Depression"]
    
    # Keep only the features we need
    available_features = [f for f in NUMERIC_FEATURES if f in X.columns]
    X = X[available_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Train + tune
    print("\n[3/5] Training and tuning model...")
    best_model = train_and_tune(X_train, y_train)

    # 4. Evaluate
    print("\n[4/5] Evaluating model on test set...")
    metrics = evaluate_model(best_model, X_test, y_test)

    # 5. Export
    print("\n[5/5] Exporting artifacts...")
    export_artifacts(best_model, metrics, available_features, SURVEY_QUESTION_MAP)
    
    # Generate sample survey with all features
    df_for_sample = df[["Depression"] + available_features]
    generate_sample_survey(df_for_sample, n_samples=10)

    print("\n" + "=" * 60)
    print("Done! Artifacts ready for Streamlit dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
