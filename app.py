"""
app.py

Streamlit dashboard for the Student Mental Health Monitoring System.
This data product allows school counselors to:
1. Assess individual student risk from a survey form
2. Upload batch survey data and view cohort-level insights
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration & Loading
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "depression_model_v1.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
SAMPLE_SURVEY_PATH = os.path.join(ARTIFACTS_DIR, "sample_survey.csv")

# ---------------------------------------------------------------------------
# Risk Tier Thresholds (centralized for consistency)
# ---------------------------------------------------------------------------
RISK_TIERS = {
    "thresholds": [0.0, 0.3, 0.6, 1.0],
    "labels": ["Low", "Moderate", "High"],
    "colors": {"Low": "green", "Moderate": "orange", "High": "red"},
    "priority": {"Low": "Routine", "Moderate": "Monitor", "High": "Immediate"},
    "recommendations": {
        "Low": "No immediate action required. Continue routine check-ins during next survey cycle.",
        "Moderate": "Schedule a follow-up conversation within 2 weeks. Monitor for changes in behavior or academic performance.",
        "High": "Prioritize for immediate counselor outreach within 48 hours. Consider involving support services.",
    },
}


@st.cache_resource
def load_model():
    """Load the trained sklearn pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model metadata (features, metrics, governance info)."""
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_sample_survey():
    """Load sample survey CSV for demo purposes."""
    return pd.read_csv(SAMPLE_SURVEY_PATH)


# ---------------------------------------------------------------------------
# Google Form Column Mapping (for CSV exports)
# ---------------------------------------------------------------------------
GOOGLE_FORM_COLUMN_MAP = {
    # Map typical Google Form question text to model feature names
    "What is your age?": "Age",
    "Age": "Age",
    "Gender": "Gender",
    "What is your gender?": "Gender",
    "Rate your academic pressure (1-5)": "Academic Pressure",
    "Academic Pressure": "Academic Pressure",
    "What is your current CGPA?": "CGPA",
    "CGPA": "CGPA",
    "Rate your study satisfaction (1-5)": "Study Satisfaction",
    "Study Satisfaction": "Study Satisfaction",
    "Average sleep hours per night": "Sleep Duration",
    "Sleep Duration": "Sleep Duration",
    "How many hours do you sleep per night?": "Sleep Duration",
    "Have you ever had suicidal thoughts?": "Have you ever had suicidal thoughts ?",
    "Have you ever had suicidal thoughts ?": "Have you ever had suicidal thoughts ?",
    "Suicidal thoughts?": "Have you ever had suicidal thoughts ?",
    "Daily work/study hours": "Work/Study Hours",
    "Work/Study Hours": "Work/Study Hours",
    "How many hours per day do you work/study?": "Work/Study Hours",
    "Rate your financial stress (1-5)": "Financial Stress",
    "Financial Stress": "Financial Stress",
    "Family history of mental illness?": "Family History of Mental Illness",
    "Family History of Mental Illness": "Family History of Mental Illness",
    "Does your family have a history of mental illness?": "Family History of Mental Illness",
    "Education level": "Education Level",
    "Education Level": "Education Level",
    "What is your education level?": "Education Level",
    "Student ID": "Student_ID",
    "Student_ID": "Student_ID",
    "ID": "Student_ID",
    "Timestamp": "Timestamp",  # Google Forms adds this automatically
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def normalize_google_form_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a Google Form CSV export to match model input schema.
    - Renames columns based on mapping
    - Converts text values to numeric codes
    - Handles Yes/No, Gender, Sleep Duration, Education Level
    - Computes derived features
    """
    df = df.copy()
    
    # 1. Rename columns using the mapping (case-insensitive match)
    df.columns = [col.strip() for col in df.columns]
    rename_map = {}
    for col in df.columns:
        if col in GOOGLE_FORM_COLUMN_MAP:
            rename_map[col] = GOOGLE_FORM_COLUMN_MAP[col]
    df = df.rename(columns=rename_map)
    
    # 2. Drop Timestamp if exists (not used in model)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    
    # 3. Normalize Gender -> Gender_Male (0/1)
    if "Gender" in df.columns:
        df["Gender_Male"] = df["Gender"].str.strip().str.lower().map({"male": 1, "female": 0, "m": 1, "f": 0}).fillna(0).astype(int)
        df = df.drop(columns=["Gender"])
    
    # 4. Normalize Yes/No binary fields to 1/0
    binary_fields = ["Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
    for field in binary_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    
    # 5. Normalize Sleep Duration (text -> numeric hours)
    if "Sleep Duration" in df.columns:
        sleep_map = {
            "less than 5 hours": 4,
            "<5 hours": 4,
            "4 hours": 4,
            "5-6 hours": 5.5,
            "5 to 6 hours": 5.5,
            "6 hours": 6,
            "7-8 hours": 7.5,
            "7 to 8 hours": 7.5,
            "8 hours": 8,
            "more than 8 hours": 9,
            ">8 hours": 9,
        }
        df["Sleep Duration"] = df["Sleep Duration"].astype(str).str.strip().str.lower().replace(sleep_map)
        # If still text or NaN, try to extract numeric value or default to 7
        df["Sleep Duration"] = pd.to_numeric(df["Sleep Duration"], errors="coerce").fillna(7)
    
    # 6. Normalize Education Level (text -> numeric code)
    if "Education Level" in df.columns:
        edu_map = {
            "school": 1,
            "high school": 1,
            "secondary": 1,
            "undergraduate": 2,
            "bachelor": 2,
            "bachelors": 2,
            "college": 2,
            "postgraduate": 3,
            "masters": 3,
            "master": 3,
            "phd": 3,
            "doctorate": 3,
        }
        df["Education Level"] = df["Education Level"].astype(str).str.strip().str.lower().replace(edu_map)
        df["Education Level"] = pd.to_numeric(df["Education Level"], errors="coerce").fillna(2).astype(int)
    
    # 7. Ensure numeric fields are numeric
    numeric_fields = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    
    # 8. Fill missing values with defaults
    defaults = {
        "Age": 20,
        "Academic Pressure": 3,
        "CGPA": 3.0,
        "Study Satisfaction": 3,
        "Sleep Duration": 7,
        "Have you ever had suicidal thoughts ?": 0,
        "Work/Study Hours": 6,
        "Financial Stress": 3,
        "Family History of Mental Illness": 0,
        "Education Level": 2,
        "Gender_Male": 0,
    }
    for field, default_val in defaults.items():
        if field in df.columns:
            df[field] = df[field].fillna(default_val)
    
    return df


def get_risk_tier(prob: float) -> str:
    """Map probability to risk tier label."""
    thresholds = RISK_TIERS["thresholds"]
    labels = RISK_TIERS["labels"]
    for i in range(len(labels)):
        if thresholds[i] <= prob < thresholds[i + 1]:
            return labels[i]
    return labels[-1]  # High if prob == 1.0


def get_tier_info(tier: str) -> dict:
    """Get priority and recommendation for a risk tier."""
    return {
        "tier": tier,
        "color": RISK_TIERS["colors"].get(tier, "gray"),
        "priority": RISK_TIERS["priority"].get(tier, "Unknown"),
        "recommendation": RISK_TIERS["recommendations"].get(tier, ""),
    }


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features that the model expects.
    - Total Stress Score = Academic Pressure + Financial Stress
    - Lifestyle Score = (user provides directly or we default to mid value)
    """
    df = df.copy()
    # If missing, calculate from components
    if "Total Stress Score" not in df.columns:
        df["Total Stress Score"] = df["Academic Pressure"] + df["Financial Stress"]
    if "Lifestyle Score" not in df.columns:
        # Approximate: (diet_points=1) + (sleep/4)
        df["Lifestyle Score"] = 1 + (df["Sleep Duration"] / 4)
    return df


def predict_single(model, features: list, input_dict: dict) -> tuple:
    """Run prediction for a single student survey."""
    df = pd.DataFrame([input_dict])
    df = compute_derived_features(df)
    # Ensure column order matches model expectation
    df = df[features]
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)
    return pred, prob


def predict_batch(model, features: list, df: pd.DataFrame, normalize_google_form: bool = False) -> pd.DataFrame:
    """
    Run predictions for a batch of survey responses.
    
    Args:
        model: Trained sklearn pipeline
        features: List of feature names expected by model
        df: Input dataframe with survey responses
        normalize_google_form: If True, apply Google Form CSV normalization first
    """
    # Normalize Google Form CSV if requested
    if normalize_google_form:
        df = normalize_google_form_csv(df)
    
    df = compute_derived_features(df)
    
    # Keep only required features (and Student_ID if present)
    id_col = None
    if "Student_ID" in df.columns:
        id_col = df["Student_ID"].copy()
    
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    result = pd.DataFrame({
        "Risk Score": probs,
        "Prediction": preds,
    })
    if id_col is not None:
        result.insert(0, "Student_ID", id_col.values)
    # Use centralized thresholds for tiering
    result["Risk Level"] = pd.cut(
        result["Risk Score"],
        bins=RISK_TIERS["thresholds"],
        labels=RISK_TIERS["labels"],
        include_lowest=True,
    )
    # Add Priority column based on tier
    result["Priority"] = result["Risk Level"].map(RISK_TIERS["priority"])
    return result

# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
}

/* Main Content Area Font Sizes */
.main .block-container {
    font-size: 1.15rem;
}
.main h1 {
    font-size: 2.5rem !important;
}
.main h2 {
    font-size: 2rem !important;
}
.main h3 {
    font-size: 1.7rem !important;
}
.main h4 {
    font-size: 1.4rem !important;
}
.main p {
    font-size: 1.15rem !important;
    line-height: 1.7;
}
.main label {
    font-size: 1.3rem !important;
    font-weight: 500 !important;
}
.main .stMarkdown {
    font-size: 1.15rem;
}
/* Form Input Styling */
.stNumberInput input,
.stSelectbox select,
.stSlider {
    font-size: 1.2rem !important;
}
.stNumberInput label,
.stSelectbox label,
.stSlider label {
    font-size: 1.3rem !important;
    font-weight: 500 !important;
}
/* Button Sizing */
.stButton button {
    font-size: 1.3rem !important;
    padding: 0.75rem 2rem !important;
}

/* DataFrames and Tables */
.stDataFrame {
    font-size: 1.2rem !important;
}
.stDataFrame th {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}
.stDataFrame td {
    font-size: 1.2rem !important;
}

/* Metrics */
.stMetric {
    font-size: 1.2rem !important;
}
.stMetric label {
    font-size: 1.3rem !important;
}
.stMetric [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
}

/* File Uploader */
.stFileUploader label {
    font-size: 1.3rem !important;
}

/* Download Button */
.stDownloadButton button {
    font-size: 1.3rem !important;
}

/* Checkbox */
.stCheckbox label {
    font-size: 1.2rem !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 1.25rem !important;
}

/* Info, Warning, Error, Success boxes */
.stAlert {
    font-size: 1.2rem !important;
}

/* Main Header Styling */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}
.main-header p {
    color: #e0e8f0;
    font-size: 1rem;
    margin-top: 0.5rem;
    opacity: 0.9;
}

/* Page Header with Icon */
.page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #3b82f6;
}
.page-header-icon {
    font-size: 2.5rem;
}
.page-header-text h2 {
    margin: 0;
    color: #1e3a5f;
    font-weight: 600;
}
.page-header-text p {
    margin: 0.25rem 0 0 0;
    color: #64748b;
    font-size: 0.95rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}
.metric-card.high-risk {
    border-left: 4px solid #ef4444;
    background: linear-gradient(145deg, #fef2f2, #fee2e2);
}
.metric-card.moderate-risk {
    border-left: 4px solid #f59e0b;
    background: linear-gradient(145deg, #fffbeb, #fef3c7);
}
.metric-card.low-risk {
    border-left: 4px solid #10b981;
    background: linear-gradient(145deg, #ecfdf5, #d1fae5);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}
.metric-value.high { color: #dc2626; }
.metric-value.moderate { color: #d97706; }
.metric-value.low { color: #059669; }
.metric-value.primary { color: #3b82f6; }
.metric-label {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* Info Cards */
.info-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.info-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #e2e8f0;
}
.info-card-icon {
    font-size: 1.5rem;
}
.info-card-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1e3a5f;
    margin: 0;
}
.info-card p {
    font-size: 1.2rem;
}

/* Section Divider */
.section-divider {
    display: flex;
    align-items: center;
    margin: 2rem 0 1.5rem 0;
}
.section-divider::before,
.section-divider::after {
    content: "";
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
}
.section-divider-text {
    padding: 0 1rem;
    font-size: 1.5rem;
    font-weight: 600;
    color: #475569;
}

/* Risk Badge */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.risk-badge.high {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
}
.risk-badge.moderate {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white;
    box-shadow: 0 4px 15px rgba(217, 119, 6, 0.4);
}
.risk-badge.low {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
}

/* Form Styling */
.stForm {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: #dce9f5;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #2c5282 !important;
    font-size: 1.25rem !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span {
    color: #2c5282 !important;
    font-size: 1.2rem !important;
}
[data-testid="stSidebar"] h1 {
    color: #2c5282 !important;
    font-size: 2.2rem !important;
}
[data-testid="stSidebar"] h2 {
    color: #2c5282 !important;
    font-size: 1.8rem !important;
}
[data-testid="stSidebar"] h3 {
    color: #2c5282 !important;
    font-size: 1.5rem !important;
}
[data-testid="stSidebar"] h4 {
    color: #2c5282 !important;
    font-size: 1.3rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(44, 82, 130, 0.2);
    margin: 1.5rem 0;
}
[data-testid="stSidebar"] label {
    color: #2c5282 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #2c5282 !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio > div {
    color: #2c5282 !important;
    font-size: 1.5rem !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    padding: 0.6rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #2c5282 !important;
}
[data-testid="stSidebar"] a {
    color: #4299e1 !important;
    font-size: 1.2rem !important;
}

/* SDG Banner */
.sdg-banner {
    background: #7cb98f;
    padding: 1.25rem;
    border-radius: 12px;
    margin: 1rem 0;
}
.sdg-banner p {
    color: #1a5c3a !important;
    margin: 0;
    line-height: 1.6;
}

/* Team Card */
.team-card {
    background: #e8f0f7;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 0.5rem;
    border: 1px solid #b8d4e8;
}
.team-card p {
    color: #4299e1 !important;
    font-weight: 600 !important;
}
.team-member {
    color: #2c5282 !important;
    padding: 0.35rem 0;
}

/* Feature Highlight */
.feature-box {
    background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.feature-box h4 {
    color: #0369a1;
    margin: 0 0 0.5rem 0;
    font-size: 1.5rem;
}
.feature-box p {
    color: #475569;
    margin: 0;
    font-size: 1.2rem;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: #f8fafc;
    border-radius: 8px;
    font-weight: 500;
}

/* Table Styling */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
}

/* Hero Image Section */
.hero-image-section {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 16px;
}
.hero-text {
    flex: 1;
}
.hero-text h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0;
}
.hero-text p {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
}
.hero-image {
    flex-shrink: 0;
}
.hero-image img {
    max-width: 200px;
    width: 100%;
    height: auto;
    display: block;
    border-radius: 12px;
}

/* Why This Matters Section */
.why-matters-section {
    background: #ffffff;
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.why-matters-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.why-matters-header span {
    font-size: 2rem;
}
.why-matters-header h2 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0;
}
.why-matters-content p {
    color: #475569;
    font-size: 1.2rem;
    line-height: 1.8;
    margin-bottom: 1rem;
}
.why-matters-content strong {
    color: #1e3a5f;
    font-size: 1.25rem;
}
.highlight-text {
    color: #7c3aed;
    font-weight: 600;
}

/* Statistics Cards Row */
.stats-row {
    display: flex;
    gap: 1.5rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.stat-box {
    flex: 1;
    min-width: 200px;
    background: linear-gradient(145deg, #f8fafc, #e2e8f0);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    border-left: 4px solid #7c3aed;
}
.stat-box .stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #7c3aed;
    margin: 0;
}
.stat-box .stat-label {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* Alert Boxes Enhancement */
.stAlert {
    border-radius: 10px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Streamlit App Layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Student Mental Health Monitoring",
        page_icon="🧠",
        layout="wide",
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load resources
    model = load_model()
    meta = load_metadata()
    features = meta["features"]
    question_map = meta["survey_question_map"]

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    # Logo/Brand Header
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="margin: 0; font-size: 1.8rem; color: #2c5282; font-weight: 700; line-height: 1.4;">Mental Health<br>Monitoring System</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "📍 Navigation",
        ["Individual Assessment", "Cohort Monitoring", "Model Info & Governance"],
    )
    
    # Sidebar - Project Goal (SDG)
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3 style="color: #2c5282; font-size: 1.5rem;">🎯 Project Goal</h3>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="sdg-banner">
            <p style="color: #1a5c3a; font-size: 1.25rem;"><strong>SDG 3: Good Health & Well-Being</strong></p>
            <p style="margin-top: 0.5rem; font-size: 1.15rem; color: #1a5c3a;">
                Early detection of student depression to provide timely intervention and support healthier learning environments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar - Contact Us
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3 style="color: #2c5282; font-size: 1.5rem;">👥 Our Team</h3>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div class="team-card">
            <p style="color: #4299e1 !important; font-weight: 600; margin-bottom: 0.5rem; font-size: 1.25rem;">WQD7001 OCC3 Group 3</p>
            <div style="color: #2c5282; font-size: 1.15rem; padding: 0.25rem 0;">• TAN IEE HONG</div>
            <div style="color: #2c5282; font-size: 1.15rem; padding: 0.25rem 0;">• YONG KA YAN</div>
            <div style="color: #2c5282; font-size: 1.15rem; padding: 0.25rem 0;">• KAUNG HTET SHYAN</div>
            <div style="color: #2c5282; font-size: 1.15rem; padding: 0.25rem 0;">• HO WEI WEN</div>
            <div style="color: #2c5282; font-size: 1.15rem; padding: 0.25rem 0;">• CHEONG MENG BEN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Sidebar - GitHub Repository
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/alsontanwork4-cmyk/Annual-Mental-Health-Screening.git" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/GitHub-View_Repository-2ea44f?style=for-the-badge&logo=github" alt="GitHub Repo">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Page 1: Individual Student Assessment
    # ---------------------------------------------------------------------------
    if page == "Individual Assessment":
        # Hero Section
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;">
                <h1 style="font-size: 2rem; font-weight: 700; color: #1e3a5f; margin: 0;">Student Depression Assessment</h1>
                <p style="font-size: 1rem; color: #64748b; margin-top: 0.5rem;">Early detection tool to identify students who may benefit from mental health support</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Why This Matters Section
        st.markdown(
            """
            <div class="why-matters-section">
                <div class="why-matters-header">
                    <span>💡</span>
                    <h2>Why This Matters</h2>
                </div>
                <div class="why-matters-content">
                    <p>
                        One pressing issue under SDG 3 is the <strong class="highlight-text">early detection of mental health conditions</strong>, 
                        such as depression among students. Despite growing awareness, early detection and preventive 
                        strategies for student depression remain limited, often leading to diagnoses when intervention 
                        is less effective.
                    </p>
                    <p>
                        This tool uses <strong>machine learning</strong> to analyze survey responses and identify students 
                        who may benefit from mental health support, enabling <strong class="highlight-text">timely intervention</strong> 
                        and promoting healthier learning environments.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Section divider for form
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-divider-text">📝 Student Survey Form</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Quick guide
        st.markdown(
            """
            <div class="feature-box">
                <h4>📋 How to Use</h4>
                <p>Fill in the student's survey responses in the form below. The ML model will analyze the data and provide a risk assessment with actionable recommendations for counselors.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("survey_form"):
            # Question 1
            age = st.number_input("1. What is your age?", min_value=15, max_value=50, value=20)
            st.markdown("")
            
            # Question 2
            gender = st.selectbox("2. Gender", options=["Female", "Male"])
            gender_male = 1 if gender == "Male" else 0
            st.markdown("")
            
            # Question 3
            education_level = st.selectbox(
                "3. Education level (1=School, 2=Undergrad, 3=Postgrad)",
                options=["School (1)", "Undergraduate (2)", "Postgraduate (3)"],
            )
            edu_map = {"School (1)": 1, "Undergraduate (2)": 2, "Postgraduate (3)": 3}
            edu_val = edu_map[education_level]
            st.markdown("")
            
            # Question 4
            cgpa = st.number_input("4. What is your current CGPA?", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
            st.markdown("")
            
            # Question 5
            academic_pressure = st.slider("5. Rate your academic pressure (1-5)", 1, 5, 3)
            st.markdown("")
            
            # Question 6
            study_satisfaction = st.slider("6. Rate your study satisfaction (1-5)", 1, 5, 3)
            st.markdown("")
            
            # Question 7
            work_hours = st.slider("7. Daily work/study hours", 1, 16, 6)
            st.markdown("")
            
            # Question 8
            sleep_duration = st.slider("8. Average sleep hours per night", 3, 10, 7)
            st.markdown("")
            
            # Question 9
            financial_stress = st.slider("9. Rate your financial stress (1-5)", 1, 5, 3)
            st.markdown("")
            
            # Question 10
            family_history = st.selectbox(
                "10. Family history of mental illness? (0=No, 1=Yes)",
                options=["No", "Yes"],
            )
            family_val = 1 if family_history == "Yes" else 0
            st.markdown("")
            
            # Question 11
            suicidal_thoughts = st.selectbox(
                "11. Have you ever had suicidal thoughts? (0=No, 1=Yes)",
                options=["No", "Yes"],
            )
            suicidal_val = 1 if suicidal_thoughts == "Yes" else 0
            st.markdown("")

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            input_dict = {
                "Age": age,
                "Academic Pressure": academic_pressure,
                "CGPA": cgpa,
                "Study Satisfaction": study_satisfaction,
                "Sleep Duration": sleep_duration,
                "Have you ever had suicidal thoughts ?": suicidal_val,
                "Work/Study Hours": work_hours,
                "Financial Stress": financial_stress,
                "Family History of Mental Illness": family_val,
                "Education Level": edu_val,
                "Gender_Male": gender_male,
            }
            pred, prob = predict_single(model, features, input_dict)

            # Get risk tier and actionable info
            tier = get_risk_tier(prob)
            tier_info = get_tier_info(tier)

            # Section divider
            st.markdown(
                """
                <div class="section-divider">
                    <span class="section-divider-text">📊 Prediction Results</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Risk level badge mapping
            badge_class = tier.lower()
            value_class = tier.lower()

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <p class="metric-value primary">{prob:.0%}</p>
                        <p class="metric-label">Risk Score</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_res2:
                st.markdown(
                    f"""
                    <div class="metric-card {badge_class}-risk">
                        <p class="metric-value {value_class}">{tier}</p>
                        <p class="metric-label">Risk Level</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_res3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <p class="metric-value primary">{tier_info["priority"]}</p>
                        <p class="metric-label">Priority</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Depression Risk (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 30], "color": "lightgreen"},
                            {"range": [30, 60], "color": "yellow"},
                            {"range": [60, 100], "color": "salmon"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # ---------------------------------------------------------------------------
            # Actionable Insights Panel
            # ---------------------------------------------------------------------------
            st.subheader("Actionable Insights for Counselor")

            # Color-coded action box
            if tier == "High":
                st.error(f"**{tier_info['priority']} Follow-up Required**")
            elif tier == "Moderate":
                st.warning(f"**{tier_info['priority']} Recommended**")
            else:
                st.success(f"**{tier_info['priority']} Check-in**")

            st.markdown(f"**Recommended Action:** {tier_info['recommendation']}")

            # Summary card
            st.markdown("---")
            st.markdown(
                f"""
                | Attribute | Value |
                |-----------|-------|
                | **Risk Tier** | {tier} |
                | **Priority Level** | {tier_info['priority']} |
                | **Risk Probability** | {prob:.1%} |
                """
            )

            st.info(
                "**Disclaimer:** This is a decision-support signal, NOT a clinical diagnosis. "
                "All follow-up actions should be conducted by qualified counseling staff."
            )
        
        # Privacy and Medical Disclaimer at Bottom
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem 0;">
                🔒 No personal data is stored. This project is intended for educational and research purposes only and is not a substitute for professional mental health diagnosis or treatment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------------------
    # Page 2: Cohort Monitoring (Batch Upload)
    # ---------------------------------------------------------------------------
    elif page == "Cohort Monitoring":
        # Hero Section with Image
        st.markdown(
            """
            <div class="hero-image-section">
                <div class="hero-text">
                    <h1>Cohort Mental Health Monitoring</h1>
                    <p>Batch analysis tool to screen multiple students and identify those requiring immediate support</p>
                </div>
                <div class="hero-image">
                    <img src="https://img.freepik.com/free-vector/group-therapy-illustration-concept_114360-3558.jpg" alt="Cohort Monitoring">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Why This Matters for Cohort
        st.markdown(
            """
            <div class="why-matters-section">
                <div class="why-matters-header">
                    <span>💡</span>
                    <h2>Batch Analysis for Early Intervention</h2>
                </div>
                <div class="why-matters-content">
                    <p>
                        Analyzing mental health data at the <strong class="highlight-text">cohort level</strong> enables 
                        schools to identify patterns and allocate resources effectively. By screening entire classes or 
                        grades, counselors can prioritize students who need immediate support.
                    </p>
                    <p>
                        Upload your survey data to get <strong>aggregate insights</strong>, risk distribution charts, 
                        and a prioritized list of students requiring follow-up.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Feature boxes
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown(
                """
                <div class="feature-box">
                    <h4>📁 Batch Upload</h4>
                    <p>Upload CSV files exported from Google Forms or your survey system for bulk analysis.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_feat2:
            st.markdown(
                """
                <div class="feature-box">
                    <h4>📈 Aggregate Insights</h4>
                    <p>View risk distribution charts, identify high-risk students, and export results.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------------
        # Google Form Export Instructions
        # ---------------------------------------------------------------------------
        with st.expander("📋 How to export from Google Forms (Click to expand)"):
            st.markdown(
                """
                <div style="font-size: 1.2rem;">
                <h3 style="font-size: 1.4rem;">Steps to export Google Form responses:</h3>
                <ol style="font-size: 1.2rem;">
                    <li>Open your Google Form and go to the <strong>Responses</strong> tab</li>
                    <li>Click the green <strong>Sheets</strong> icon to create/open the linked spreadsheet</li>
                    <li>In the spreadsheet, click <strong>File → Download → Comma-separated values (.csv)</strong></li>
                    <li>Upload the downloaded CSV file below</li>
                </ol>
                
                <p style="font-size: 1.2rem;"><strong>Note:</strong> The system will automatically:</p>
                <ul style="font-size: 1.2rem;">
                    <li>Rename columns to match model features</li>
                    <li>Convert Yes/No to numeric values</li>
                    <li>Map text responses (e.g., "5-6 hours" for sleep) to numeric codes</li>
                    <li>Handle Gender and Education Level text inputs</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Sample data download
        sample_df = load_sample_survey()
        st.download_button(
            label="Download Sample Survey Template",
            data=sample_df.to_csv(index=False),
            file_name="sample_survey_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Upload Survey CSV", type=["csv"])

        # Add option to apply Google Form normalization
        use_google_form = st.checkbox(
            "This CSV is exported from Google Forms (apply automatic column mapping)",
            value=False,
            help="Check this if you exported directly from Google Forms. The system will normalize column names and values.",
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown('<h3 style="font-size: 1.5rem;">Uploaded Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head())

            if st.button("Run Batch Predictions"):
                with st.spinner("Predicting..."):
                    try:
                        results = predict_batch(model, features, df, normalize_google_form=use_google_form)
                    except Exception as e:
                        st.error(f"Error processing CSV: {str(e)}")
                        st.info("If you exported from Google Forms, make sure to check the 'Google Forms' checkbox above.")
                        st.stop()

                # ---------------------------------------------------------------------------
                # Actionable Summary Panel (top of results)
                # ---------------------------------------------------------------------------
                st.markdown(
                    """
                    <div class="section-divider">
                        <span class="section-divider-text">📊 Risk Summary Dashboard</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                total = len(results)
                high_risk = (results["Risk Level"] == "High").sum()
                moderate_risk = (results["Risk Level"] == "Moderate").sum()
                low_risk = (results["Risk Level"] == "Low").sum()
                high_pct = (high_risk / total * 100) if total > 0 else 0

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <p class="metric-value primary">{total}</p>
                            <p class="metric-label">Total Students</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_s2:
                    st.markdown(
                        f"""
                        <div class="metric-card high-risk">
                            <p class="metric-value high">{high_risk}</p>
                            <p class="metric-label">🚨 High Risk</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_s3:
                    st.markdown(
                        f"""
                        <div class="metric-card moderate-risk">
                            <p class="metric-value moderate">{moderate_risk}</p>
                            <p class="metric-label">⚠️ Moderate Risk</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_s4:
                    st.markdown(
                        f"""
                        <div class="metric-card low-risk">
                            <p class="metric-value low">{low_risk}</p>
                            <p class="metric-label">✅ Low Risk</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Priority action callout
                if high_risk > 0:
                    st.error(
                        f"**{high_risk} student(s) require immediate follow-up.**"
                    )
                elif moderate_risk > 0:
                    st.warning(
                        f"**{moderate_risk} student(s) flagged for monitoring.** "
                        f"Schedule follow-up conversations within 2 weeks."
                    )
                else:
                    st.success("All students in this cohort are at low risk. Continue routine monitoring.")

                # Download results
                results_sorted = results.sort_values("Risk Score", ascending=False)
                csv_out = results_sorted.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv_out,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )

                # ---------------------------------------------------------------------------
                # High-Risk Students (Immediate Action Required)
                # ---------------------------------------------------------------------------
                st.markdown('<h3 style="font-size: 1.5rem;">Immediate Action Required (High Risk)</h3>', unsafe_allow_html=True)
                high_risk_df = results[results["Risk Level"] == "High"].sort_values("Risk Score", ascending=False)
                if len(high_risk_df) > 0:
                    st.error(f"**{len(high_risk_df)} student(s) need immediate counselor outreach within 48 hours.**")
                    st.dataframe(high_risk_df, use_container_width=True)
                    st.markdown(f'<p style="font-size: 1.2rem;"><strong>Recommended Action:</strong> {RISK_TIERS["recommendations"]["High"]}</p>', unsafe_allow_html=True)
                else:
                    st.success("No high-risk students identified in this cohort.")

                # ---------------------------------------------------------------------------
                # Moderate-Risk Students (Monitor)
                # ---------------------------------------------------------------------------
                st.markdown('<h3 style="font-size: 1.5rem;">Monitor (Moderate Risk)</h3>', unsafe_allow_html=True)
                moderate_risk_df = results[results["Risk Level"] == "Moderate"].sort_values("Risk Score", ascending=False)
                if len(moderate_risk_df) > 0:
                    st.warning(f"**{len(moderate_risk_df)} student(s) flagged for follow-up within 2 weeks.**")
                    st.dataframe(moderate_risk_df, use_container_width=True)
                    st.markdown(f'<p style="font-size: 1.2rem;"><strong>Recommended Action:</strong> {RISK_TIERS["recommendations"]["Moderate"]}</p>', unsafe_allow_html=True)
                else:
                    st.info("No moderate-risk students in this cohort.")

                # ---------------------------------------------------------------------------
                # Visualizations
                # ---------------------------------------------------------------------------
                st.markdown('<h3 style="font-size: 1.5rem;">Cohort Visualizations</h3>', unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    # Risk distribution pie chart
                    risk_counts = results["Risk Level"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Level", "Count"]
                    fig_pie = px.pie(
                        risk_counts,
                        names="Risk Level",
                        values="Count",
                        title="Risk Level Distribution",
                        color="Risk Level",
                        color_discrete_map=RISK_TIERS["colors"],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_chart2:
                    # Risk score histogram
                    fig_hist = px.histogram(
                        results,
                        x="Risk Score",
                        nbins=20,
                        title="Risk Score Distribution",
                        color_discrete_sequence=["steelblue"],
                    )
                    # Add tier threshold lines
                    fig_hist.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Low/Moderate")
                    fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="Moderate/High")
                    st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.info("Please upload a CSV file to begin batch analysis.")

    # ---------------------------------------------------------------------------
    # Page 3: Model Info & Governance
    # ---------------------------------------------------------------------------
    elif page == "Model Info & Governance":
        # Hero Section with Image
        st.markdown(
            """
            <div class="hero-image-section">
                <div class="hero-text">
                    <h1>Model Information & Governance</h1>
                    <p>Transparency, performance metrics, and ethical guidelines for responsible AI use</p>
                </div>
                <div class="hero-image">
                    <img src="https://img.freepik.com/free-vector/artificial-intelligence-concept-illustration_114360-7135.jpg" alt="AI Model">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Why This Matters for Transparency
        st.markdown(
            """
            <div class="why-matters-section">
                <div class="why-matters-header">
                    <span>💡</span>
                    <h2>Responsible AI in Mental Health</h2>
                </div>
                <div class="why-matters-content">
                    <p>
                        <strong>Transparency and ethical use</strong> are critical when applying AI to sensitive domains 
                        like mental health. This section provides full visibility into how our model works, its 
                        performance metrics, and guidelines for responsible use.
                    </p>
                    <p>
                        Our model is designed as a <strong class="highlight-text">decision-support tool</strong>, not a 
                        replacement for professional judgment. All predictions should be reviewed by qualified 
                        counseling staff.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Details Card
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-divider-text">🤖 Model Details</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">📛</span>
                        <span class="info-card-title">Model Name</span>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #3b82f6;">{meta['model_name']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_info2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">🏷️</span>
                        <span class="info-card-title">Version</span>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #3b82f6;">{meta['version']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_info3:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">📅</span>
                        <span class="info-card-title">Training Date</span>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: 600; color: #3b82f6;">{meta['training_date'][:10]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Performance Metrics
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-divider-text">📈 Performance Metrics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        metrics = meta["metrics"]
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value primary">{metrics['accuracy']:.0%}</p>
                    <p class="metric-label">Accuracy</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value primary">{metrics['recall']:.0%}</p>
                    <p class="metric-label">Recall</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value primary">{metrics['roc_auc']:.2f}</p>
                    <p class="metric-label">ROC-AUC</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confusion matrix heatmap
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Depression", "Depression"],
            y=["No Depression", "Depression"],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins, sans-serif"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature List
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-divider-text">📋 Model Features</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-header">
                    <span class="info-card-icon">📝</span>
                    <span class="info-card-title">Survey Features Used by the Model</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Display features in two columns
        feat_col1, feat_col2 = st.columns(2)
        half = len(features) // 2 + len(features) % 2
        with feat_col1:
            for feat in features[:half]:
                question = question_map.get(feat, feat)
                st.markdown(f'<p style="font-size: 1.2rem; margin: 0.5rem 0;">• <strong>{feat}</strong>: {question}</p>', unsafe_allow_html=True)
        with feat_col2:
            for feat in features[half:]:
                question = question_map.get(feat, feat)
                st.markdown(f'<p style="font-size: 1.2rem; margin: 0.5rem 0;">• <strong>{feat}</strong>: {question}</p>', unsafe_allow_html=True)

        # Governance Section
        st.markdown(
            """
            <div class="section-divider">
                <span class="section-divider-text">⚖️ Governance & Ethics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        gov = meta["governance"]
        
        gov_col1, gov_col2 = st.columns(2)
        with gov_col1:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">📊</span>
                        <span class="info-card-title">Data Source</span>
                    </div>
                    <p style="color: #475569; font-size: 1.2rem;">{gov['data_source']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">🎯</span>
                        <span class="info-card-title">Intended Use</span>
                    </div>
                    <p style="color: #475569; font-size: 1.2rem;">{gov['intended_use']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gov_col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-card-header">
                        <span class="info-card-icon">🔒</span>
                        <span class="info-card-title">Privacy Note</span>
                    </div>
                    <p style="color: #475569; font-size: 1.2rem;">{gov['privacy_note']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Important Caveats
        st.markdown('<h3 style="font-size: 1.5rem;">⚠️ Important Caveats</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">• This tool is a <strong>decision-support aid</strong>, NOT a diagnostic instrument.</p>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">• All predictions should be reviewed by <strong>qualified counseling staff</strong>.</p>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">• Students identified as high-risk should be offered <strong>supportive follow-up</strong>, not punitive action.</p>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">• Model performance may vary for populations different from the training data.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
