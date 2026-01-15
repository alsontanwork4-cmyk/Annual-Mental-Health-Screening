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
# Streamlit App Layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Student Mental Health Monitoring",
        page_icon="🎓",
        layout="wide",
    )

    # Load resources
    model = load_model()
    meta = load_metadata()
    features = meta["features"]
    question_map = meta["survey_question_map"]

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Individual Assessment", "Cohort Monitoring", "Model Info & Governance"],
    )

    # ---------------------------------------------------------------------------
    # Page 1: Individual Student Assessment
    # ---------------------------------------------------------------------------
    if page == "Individual Assessment":
        st.title("Individual Student Mental Health Assessment")
        st.markdown(
            """
            Enter a student's survey responses below to predict their mental health risk status.
            """
        )

        with st.form("survey_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input(question_map["Age"], min_value=15, max_value=50, value=20)
                gender = st.selectbox("Gender", options=["Female", "Male"])
                gender_male = 1 if gender == "Male" else 0
                academic_pressure = st.slider(question_map["Academic Pressure"], 1, 5, 3)
                cgpa = st.number_input(question_map["CGPA"], min_value=0.0, max_value=4.0, value=3.0, step=0.1)
                study_satisfaction = st.slider(question_map["Study Satisfaction"], 1, 5, 3)
                sleep_duration = st.slider(question_map["Sleep Duration"], 3, 10, 7)

            with col2:
                suicidal_thoughts = st.selectbox(
                    question_map["Have you ever had suicidal thoughts ?"],
                    options=["No", "Yes"],
                )
                suicidal_val = 1 if suicidal_thoughts == "Yes" else 0
                work_hours = st.slider(question_map["Work/Study Hours"], 1, 16, 6)
                financial_stress = st.slider(question_map["Financial Stress"], 1, 5, 3)
                family_history = st.selectbox(
                    question_map["Family History of Mental Illness"],
                    options=["No", "Yes"],
                )
                family_val = 1 if family_history == "Yes" else 0
                education_level = st.selectbox(
                    question_map["Education Level"],
                    options=["School (1)", "Undergraduate (2)", "Postgraduate (3)"],
                )
                edu_map = {"School (1)": 1, "Undergraduate (2)": 2, "Postgraduate (3)": 3}
                edu_val = edu_map[education_level]

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

            st.subheader("Prediction Result")

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Risk Score", f"{prob:.2%}")
            with col_res2:
                st.markdown(
                    f"<h3 style='color:{tier_info['color']};'>{tier} Risk</h3>",
                    unsafe_allow_html=True,
                )
            with col_res3:
                st.metric("Priority", tier_info["priority"])

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

    # ---------------------------------------------------------------------------
    # Page 2: Cohort Monitoring (Batch Upload)
    # ---------------------------------------------------------------------------
    elif page == "Cohort Monitoring":
        st.title("Cohort Mental Health Monitoring")
        st.markdown(
            """
            Upload a CSV file containing survey responses for multiple students.
            The system will predict risk levels and provide aggregate insights.
            """
        )

        # ---------------------------------------------------------------------------
        # Google Form Export Instructions
        # ---------------------------------------------------------------------------
        with st.expander("How to export from Google Forms (Click to expand)"):
            st.markdown(
                """
                ### Steps to export Google Form responses:
                1. Open your Google Form and go to the **Responses** tab
                2. Click the green **Sheets** icon to create/open the linked spreadsheet
                3. In the spreadsheet, click **File → Download → Comma-separated values (.csv)**
                4. Upload the downloaded CSV file below
                
                **Note:** The system will automatically:
                - Rename columns to match model features
                - Convert Yes/No to numeric values
                - Map text responses (e.g., "5-6 hours" for sleep) to numeric codes
                - Handle Gender and Education Level text inputs
                """
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
            st.subheader("Uploaded Data Preview")
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
                st.subheader("Actionable Risk Summary")

                total = len(results)
                high_risk = (results["Risk Level"] == "High").sum()
                moderate_risk = (results["Risk Level"] == "Moderate").sum()
                low_risk = (results["Risk Level"] == "Low").sum()
                high_pct = (high_risk / total * 100) if total > 0 else 0

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Total Students", total)
                col_s2.metric("High Risk (Immediate)", high_risk, delta=f"{high_pct:.1f}%", delta_color="inverse")
                col_s3.metric("Moderate Risk (Monitor)", moderate_risk)
                col_s4.metric("Low Risk (Routine)", low_risk)

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
                st.subheader("Immediate Action Required (High Risk)")
                high_risk_df = results[results["Risk Level"] == "High"].sort_values("Risk Score", ascending=False)
                if len(high_risk_df) > 0:
                    st.error(f"**{len(high_risk_df)} student(s) need immediate counselor outreach within 48 hours.**")
                    st.dataframe(high_risk_df, use_container_width=True)
                    st.markdown(f"**Recommended Action:** {RISK_TIERS['recommendations']['High']}")
                else:
                    st.success("No high-risk students identified in this cohort.")

                # ---------------------------------------------------------------------------
                # Moderate-Risk Students (Monitor)
                # ---------------------------------------------------------------------------
                st.subheader("Monitor (Moderate Risk)")
                moderate_risk_df = results[results["Risk Level"] == "Moderate"].sort_values("Risk Score", ascending=False)
                if len(moderate_risk_df) > 0:
                    st.warning(f"**{len(moderate_risk_df)} student(s) flagged for follow-up within 2 weeks.**")
                    st.dataframe(moderate_risk_df, use_container_width=True)
                    st.markdown(f"**Recommended Action:** {RISK_TIERS['recommendations']['Moderate']}")
                else:
                    st.info("No moderate-risk students in this cohort.")

                # ---------------------------------------------------------------------------
                # Visualizations
                # ---------------------------------------------------------------------------
                st.subheader("Cohort Visualizations")

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
        st.title("Model Information & Governance")

        st.subheader("Model Details")
        st.markdown(f"**Model Name:** {meta['model_name']}")
        st.markdown(f"**Version:** {meta['version']}")
        st.markdown(f"**Training Date:** {meta['training_date'][:10]}")

        st.subheader("Model Performance Metrics")
        metrics = meta["metrics"]
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col_m2.metric("Recall", f"{metrics['recall']:.2%}")
        col_m3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

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
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Feature List")
        st.write("The model uses the following features from the annual student survey:")
        for feat in features:
            question = question_map.get(feat, feat)
            st.markdown(f"- **{feat}**: {question}")

        st.subheader("Governance & Ethical Guidelines")
        gov = meta["governance"]
        st.warning(
            f"""
            **Data Source:** {gov['data_source']}
            
            **Intended Use:** {gov['intended_use']}
            
            **Privacy Note:** {gov['privacy_note']}
            """
        )

        st.info(
            """
            **Important Caveats:**
            - This tool is a decision-support aid, NOT a diagnostic instrument.
            - All predictions should be reviewed by qualified counseling staff.
            - Students identified as high-risk should be offered supportive follow-up, not punitive action.
            - Model performance may vary for populations different from the training data.
            """
        )


if __name__ == "__main__":
    main()
