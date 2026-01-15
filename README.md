# Student Mental Health Monitoring System

A data product for school counselors to monitor student mental health risk using an ML-based prediction model trained on annual student survey data.

## Overview

This system demonstrates the deployment of a machine learning model as a data product. It enables:

1. **Individual Assessment**: Enter a single student's survey responses to predict their mental health risk status
2. **Cohort Monitoring**: Upload batch survey data (CSV) to get aggregate insights and identify high-risk students
3. **Model Transparency**: View model performance metrics and governance guidelines

## Project Structure

```
data_product/
├── app.py                  # Streamlit dashboard application
├── train_export.py         # Model training and export script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── artifacts/
    ├── depression_model_v1.joblib   # Trained sklearn pipeline
    ├── metadata.json                # Model metadata and metrics
    └── sample_survey.csv            # Sample survey template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train the Model

If you want to regenerate the model artifacts:

```bash
python train_export.py
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Features

### Individual Assessment Page
- Input form for all survey questions
- Real-time risk prediction with probability score
- Visual gauge chart showing risk level
- Color-coded risk status (Low/Moderate/High)

### Cohort Monitoring Page
- CSV file upload for batch predictions
- Download sample survey template
- Summary statistics (total students, high-risk count)
- Risk distribution pie chart
- Risk score histogram with threshold line
- Exportable prediction results

### Model Info & Governance Page
- Model version and training date
- Performance metrics (Accuracy, Recall, ROC-AUC)
- Confusion matrix visualization
- Feature list with survey question mapping
- Ethical guidelines and usage caveats

## Model Details

- **Algorithm**: Logistic Regression (tuned via GridSearchCV)
- **Features**: 13 numeric features derived from student survey
- **Target**: Binary classification (Depression: Yes/No)
- **Preprocessing**: Median imputation + StandardScaler

### Input Features

| Feature | Survey Question |
|---------|-----------------|
| Age | What is your age? |
| Academic Pressure | Rate your academic pressure (1-5) |
| CGPA | What is your current CGPA? |
| Study Satisfaction | Rate your study satisfaction (1-5) |
| Sleep Duration | Average sleep hours per night |
| Suicidal Thoughts | Have you ever had suicidal thoughts? (0=No, 1=Yes) |
| Work/Study Hours | Daily work/study hours |
| Financial Stress | Rate your financial stress (1-5) |
| Family History | Family history of mental illness? (0=No, 1=Yes) |
| Education Level | 1=School, 2=Undergrad, 3=Postgrad |
| Gender | 0=Female, 1=Male |

*Note: Total Stress Score and Lifestyle Score are computed automatically from other inputs.*

## Important Caveats

- **This is a demonstration/educational tool only**
- Not intended for clinical diagnosis
- All predictions should be reviewed by qualified counseling staff
- Students identified as high-risk should be offered supportive follow-up
- Model performance may vary for different populations

## Using Google Forms for Data Collection

The system supports direct CSV uploads from Google Forms:

### Setup Google Form
1. Create a Google Form with questions matching the survey schema (see feature list above)
2. Link responses to a Google Sheet (Responses tab → green Sheets icon)

### Export and Upload
1. In the linked Google Sheet: **File → Download → Comma-separated values (.csv)**
2. Upload the CSV in the "Cohort Monitoring" page
3. Check the "This CSV is exported from Google Forms" option
4. Click "Run Batch Predictions"

The system will automatically:
- Map Google Form question text to model features
- Convert Yes/No responses to numeric values
- Normalize sleep duration and education level text
- Compute derived features

A sample Google Form CSV is available at `artifacts/google_form_sample.csv` for reference.

### Handling Missing Data
The system automatically handles missing or invalid survey responses through three layers:
1. **CSV Normalization**: Fills missing values with sensible defaults (e.g., Age=20, Sleep=7 hours)
2. **Model Pipeline**: Built-in imputer uses training data medians as fallback
3. **Validation**: Converts invalid entries (text in numeric fields) to defaults

See `MISSING_DATA_HANDLING.md` for detailed documentation and test `artifacts/missing_data_test.csv` for examples.

## For Presentation

1. **Start the app**: `streamlit run app.py`
2. **Demo individual assessment**: Fill in sample values and show prediction
3. **Demo batch upload**: Use the Google Form sample CSV to show cohort monitoring
4. **Show model governance**: Highlight ethical considerations and Google Form integration

## License

Educational use only. Part of WQD7001 - Principles of Data Science coursework.
