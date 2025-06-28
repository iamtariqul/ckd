# Chronic Kidney Disease Prediction App

This is a web application built with Streamlit that predicts the likelihood of Chronic Kidney Disease (CKD) based on various medical parameters and blood sample data.

## Features

- Interactive web interface for input of medical parameters
- Real-time prediction using Random Forest Classifier
- Visualization of prediction results and model performance
- Feature importance analysis
- Detailed information about each medical parameter

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd ckd
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
pip install streamlit pandas numpy scikit-learn plotly

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Enter the patient's medical parameters in the sidebar
2. Click the "Predict" button to get the prediction
3. View the results, including:
   - Diagnosis (High/Low risk of CKD)
   - Probability of CKD
   - Model performance metrics
   - Feature importance analysis

## Data Description

The model uses the following medical parameters for prediction:

- Age: Patient's age in years
- Blood Pressure (Bp): mm/Hg
- Specific Gravity (Sg)
- Albumin (Al)
- Sugar (Su)
- Blood Glucose Random (Bgr): mgs/dl
- Blood Urea (Bu): mgs/dl
- Serum Creatinine (Sc): mgs/dl
- Sodium (Sod): mEq/L
- Potassium (Pot): mEq/L
- Hemoglobin (Hemo): gms
- Packed Cell Volume (Pcv)
- White Blood Cell Count (Wbcc): cells/cumm
- Red Blood Cell Count (Rbcc): millions/cmm
- And various other categorical parameters

## Model

The application uses a Random Forest Classifier trained on the CKD dataset. The model is trained with the following specifications:

- Train-test split: 80-20
- Feature scaling: StandardScaler
- Number of trees: 100
- Random state: 42

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly


