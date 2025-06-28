import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="CKD Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Chronic Kidney Disease Prediction")
st.markdown("""
This application uses machine learning to predict the likelihood of Chronic Kidney Disease (CKD) 
based on blood sample and other medical parameters. Please input the required values below.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('CKDDataset.csv')
    return df

# Load and preprocess data
df = load_data()

# Handle missing values
df = df.replace('?', np.nan)
df = df.replace('', np.nan)

# Convert numeric columns
numeric_columns = ['Age', 'Bp', 'Sg', 'Al', 'Su', 'Bgr', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Pcv', 'Wbcc', 'Rbcc']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
categorical_columns = ['Rbc', 'Pc', 'Pcc', 'Ba', 'Htn', 'Dm', 'Cad', 'Appet', 'Pe', 'Ane']
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Handle missing values in the target variable
df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
df = df.dropna(subset=['Class'])  # Drop rows with missing target values

# Encode categorical variables
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes

# Prepare features and target
X = df.drop(['Class', 'Patient_ID'], axis=1)
y = df['Class']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model()

# Create columns for input
st.sidebar.header("Patient Information")

# Create input fields
def get_user_input():
    input_data = {}
    
    # Numeric inputs
    input_data['Age'] = st.sidebar.number_input("Age", min_value=1, max_value=100, value=50)
    input_data['Bp'] = st.sidebar.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=200, value=120)
    input_data['Sg'] = st.sidebar.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025])
    input_data['Al'] = st.sidebar.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5])
    input_data['Su'] = st.sidebar.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5])
    input_data['Bgr'] = st.sidebar.number_input("Blood Glucose Random (mgs/dl)", min_value=0, max_value=500, value=120)
    input_data['Bu'] = st.sidebar.number_input("Blood Urea (mgs/dl)", min_value=0, max_value=200, value=50)
    input_data['Sc'] = st.sidebar.number_input("Serum Creatinine (mgs/dl)", min_value=0.0, max_value=20.0, value=1.2)
    input_data['Sod'] = st.sidebar.number_input("Sodium (mEq/L)", min_value=100, max_value=200, value=135)
    input_data['Pot'] = st.sidebar.number_input("Potassium (mEq/L)", min_value=2.0, max_value=8.0, value=4.0)
    input_data['Hemo'] = st.sidebar.number_input("Hemoglobin (gms)", min_value=0.0, max_value=20.0, value=12.0)
    input_data['Pcv'] = st.sidebar.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40)
    input_data['Wbcc'] = st.sidebar.number_input("White Blood Cell Count (cells/cumm)", min_value=0, max_value=20000, value=9000)
    input_data['Rbcc'] = st.sidebar.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=8.0, value=4.5)

    # Categorical inputs
    input_data['Rbc'] = st.sidebar.selectbox("Red Blood Cells", options=['normal', 'abnormal'])
    input_data['Pc'] = st.sidebar.selectbox("Pus Cell", options=['normal', 'abnormal'])
    input_data['Pcc'] = st.sidebar.selectbox("Pus Cell Clumps", options=['present', 'notpresent'])
    input_data['Ba'] = st.sidebar.selectbox("Bacteria", options=['present', 'notpresent'])
    input_data['Htn'] = st.sidebar.selectbox("Hypertension", options=['yes', 'no'])
    input_data['Dm'] = st.sidebar.selectbox("Diabetes Mellitus", options=['yes', 'no'])
    input_data['Cad'] = st.sidebar.selectbox("Coronary Artery Disease", options=['yes', 'no'])
    input_data['Appet'] = st.sidebar.selectbox("Appetite", options=['good', 'poor'])
    input_data['Pe'] = st.sidebar.selectbox("Pedal Edema", options=['yes', 'no'])
    input_data['Ane'] = st.sidebar.selectbox("Anemia", options=['yes', 'no'])
    
    return input_data

# Get user input
user_input = get_user_input()

# Create a DataFrame from user input
def preprocess_user_input(user_input):
    # Convert categorical inputs to numeric
    categorical_map = {'normal': 0, 'abnormal': 1, 'present': 1, 'notpresent': 0, 'yes': 1, 'no': 0, 'good': 1, 'poor': 0}
    user_df = pd.DataFrame([user_input])
    
    for col in categorical_columns:
        user_df[col] = user_df[col].map(categorical_map)
    
    # Ensure column order matches training data
    user_df = user_df[X.columns]
    
    # Scale the input
    user_df_scaled = scaler.transform(user_df)
    
    return user_df_scaled

# Make prediction when button is clicked
if st.sidebar.button("Predict"):
    # Preprocess user input
    user_data_scaled = preprocess_user_input(user_input)
    
    # Make prediction
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)
    
    # Display results
    st.header("Prediction Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnosis")
        if prediction[0] == 1:
            st.error("High risk of Chronic Kidney Disease")
        else:
            st.success("Low risk of Chronic Kidney Disease")
            
        st.subheader("Probability")
        prob_ckd = prediction_proba[0][1] * 100
        st.write(f"Probability of CKD: {prob_ckd:.2f}%")
        
    with col2:
        # Create a gauge chart for the probability
        fig = px.pie(values=[prob_ckd, 100-prob_ckd], 
                    names=['CKD Risk', 'Normal'],
                    hole=0.7,
                    color_discrete_sequence=['#FF4B4B', '#45A3E5'])
        fig.update_layout(
            annotations=[dict(text=f"{prob_ckd:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig)

# Display model performance metrics
# st.header("Model Performance")
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Training Accuracy")
#     train_pred = model.predict(X_train_scaled)
#     train_accuracy = accuracy_score(y_train, train_pred)
#     st.write(f"{train_accuracy*100:.2f}%")

# with col2:
#     st.subheader("Test Accuracy")
#     test_pred = model.predict(X_test_scaled)
#     test_accuracy = accuracy_score(y_test, test_pred)
#     st.write(f"{test_accuracy*100:.2f}%")

# Feature importance plot
st.header("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
             title='Feature Importance in Prediction')
st.plotly_chart(fig)

# Add information about the features
st.header("Feature Information")
st.markdown("""
| Feature | Description |
|---------|-------------|
| Age | Patient's age in years |
| Bp | Blood Pressure (mm/Hg) |
| Sg | Specific Gravity |
| Al | Albumin |
| Su | Sugar |
| Bgr | Blood Glucose Random (mgs/dl) |
| Bu | Blood Urea (mgs/dl) |
| Sc | Serum Creatinine (mgs/dl) |
| Sod | Sodium (mEq/L) |
| Pot | Potassium (mEq/L) |
| Hemo | Hemoglobin (gms) |
| Pcv | Packed Cell Volume |
| Wbcc | White Blood Cell Count (cells/cumm) |
| Rbcc | Red Blood Cell Count (millions/cmm) |
| Rbc | Red Blood Cells (normal/abnormal) |
| Pc | Pus Cell (normal/abnormal) |
| Pcc | Pus Cell Clumps (present/notpresent) |
| Ba | Bacteria (present/notpresent) |
| Htn | Hypertension (yes/no) |
| Dm | Diabetes Mellitus (yes/no) |
| Cad | Coronary Artery Disease (yes/no) |
| Appet | Appetite (good/poor) |
| Pe | Pedal Edema (yes/no) |
| Ane | Anemia (yes/no) |
""")
