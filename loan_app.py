import streamlit as st
import joblib
import numpy as np

# Load your trained Random Forest model
# Ensure to replace the path with the correct one where your model is saved
model = joblib.load('randomforest.joblib')

# Input fields for loan features
st.title("Loan Prediction App")

st.header("Enter Applicant Details")

# Collecting inputs from the user
gender = st.selectbox("Gender", ('Male', 'Female'))
married = st.selectbox("Married", ('Yes', 'No'))
dependents = st.selectbox("Dependents", ('0', '1', '2', '3+'))
education = st.selectbox("Education", ('Graduate', 'Not Graduate'))
self_employed = st.selectbox("Self Employed", ('Yes', 'No'))
credit_history = st.selectbox("Credit History", ('0', '1'))
property_area = st.selectbox("Property Area", ('Urban', 'Rural', 'Semiurban'))
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0, value=360)

# Log transformations for income and loan features
applicant_income_log = np.log1p(applicant_income)
total_income_log = np.log1p(applicant_income + coapplicant_income)
loan_amount_log = np.log1p(loan_amount)
loan_amount_term_log = np.log1p(loan_amount_term)

# Prepare the features for prediction
features = [[
    1 if gender == 'Male' else 0,  # Encoding gender as 1 for Male, 0 for Female
    1 if married == 'Yes' else 0,  # Encoding married status
    int(dependents.replace('3+', '3')),  # Encoding dependents as integers
    1 if education == 'Graduate' else 0,  # Encoding education
    1 if self_employed == 'Yes' else 0,  # Encoding self-employed
    int(credit_history),  # Converting credit history to integer
    0 if property_area == 'Urban' else (1 if property_area == 'Rural' else 2),  # Encoding property area
    applicant_income_log,  # Log of Applicant Income
    loan_amount_log,  # Log of Loan Amount
    loan_amount_term_log,  # Log of Loan Amount Term
    total_income_log  # Log of Total Income (Applicant + Coapplicant)
]]

# Check loan approval when the button is clicked
if st.button("Check Loan Approval"):
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    # Display the prediction result on the main screen
    st.header("Prediction Result")
    
    if prediction[0] == 1:
        st.success("Loan Approved.")
    else:
        st.error("Loan Not Approved.")
