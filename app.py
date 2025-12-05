import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    with open("ada_bundle.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["threshold"]

model, threshold = load_model()

# App title
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer will churn based on their profile and service usage")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)
    
    st.subheader("Personal Details")
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    
    multiple_lines = st.selectbox("Multiple Lines", 
                                  ["No", "Yes", "No phone service"])
    
    internet_service = st.selectbox("Internet Service", 
                                   ["DSL", "Fiber optic", "No"])
    
    online_security = st.selectbox("Online Security", 
                                  ["No", "Yes", "No internet service"])
    
    online_backup = st.selectbox("Online Backup", 
                                ["No", "Yes", "No internet service"])
    
    device_protection = st.selectbox("Device Protection", 
                                    ["No", "Yes", "No internet service"])
    
    tech_support = st.selectbox("Tech Support", 
                               ["No", "Yes", "No internet service"])

# Additional services
st.subheader("Streaming Services")
col3, col4 = st.columns(2)

with col3:
    streaming_tv = st.selectbox("Streaming TV", 
                               ["No", "Yes", "No internet service"])

with col4:
    streaming_movies = st.selectbox("Streaming Movies", 
                                   ["No", "Yes", "No internet service"])

# Contract and billing
st.subheader("Contract & Billing")
col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract Type", 
                           ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col6:
    payment_method = st.selectbox("Payment Method", 
                                 ["Electronic check", 
                                  "Mailed check", 
                                  "Bank transfer (automatic)", 
                                  "Credit card (automatic)"])

# Prepare input data
def prepare_input():
    # Initialize all features to 0
    input_dict = {
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }
    
    return pd.DataFrame([input_dict])

# Prediction
st.markdown("---")
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    input_data = prepare_input()
    
    # Get prediction probability
    pred_proba = model.predict_proba(input_data)[0][1]
    
    # Apply threshold
    prediction = 1 if pred_proba >= threshold else 0
    
    # Display results
    st.subheader("Prediction Results")
    
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK")
            st.metric("Churn Prediction", "YES")
        else:
            st.success("✅ LOW RISK")
            st.metric("Churn Prediction", "NO")
    
    with col_result2:
        st.metric("Churn Probability", f"{pred_proba:.2%}")
    
    with col_result3:
        st.metric("Model Threshold", f"{threshold:.3f}")
    
    # Progress bar
    st.write("Churn Risk Level:")
    st.progress(pred_proba)
    
    # Detailed explanation
    if pred_proba >= threshold:
        st.warning(f"""
        **Analysis:** This customer has a **{pred_proba:.2%}** probability of churning, 
        which exceeds the model threshold of {threshold:.3f}. 
        Consider retention strategies such as:
        - Offering discounts or promotions
        - Improving customer service
        - Providing upgrade options
        """)
    else:
        st.info(f"""
        **Analysis:** This customer has a **{pred_proba:.2%}** probability of churning, 
        which is below the model threshold of {threshold:.3f}. 
        The customer is likely to stay, but continue monitoring their satisfaction.
        """)

# Footer
st.markdown("---")
st.caption("Model: AdaBoost Classifier | Powered by Streamlit")