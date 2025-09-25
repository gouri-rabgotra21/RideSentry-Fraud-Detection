import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Data ---
# Use st.cache_data to load the model only once
@st.cache_data
def load_model():
    model = joblib.load('fraud_model.joblib')
    return model

model = load_model()

df = pd.read_csv('rideshare_fraud_data.csv')

# --- Dashboard Title ---
st.title("RideSentry - Fraud Detection Dashboard")

# --- Sidebar for User Input ---
st.sidebar.header("Input Features for Prediction")

# Create sliders and input boxes for the features our model needs
# We'll use some reasonable min/max values based on the data
fare = st.sidebar.slider("Fare Amount ($)", 5.0, 200.0, 50.0)
user_trip_count = st.sidebar.number_input("User's Previous Trip Count", min_value=0, max_value=500, value=10)
seconds_since_signup = st.sidebar.slider("Seconds Since User Signup", 1000, 15000000, 500000)
card_degree = st.sidebar.slider("Payment Card's Network Degree", 1, 50, 5)
device_degree = st.sidebar.slider("Device's Network Degree", 1, 50, 5)

# --- Prediction Logic ---
# A button to trigger the prediction
if st.sidebar.button("Predict Fraud Risk"):
    
    # Create a numpy array from the inputs in the correct order
    # Note: We're simplifying and leaving out some features for this UI
    features = [
        fare,
        seconds_since_signup,
        user_trip_count,
        1, # Placeholder for user_trips_last_1h
        2, # Placeholder for num_users_on_device
        device_degree,
        card_degree
    ]
    
    # The model expects a 2D array, so we reshape
    prediction_features = np.array(features).reshape(1, -1)
    
    # Get the prediction probability
    probability = model.predict_proba(prediction_features)[0][1]
    
    st.subheader("Prediction Result")
    if probability > 0.5: # Example threshold
        st.error(f"High Fraud Risk Detected! (Probability: {probability:.2f})")
    else:
        st.success(f"Low Fraud Risk Detected. (Probability: {probability:.2f})")

# --- Display Raw Data ---
st.subheader("Raw Data Sample")
st.dataframe(df.head())
import joblib

# 'model' is your final trained LightGBM model object from the previous cell
joblib.dump(model, 'fraud_model.joblib')

print("Model saved successfully as fraud_model.joblib!")