"""Simple Streamlit UI for Vocal-Guard.

This app collects numeric voice features from the user
and sends them to the prediction API.
"""

import streamlit as st
import requests

# App title and short description
st.title("🧠 Vocal-Guard")
st.write("Enter voice features to predict Parkinson’s risk")

# URL of the running FastAPI prediction service
API_URL = "http://127.0.0.1:8000/predict"

# Ordered list of feature names expected by the model
feature_names = [
"MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
"MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
"MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
"MDVP:Shimmer", "MDVP:Shimmer(dB)",
"Shimmer:APQ3", "Shimmer:APQ5",
"MDVP:APQ", "Shimmer:DDA",
"NHR", "HNR",
"RPDE", "DFA",
"spread1", "spread2",
"D2", "PPE"
]

features = []

st.subheader("Input Features")

for name in feature_names:
    # Show a numeric input for each feature. Keep 6 decimal places.
    val = st.number_input(name, value=0.0, format="%.6f", step=0.000001)
    # Store values as floats rounded to 6 decimal places to match model input
    val = round(float(val), 6)
    features.append(val)

if st.button("Predict"):
    # Prepare payload and call prediction API
    payload = {"features": features}

    response = requests.post(API_URL, json=payload)

    # Handle API response
    if response.status_code == 200:
        result = response.json()

        if result["class"] == 1:
            st.error("⚠ Parkinson’s Detected")
        else:
            st.success("✅ Healthy")
    else:
        st.warning("API Error — is backend running?")
