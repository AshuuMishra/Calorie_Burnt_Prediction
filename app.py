

import streamlit as st
# import numpy as np
import joblib

# Load model and transformers
model = joblib.load("RFR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# Debug info (temporarily)
st.write("Encoder expects:", encoder.classes_)
st.write("You selected:", gender)

st.title("🔥 Calories Burnt Prediction App")
st.markdown("Enter your workout details below:")

# Input form
gender = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", 10, 80, 25)
height = st.number_input("Height (cm)", 100, 250, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
duration = st.slider("Duration (minutes)", 0, 300, 60)
heart_rate = st.slider("Heart Rate (bpm)", 50, 200, 120)
body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

if st.button("Predict Calories Burnt"):
    # Process input
    gender_encoded = encoder.transform([gender])[0]
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Calories Burnt: 🔥 **{prediction:.2f} kcal**")