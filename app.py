import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model, scaler, label encoder, and PCA
model = load_model('deep_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Streamlit app
st.title("BitcoinHeist Classification with Deep Learning")
st.write("Input the data to classify the Bitcoin address:")

# Create input fields for user inputs
st.header("Input Features")
input_data = {}
for feature in ['year', 'day', 'length', 'weight', 'count', 'looped',
       'neighbors', 'income']:  # Replace with actual feature names
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input data to DataFrame for processing
input_df = pd.DataFrame([input_data])

# Button to classify
if st.button("Classify"):
    # Scale and apply PCA
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    
    # Predict the class
    prediction = model.predict(input_pca)
    predicted_class = label_encoder.inverse_transform((prediction > 0.5).astype(int).flatten())[0]
    
    # Display the result
    st.subheader("Prediction")
    st.markdown(
    f"""
    <div style="background-color: #121212; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="color: #4CAF50; font-family: Arial, sans-serif;">Prediction Result</h3>
        <p style="font-size: 20px; color: #ffffff; font-family: Arial, sans-serif;">The predicted class is:</p>
        <h1 style="color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">{predicted_class}</h1>
    </div>
    """,
    unsafe_allow_html=True
    )
