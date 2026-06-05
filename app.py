import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model("fraud_model.h5")

# Load sequence length
seq_len = joblib.load("seq_len.pkl")

st.title("💳 AI Fraud Detection System")

st.write("Upload transaction CSV (sequence format)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write("Preview:")
    st.dataframe(df.head())

    # remove label if exists
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    if st.button("Predict Fraud Risk"):

        data = df.values

        # reshape to sequence format
        try:
            sample = data[:seq_len].reshape(1, seq_len, -1)

            pred = model.predict(sample)[0][0]

            st.subheader("Fraud Probability")
            st.write(float(pred))

            if pred > 0.5:
                st.error("⚠ High Fraud Risk Detected")
            else:
                st.success("✔ Safe Transaction")

        except Exception as e:
            st.error("Error: Check input format")
            st.write(e)