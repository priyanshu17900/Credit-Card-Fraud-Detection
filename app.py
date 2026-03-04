import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

# Load dataset for demo transactions
df = pd.read_csv("creditcard.csv")

st.title("💳 Credit Card Fraud Detection System")

st.write(
"""
This application predicts whether a credit card transaction is **fraudulent or legitimate**
using a trained Machine Learning model.
"""
)

st.divider()

# Feature names from dataset
feature_names = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
"V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
"V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

st.subheader("Enter Transaction Features")

features = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    features.append(value)

features = np.array(features).reshape(1,-1)

if st.button("Predict Transaction"):

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠ Fraudulent Transaction (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability: {probability:.2f})")

st.divider()

st.subheader("Test With Real Dataset Transactions")

col1, col2 = st.columns(2)

with col1:

    if st.button("Load Random Legitimate Transaction"):

        sample = df[df["Class"] == 0].sample(1)

        st.write(sample.drop("Class", axis=1))

        pred = model.predict(sample.drop("Class", axis=1))

        if pred == 1:
            st.error("Fraud Detected")
        else:
            st.success("Legitimate Transaction")


with col2:

    if st.button("Load Random Fraud Transaction"):

        sample = df[df["Class"] == 1].sample(1)

        st.write(sample.drop("Class", axis=1))

        pred = model.predict(sample.drop("Class", axis=1))

        if pred == 1:
            st.error("Fraud Detected")
        else:
            st.success("Legitimate Transaction")

st.divider()

st.caption("Machine Learning Model: Random Forest | Dataset: Credit Card Fraud Detection")
st.divider()

st.subheader("📊 Fraud Analytics Dashboard")

# Fraud vs Normal distribution
st.write("### Fraud vs Normal Transactions")

fig1, ax1 = plt.subplots()

sns.countplot(x="Class", data=df, ax=ax1)

ax1.set_xticklabels(["Normal","Fraud"])

st.pyplot(fig1)


# Transaction Amount Distribution
st.write("### Transaction Amount Distribution")

fig2, ax2 = plt.subplots()

sns.histplot(df[df["Class"]==0]["Amount"], bins=50, color="blue", label="Normal", ax=ax2)

sns.histplot(df[df["Class"]==1]["Amount"], bins=50, color="red", label="Fraud", ax=ax2)

ax2.legend()

st.pyplot(fig2)


# Correlation Heatmap
st.write("### Feature Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)

st.pyplot(fig3)