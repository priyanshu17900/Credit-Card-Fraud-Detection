# Credit Card Fraud Detection System

## Overview
This project implements a machine learning system to detect fraudulent credit card transactions.  
The model is trained on the **Credit Card Fraud Detection dataset** containing 284,807 transactions.

Due to extreme class imbalance (~0.17% fraud cases), techniques such as **SMOTE** were used to improve model performance.

## Features
- Data preprocessing and imbalance handling
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model comparison using Precision, Recall, and F1 Score
- Fraud prediction system
- Interactive web application using Streamlit
- Fraud analytics dashboard

## Dataset
Dataset source: Kaggle Credit Card Fraud Detection Dataset.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Features include:
- Time
- V1–V28 (PCA-transformed features)
- Amount
- Class (Fraud / Normal)

## Model Performance

| Model | Precision | Recall | F1 Score |
|------|------|------|------|
| Logistic Regression | 0.082 | 0.897 | 0.151 |
| Random Forest | 0.872 | 0.836 | 0.854 |
| XGBoost | 0.768 | 0.846 | 0.806 |

Random Forest achieved the best balance between precision and recall.

## Web Application

The system includes a Streamlit web interface for:

- Transaction fraud prediction
- Random transaction testing
- Fraud analytics dashboard

Run the application:  streamlit run app.py


## Technologies Used

- Python
- Scikit-Learn
- XGBoost
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn

## Future Improvements

- Real-time fraud detection pipeline
- Deep learning models
- SHAP model explainability
- Cloud deployment
