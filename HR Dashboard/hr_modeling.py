import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(layout="wide")
st.title("🧠 AI-Powered HR Analytics Dashboard")
st.markdown("Track employee performance & attrition using machine learning.")

# Load Data
@st.cache_data
def load_data():
    path = r"C:\Users\dhara\Downloads\HR_Analytics_Dashboard\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
    return pd.read_csv(path)

# Preprocess
def preprocess(df):
    df = df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df

# Train Model
def train_model(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc, X_test

# SHAP values
def plot_shap_summary(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    st.subheader("Feature Importance (SHAP Summary Plot)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

# Main App
df = load_data()
st.sidebar.subheader("Data Summary")
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(df)

st.sidebar.markdown("### Model Options")
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        df_encoded = preprocess(df)
        model, acc, X_test = train_model(df_encoded)
        st.success(f"Model trained with Accuracy: {acc:.2f}")
        plot_shap_summary(model, X_test)
