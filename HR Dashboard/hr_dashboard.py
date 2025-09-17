import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import streamlit as st
# --- Page Setup ---
st.set_page_config(page_title="AI-Powered HR Analytics Dashboard", layout="wide")
st.title("🧠 AI-Powered HR Analytics Dashboard")
st.markdown("Track employee performance and retention rates with AI-powered analytics.")

# Load and cache data
@st.cache_data
def load_data():
    path = r"C:\Users\dhara\Downloads\HR_Analytics_Dashboard\data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
    return pd.read_csv(path)

raw_df = load_data()
df = raw_df.copy()
# --- Preprocess Data ---
def preprocess(df):
    df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'])
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    return df_encoded, label_encoders

df_encoded, label_encoders = preprocess(df)
# --- Model Training ---
@st.cache_resource
def train_model(df_encoded):
    df_model = df_encoded.copy()
    y = df_model['Attrition']
    X = df_model.drop(columns=['Attrition', 'EmployeeNumber'])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc, X_test
# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
tabs = ["Overview", "Performance", "Demographics", "Predict", "Explain", "👤 Employee Profile", "💰 ELTV Leaderboard", "📅 Time-Based Turnover"]

selection = st.sidebar.radio("Go to", tabs)

# --- Overview Charts ---
if selection == "Overview":
    st.header("📊 Attrition Overview")
    attrition_counts = df['Attrition'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)



# --- Performance Charts ---
elif selection == "Performance":
    st.header("📈 Job Satisfaction & Performance")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['JobSatisfaction'], bins=4, kde=True, color='blue', ax=ax2)
    ax2.set_title('Job Satisfaction')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.histplot(df['PerformanceRating'], bins=4, kde=True, color='green', ax=ax3)
    ax3.set_title('Performance Rating')
    st.pyplot(fig3)

# --- Demographic Breakdown ---
elif selection == "Demographics":
    st.header("👥 Demographic Analysis")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df, x='Gender', hue='Attrition', ax=ax4)
    ax4.set_title('Gender vs Attrition')
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x='JobRole', hue='Attrition', ax=ax5)
    ax5.set_title('Job Role vs Attrition')
    plt.xticks(rotation=45)
    st.pyplot(fig5)
# --- Predict Attrition for New Employee ---
# --- Predict Attrition & Retention for New Employee ---
elif selection == "Predict":
    st.header("🧾 Predict Attrition & Retention for New Employee")
    model, scaler, acc, _ = train_model(df_encoded)

    st.markdown(f"Model trained with **{acc:.2f}** accuracy.")
    st.markdown("Enter employee details below to predict attrition and retention probability:")

    # Input fields
    age = st.slider("Age", 18, 60, 30)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years = st.slider("Years at Company", 0, 40, 5)
    dist = st.slider("Distance From Home (miles)", 1, 30, 10)
    satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    overtime = st.selectbox("OverTime", ['No', 'Yes'])

    if st.button("Predict"):
        # Create a sample row with default structure
        sample = df_encoded.drop(columns=['Attrition', 'EmployeeNumber']).iloc[0:1].copy()
        sample.iloc[0] = 0  # reset all values to 0
        sample['Age'] = age
        sample['JobLevel'] = job_level
        sample['MonthlyIncome'] = income
        sample['YearsAtCompany'] = years
        sample['DistanceFromHome'] = dist
        sample['JobSatisfaction'] = satisfaction
        sample['OverTime'] = 1 if overtime == 'Yes' else 0

        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled)[0]
        prob_attrition = model.predict_proba(sample_scaled)[0][1]
        prob_retention = 1 - prob_attrition

        # Display Results
        if pred:
            st.error(f"⚠️ Prediction: **Attrition Likely** with **{prob_attrition:.2%}** probability.")
        else:
            st.success(f"✅ Prediction: **No Attrition** with **{prob_retention:.2%}** retention probability.")

        # Retention metric card
        st.markdown("---")
        st.subheader("📈 Retention Probability Score")
        st.metric(label="Likelihood to Stay", value=f"{prob_retention:.2%}")

# --- SHAP Explainability ---
elif selection == "Explain":
    st.header("🔍 SHAP Feature Importance")
    model, _, _, X_test = train_model(df_encoded)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)
# --- Employee Profile Viewer ---
elif selection == "👤 Employee Profile":
    st.header("👤 View Individual Employee Details")

    employee_ids = raw_df['EmployeeNumber'].unique()
    selected_id = st.selectbox("Select Employee Number", sorted(employee_ids))

    emp_data = raw_df[raw_df['EmployeeNumber'] == selected_id].iloc[0]

    st.subheader("📌 Basic Info")
    st.write({
        "Age": emp_data['Age'],
        "Gender": emp_data['Gender'],
        "Job Role": emp_data['JobRole'],
        "Department": emp_data['Department'],
        "Distance From Home": emp_data['DistanceFromHome'],
    })

    st.subheader("💼 Job & Performance")
    st.write({
        "Monthly Income": emp_data['MonthlyIncome'],
        "Job Satisfaction": emp_data['JobSatisfaction'],
        "Performance Rating": emp_data['PerformanceRating'],
        "Years At Company": emp_data['YearsAtCompany'],
        "OverTime": emp_data['OverTime'],
    })

    st.subheader("📉 Attrition Status")
    st.write(f"Attrition: **{emp_data['Attrition']}**")
    if st.button("Predict Attrition & Retention for This Employee"):
        # Load model
        model, scaler, _, _ = train_model(df_encoded)
        emp_encoded = df_encoded[df['EmployeeNumber'] == selected_id].drop(columns=['Attrition', 'EmployeeNumber'])
        emp_scaled = scaler.transform(emp_encoded)
        pred = model.predict(emp_scaled)[0]
        prob_attrition = model.predict_proba(emp_scaled)[0][1]
        prob_retention = 1 - prob_attrition

        # Display prediction
        if pred:
            st.error(f"⚠️ Prediction: **Attrition Likely** with **{prob_attrition:.2%}** probability.")
        else:
            st.success(f"✅ Prediction: **No Attrition** with **{prob_retention:.2%}** retention probability.")

        # Retention score metric
        st.markdown("---")
        st.subheader("📈 Retention Probability Score")
        st.metric(label="Likelihood to Stay", value=f"{prob_retention:.2%}")
        # --- ELTV CALCULATION ---
        st.markdown("---")
        st.subheader("💡 Estimated Employee Lifetime Value (ELTV)")

        # Base values from data
        base_income = emp_data['MonthlyIncome']
        base_perf = emp_data['PerformanceRating']
        base_tenure = emp_data['YearsAtCompany']

        # Simulation sliders
        mod_income = st.slider("Modify Monthly Income ($)", 1000, 25000, int(base_income), step=500)
        mod_tenure = st.slider("Modify Tenure (Years)", 0, 40, int(base_tenure))

        # ELTV formula
        eltv_factor = 1.2  # Adjustable constant
        eltv = mod_income * base_perf * mod_tenure * eltv_factor

        st.metric(label="Estimated ELTV", value=f"${eltv:,.0f}")
        st.caption("Calculated as: Monthly Income × Performance Rating × Tenure × Factor (1.2)")


 # SHAP Force Plot
        explainer = shap.Explainer(model)
        shap_value = explainer(emp_scaled)
        st.subheader("🔍 SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_value[0], show=False)
        st.pyplot(fig)

# --- ELTV Leaderboard ---
elif selection == "💰 ELTV Leaderboard":
    st.header("💰 Employee Lifetime Value (ELTV) Leaderboard")

    # ELTV calculation function
    def calculate_eltv(row, factor=1.2):
        return row['MonthlyIncome'] * row['PerformanceRating'] * row['YearsAtCompany'] * factor

    eltv_df = raw_df.copy()
    eltv_df['ELTV'] = eltv_df.apply(calculate_eltv, axis=1)

    # Sort by ELTV
    eltv_df = eltv_df.sort_values(by='ELTV', ascending=False)

    # Display top N (optional)
    top_n = st.slider("Select number of top employees to view", 5, 50, 10)
    st.dataframe(
        eltv_df[['EmployeeNumber', 'JobRole', 'MonthlyIncome', 'PerformanceRating', 'YearsAtCompany', 'ELTV']]
        .head(top_n)
        .style.format({"MonthlyIncome": "${:,.0f}", "ELTV": "${:,.0f}"})
    )

    # Plot
    st.subheader("📊 Top ELTV Contributors")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=eltv_df.head(top_n), x='EmployeeNumber', y='ELTV', palette='viridis')
    ax.set_title('Top Employees by Lifetime Value')
    ax.set_ylabel("ELTV ($)")
    ax.set_xlabel("Employee Number")
    plt.xticks(rotation=45)
    st.pyplot(fig)
elif selection == "📅 Time-Based Turnover":
    st.header("📅 Time-Based Turnover Analysis")

    # --- Filter by Individual Employee ---
    st.subheader("🔍 View Individual Employee Timeline")
    employee_ids = df["EmployeeNumber"].unique()
    selected_emp_id = st.selectbox("Select Employee Number:", sorted(employee_ids))

    emp_data = df[df["EmployeeNumber"] == selected_emp_id].iloc[0]

    with st.expander("📊 Employee Time-Based Metrics"):
        st.write(f"**Employee Name:** {emp_data['EmployeeNumber']}")
        st.write(f"**Department:** {emp_data['Department']}")
        st.write(f"**Job Role:** {emp_data['JobRole']}")
        st.write(f"**Gender:** {emp_data['Gender']}")
        st.write("---")
        st.metric(label="Years at Company", value=emp_data['YearsAtCompany'])
        st.metric(label="Years in Current Role", value=emp_data['YearsInCurrentRole'])
        st.metric(label="Years Since Last Promotion", value=emp_data['YearsSinceLastPromotion'])
        st.metric(label="Years with Current Manager", value=emp_data['YearsWithCurrManager'])

    st.markdown("---")

    # --- Optional: Compare with Overall Distribution ---
    st.subheader("📈 Compare With Overall Distribution")

    cols = ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    for col in cols:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=col, hue="Attrition", kde=True, ax=ax, palette="Set2", element="step")
        ax.axvline(emp_data[col], color='red', linestyle='--', label='Selected Employee')
        ax.set_title(f'{col} Distribution vs Selected Employee')
        ax.legend()
        st.pyplot(fig)

    st.markdown("""
    💡 **Interpretation Tip:** Red dashed line represents the selected employee’s value.
    Use this to see how their career timeline compares with peers who stayed vs those who left.
    """)
