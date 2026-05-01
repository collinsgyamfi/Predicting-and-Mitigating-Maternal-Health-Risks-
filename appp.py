# app.py - Complete Maternal Health Risk Predictor
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import shap
import pickle
import os
import json

# --------------------------- 
# Imports
# ---------------------------
try:
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("Missing libraries. Run: pip install xgboost==2.1.3 scikit-learn shap")

# ---------------------------
# 1. AUTHENTICATION WITH PERMANENT STORAGE
# ---------------------------
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    users = load_users()
    hashed = hash_password(password)
    if username in users and users[username] == hashed:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        return True
    return False

def signup(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_password(password)
    save_users(users)
    return True

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None

# Initialize session state
if "users" not in st.session_state:
    st.session_state.users = load_users()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ---------------------------
# 2. LOAD MODELS
# ---------------------------
MODEL_PATH = "maternal_risk_models.pkl"

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    if not ML_AVAILABLE or not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------------------------
# 3. PREDICTION FUNCTION
# ---------------------------
def predict_with_shap(row_df, models, risk_idx=0):
    numeric_cols = ['age', 'parity', 'gestational_age', 'hemoglobin', 'bmi', 'glucose', 'distance_to_clinic']
    binary_cols = ['hypertension_hist', 'anemia_hist', 'diabetes_hist']

    for col in numeric_cols + binary_cols:
        if col in row_df.columns:
            if col in numeric_cols:
                row_df[col] = row_df[col].fillna(row_df[col].median())
            else:
                row_df[col] = row_df[col].fillna(row_df[col].mode()[0])

    X_num = models['scaler'].transform(row_df[numeric_cols])
    X_bin = row_df[binary_cols].values
    X = np.hstack([X_num, X_bin])
    X_df = pd.DataFrame(X, columns=models['feature_names'])

    preds = [round(float(model.predict_proba(X)[0, 1]), 3) for model in models['xgb_models']]
    shap_vals = models['explainers'][risk_idx].shap_values(X)[0]

    return preds, shap_vals, X_df.iloc[0]

# ---------------------------
# 4. LAYMAN EXPLANATIONS
# ---------------------------
risk_explanations = {
    "pre_eclampsia_risk": "Pre-eclampsia is a serious condition where the mother develops **very high blood pressure** during pregnancy. It can harm her kidneys, liver, and brain, and can also affect the baby. It needs urgent medical attention.",
    "gestational_diabetes_risk": "This is diabetes that starts during pregnancy. It can make the baby grow too large, cause delivery complications, and increase the mother's chance of developing diabetes later in life.",
    "preterm_birth_risk": "Preterm birth means the baby is born too early (before 37 weeks). Premature babies often need special care in hospital and may have breathing, feeding, or developmental problems.",
    "maternal_deterioration_risk": "This means the mother's health could suddenly become much worse — for example, severe bleeding, extremely high blood pressure, infection, or organ problems."
}

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Maternal Health Risk Predictor", layout="wide")

with st.sidebar:
    st.title("🔐 Authentication")
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if login(username, password):
                    st.success(f"Welcome {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with tab2:
            new_user = st.text_input("New Username", key="signup_user")
            new_pass = st.text_input("New Password", type="password", key="signup_pass")
            if st.button("Sign Up"):
                if signup(new_user, new_pass):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists")
    else:
        st.write(f"Logged in as: **{st.session_state.current_user}**")
        if st.button("Logout"):
            logout()
            st.rerun()

if not st.session_state.logged_in:
    st.stop()

st.title("🤰 Early Maternal Health Risk Detector")
st.write("**Purpose:** Predict maternal risks early → prevent deaths in rural Ghana.")
st.write("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
st.markdown("### SEFWI JUABOSO/BOINZAN DISTRICT (piloting) - Western North REGION - GHANA")

models = load_models()

uploaded_file = st.file_uploader("Upload Patient Data (CSV or Excel - any number of patients)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    clinical_data = df.iloc[:, :10].copy()
    clinical_data.columns = models['feature_names'][:10]

    # Generate predictions
    predictions = []
    for _, row in clinical_data.iterrows():
        row_df = pd.DataFrame([row])
        pred = predict_with_shap(row_df, models, 0)[0]
        predictions.append(pred)

    results_df = pd.DataFrame(predictions, columns=models['target_names']).round(3)

    # ==================== TABS ====================
    tab1, tab2 = st.tabs(["📊 Predictions", "Predicted Summary"])

    with tab1:
        st.subheader("Risk Predictions (Probability)")
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn_r', axis=None))

        # High-Risk Patients Summary
        st.subheader("⚠️ High-Risk Patients Summary")
        
        high_risk_patients = []
        for idx, row in results_df.iterrows():
            high_risks = [risk.replace("_", " ").title() for risk in models['target_names'] if row[risk] >= 0.5]
            if high_risks:
                high_risk_patients.append((idx, high_risks))
        
        if high_risk_patients:
            st.warning(f"🔴 **{len(high_risk_patients)} patient(s) identified with HIGH RISK:**")
            for idx, risks in high_risk_patients:
                st.markdown(f"- **Patient {idx+1}** : {', '.join(risks)}")
        else:
            st.success("✅ **No high-risk patients detected.** All patients show low risk levels.")
        
        # Force Plot for selected patient
        st.subheader("🔍 SHAP Force Plot Explanation")
        patient_idx = st.selectbox("Select Patient to Explain with Force Plot", 
                                   options=range(len(results_df)), 
                                   format_func=lambda x: f"Patient {x+1}")
        
        st.write(f"**Showing SHAP explanation for Patient {patient_idx + 1}**")
        
        sample_df = clinical_data.iloc[[patient_idx]].copy()
        highest_risk_idx = results_df.iloc[patient_idx].argmax()
        _, shap_vals, features = predict_with_shap(sample_df, models, risk_idx=highest_risk_idx)

        try:
            force_plot = shap.force_plot(
                models['explainers'][highest_risk_idx].expected_value,
                shap_vals,
                features,
                feature_names=models['feature_names'],
                matplotlib=False
            )
            st.components.v1.html(shap.getjs() + force_plot.html(), height=420, scrolling=True)
        except:
            st.warning("Could not display Force Plot.")

    with tab2:
        st.subheader("🗣️ Detailed Patient Summary")
        
        high_risk_count = (results_df >= 0.5).any(axis=1).sum()
        st.markdown(f"**Overview**: Out of **{len(results_df)}** patients, **{high_risk_count}** have at least one **High Risk**.")

        st.subheader("Patient-by-Patient Details")
        for idx, row in results_df.iterrows():
            high_risks = [risk for risk in models['target_names'] if row[risk] >= 0.5]
            has_high_risk = len(high_risks) > 0
            
            expander_label = f"Patient {idx + 1} {'🔴 HIGH RISK' if has_high_risk else '🟢 Low Risk'}"
            with st.expander(expander_label):
                st.write("**Risk Levels:**")
                for risk in models['target_names']:
                    prob = row[risk]
                    if prob >= 0.5:
                        status = "🔴 **High Risk**"
                        st.warning(f"- **{risk.replace('_', ' ').title()}**: {prob:.1%} → {status}")
                    elif prob >= 0.3:
                        status = "🟡 **Moderate Risk**"
                        st.info(f"- **{risk.replace('_', ' ').title()}**: {prob:.1%} → {status}")
                    else:
                        status = "🟢 **Low Risk**"
                        st.write(f"- **{risk.replace('_', ' ').title()}**: {prob:.1%} → {status}")

                if has_high_risk:
                    st.write("**What this means in simple words:**")
                    for risk in high_risks:
                        st.error(f"**{risk.replace('_', ' ').title()}** is HIGH: {risk_explanations[risk]}")
                else:
                    st.success("✅ No high-risk conditions detected for this patient.")

        st.subheader("📖 What Do These 4 Risks Mean?")
        for name, explanation in risk_explanations.items():
            st.markdown(f"**{name.replace('_', ' ').title()}**: {explanation}")

        st.subheader("💡 Recommendations")
        st.info("""
        - **High Risk (🔴)** patients should be referred to a hospital **immediately**.
        - **Moderate Risk** patients need **closer monitoring**.
        - All pregnant women should attend regular antenatal check-ups.
        - Early detection can save lives of both mother and baby.
        """)

else:
    st.info("👆 Please upload your patient data (CSV or Excel file). You can upload any number of patients.")

# Sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Required first 10 columns:**  
age, parity, gestational_age, hemoglobin, bmi, glucose,  
hypertension_hist, anemia_hist, diabetes_hist, distance_to_clinic
""")