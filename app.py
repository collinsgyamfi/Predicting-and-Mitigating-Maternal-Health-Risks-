# app.py
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import shap
import matplotlib.pyplot as plt
import pickle
import os

# --------------------------- 
# Model Imports
# ---------------------------
try:
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("Required ML libraries not installed. Run: pip install xgboost scikit-learn shap")

# ---------------------------
# 1. MOCK AUTHENTICATION
# ---------------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    hashed = hash_password(password)
    if username in st.session_state.users and st.session_state.users[username] == hashed:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        return True
    return False

def signup(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = hash_password(password)
    return True

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None

# ---------------------------
# 2. LOAD / SAVE MODELS + SHAP EXPLAINERS
# ---------------------------
MODEL_PATH = "maternal_models.pkl"

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    if not ML_AVAILABLE:
        return {"fallback_mode": True, 
                "target_names": ['pre_eclampsia_risk', 'gestational_diabetes_risk',
                                 'preterm_birth_risk', 'maternal_deterioration_risk']}

    # Load models if they exist
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Otherwise, train and save
    st.info("Training models for the first time... This may take 30-60 seconds.")
    
    np.random.seed(42)

    def generate_synthetic_data(n_samples=2500):
        age = np.random.normal(28, 6, n_samples).clip(15, 45)
        parity = np.random.poisson(1.5, n_samples).clip(0, 8)
        gest_age = np.random.normal(28, 8, n_samples).clip(6, 42)
        hemoglobin = np.random.normal(11.5, 1.5, n_samples).clip(7, 16)
        bmi = np.random.normal(24, 5, n_samples).clip(16, 45)
        glucose = np.random.normal(90, 15, n_samples).clip(60, 200)
        hypertension_hist = np.random.binomial(1, 0.15, n_samples)
        anemia_hist = np.random.binomial(1, 0.20, n_samples)
        diabetes_hist = np.random.binomial(1, 0.10, n_samples)
        distance = np.random.lognormal(1.5, 0.8, n_samples).clip(0.5, 50)

        clinical_df = pd.DataFrame({
            'age': age, 'parity': parity, 'gestational_age': gest_age,
            'hemoglobin': hemoglobin, 'bmi': bmi, 'glucose': glucose,
            'hypertension_hist': hypertension_hist, 'anemia_hist': anemia_hist,
            'diabetes_hist': diabetes_hist, 'distance_to_clinic': distance
        })

        target_names = ['pre_eclampsia_risk', 'gestational_diabetes_risk',
                        'preterm_birth_risk', 'maternal_deterioration_risk']
        y = np.zeros((n_samples, 4))
        for i in range(n_samples):
            y[i, 0] = 1 if np.random.rand() < 0.18 else 0
            y[i, 1] = 1 if np.random.rand() < 0.22 else 0
            y[i, 2] = 1 if np.random.rand() < 0.15 else 0
            y[i, 3] = 1 if np.random.rand() < 0.12 else 0

        clinical_df = clinical_df.apply(lambda x: x.mask(np.random.rand(len(x)) < 0.05, np.nan))
        return clinical_df, y, target_names

    clinical_df, y, target_names = generate_synthetic_data()

    numeric_cols = ['age', 'parity', 'gestational_age', 'hemoglobin', 'bmi', 'glucose', 'distance_to_clinic']
    binary_cols = ['hypertension_hist', 'anemia_hist', 'diabetes_hist']
    feature_names = numeric_cols + binary_cols

    for col in numeric_cols:
        clinical_df[col].fillna(clinical_df[col].median(), inplace=True)
    for col in binary_cols:
        clinical_df[col].fillna(clinical_df[col].mode()[0], inplace=True)

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(clinical_df[numeric_cols])
    X_binary = clinical_df[binary_cols].values
    X_clinical = np.hstack([X_numeric, X_binary])

    # Train XGBoost + Create SHAP explainers
    xgb_models = []
    explainers = []
    for i in range(4):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_clinical, y[:, i])
        xgb_models.append(model)
        
        explainer = shap.TreeExplainer(model)
        explainers.append(explainer)

    # Save everything
    model_dict = {
        'scaler': scaler,
        'xgb_models': xgb_models,
        'explainers': explainers,
        'feature_names': feature_names,
        'target_names': target_names,
        'fallback_mode': False
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_dict, f)

    st.success("Models trained and saved successfully!")
    return model_dict

# ---------------------------
# 3. PREDICTION + SHAP FUNCTION
# ---------------------------
def predict_risk_with_shap(clinical_row_df, models_dict, risk_idx=0):
    if models_dict.get('fallback_mode', False):
        return [round(np.random.uniform(0.1, 0.7), 3) for _ in range(4)], None, None

    numeric_cols = ['age', 'parity', 'gestational_age', 'hemoglobin', 'bmi', 'glucose', 'distance_to_clinic']
    binary_cols = ['hypertension_hist', 'anemia_hist', 'diabetes_hist']

    # Impute
    for col in numeric_cols:
        if col in clinical_row_df.columns:
            clinical_row_df[col] = clinical_row_df[col].fillna(clinical_row_df[col].median())
    for col in binary_cols:
        if col in clinical_row_df.columns:
            clinical_row_df[col] = clinical_row_df[col].fillna(clinical_row_df[col].mode()[0])

    X_num = models_dict['scaler'].transform(clinical_row_df[numeric_cols])
    X_bin = clinical_row_df[binary_cols].values
    X = np.hstack([X_num, X_bin])
    X_df = pd.DataFrame(X, columns=models_dict['feature_names'])

    preds = []
    for model in models_dict['xgb_models']:
        prob = model.predict_proba(X)[0, 1]
        preds.append(round(float(prob), 3))

    explainer = models_dict['explainers'][risk_idx]
    shap_values = explainer.shap_values(X)

    return preds, shap_values[0], X_df.iloc[0]

# ---------------------------
# 4. STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Maternal Health Risk Predictor", layout="wide")

# Authentication
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
            new_user = st.text_input("Username", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
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
    st.info("👈 Please login or sign up from the sidebar.")
    st.stop()

st.title("🤰 Maternal Health Risk Predictor")
st.markdown("### Rural Ghana - Western North Region")

models = load_models()

uploaded_file = st.file_uploader("Upload patient data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        clinical_cols = models['feature_names'][:10]
        clinical_data = df.iloc[:, :10].copy()
        clinical_data.columns = clinical_cols

        # Predictions
        results = []
        for _, row in clinical_data.iterrows():
            row_df = pd.DataFrame([row])
            pred = predict_risk_with_shap(row_df, models, risk_idx=0)[0]
            results.append(pred)

        results_df = pd.DataFrame(results, columns=models['target_names'])
        
        st.subheader("🔮 Risk Predictions (Probability)")
        st.dataframe(results_df.style.background_gradient(cmap='RdYlGn_r', axis=None))

        st.subheader("⚠️ Risk Alerts (≥ 50% = High Risk)")
        st.dataframe((results_df >= 0.5).astype(int))

        # ==================== SHAP EXPLAINABILITY ====================
        st.subheader("🔍 Explainable AI - Why was this risk predicted?")

        selected_risk = st.selectbox("Select risk to explain:", models['target_names'])
        risk_idx = models['target_names'].index(selected_risk)

        if len(clinical_data) > 0:
            sample_df = clinical_data.iloc[[0]].copy()   # Explain first patient
            probs, shap_vals, features = predict_risk_with_shap(sample_df, models, risk_idx)

            st.write(f"**Predicted probability for {selected_risk.replace('_', ' ').title()}: {probs[risk_idx]:.1%}**")

            # Force Plot
            st.write("**Force Plot**")
            force_plot = shap.force_plot(
                models['explainers'][risk_idx].expected_value,
                shap_vals,
                features,
                feature_names=models['feature_names'],
                matplotlib=False
            )
            st.components.v1.html(f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>", 
                                height=320, scrolling=True)

            # Additional Plots
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Waterfall Plot**")
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_vals,
                                     base_values=models['explainers'][risk_idx].expected_value,
                                     data=features.values,
                                     feature_names=models['feature_names']),
                    max_display=10, show=False
                )
                st.pyplot(fig)

            with col2:
                st.write("**Feature Importance**")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                shap.plots.bar(
                    shap.Explanation(values=shap_vals,
                                     base_values=models['explainers'][risk_idx].expected_value,
                                     data=features.values,
                                     feature_names=models['feature_names']),
                    max_display=10, show=False
                )
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a CSV/Excel file or try the demo below.")
    
    if st.button("📊 Generate Demo Patient"):
        # === Improved Risky Demo Patient ===
        demo_data = pd.DataFrame([{
            'age': 37, 
            'parity': 4, 
            'gestational_age': 28, 
            'hemoglobin': 9.2,
            'bmi': 34.5, 
            'glucose': 142, 
            'hypertension_hist': 1,
            'anemia_hist': 1, 
            'diabetes_hist': 1, 
            'distance_to_clinic': 22.0
        }])

        pred_probs, shap_vals, features = predict_risk_with_shap(demo_data, models, risk_idx=0)
        
        st.subheader("Demo Patient Risk Assessment")
        for name, prob in zip(models['target_names'], pred_probs):
            delta = "🔴 High Risk" if prob >= 0.5 else "🟢 Low Risk"
            st.metric(
                label=name.replace("_", " ").title(), 
                value=f"{prob:.1%}", 
                delta=delta
            )

        # Show SHAP for the highest risk
        highest_risk_idx = np.argmax(pred_probs)
        st.info(f"Showing SHAP explanation for: **{models['target_names'][highest_risk_idx].replace('_', ' ').title()}**")

        _, shap_vals, features = predict_risk_with_shap(demo_data, models, risk_idx=highest_risk_idx)

        st.subheader("🔍 SHAP Explanation")
        st.write(f"**Predicted Probability**: {pred_probs[highest_risk_idx]:.1%}")

        try:
            # Force Plot
            force_plot = shap.force_plot(
                models['explainers'][highest_risk_idx].expected_value,
                shap_vals,
                features,
                feature_names=models['feature_names'],
                matplotlib=False
            )
            st.components.v1.html(
                f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>", 
                height=380, scrolling=True
            )
        except Exception as e:
            st.error(f"Could not display Force Plot: {e}")
            st.write("SHAP explanation is available but rendering failed.")

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.info("""
**Expected first 10 columns:**  
age, parity, gestational_age, hemoglobin, bmi, glucose,  
hypertension_hist, anemia_hist, diabetes_hist, distance_to_clinic
""")