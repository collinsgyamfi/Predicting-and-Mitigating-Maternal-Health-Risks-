# Predicting-and-Mitigating-Maternal-Health-Risks-
Predicting and Mitigating Maternal Health Risks Using Multimodal Machine Learning with Wearable Device Data and Explainable AI
# 🤰 Maternal Health Risk Predictor

## Early Detection of Pregnancy Complications 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-green.svg)](https://xgboost.ai/)

---

## 📌 Overview

This project aims to **reduce maternal mortality**  by providing an AI-powered early warning system for pregnant women. The system combines **wearable device data** (heart rate, blood pressure, oxygen levels, sleep, activity) with **clinical data** (age, BMI, medical history) to predict four major pregnancy risks:

- ⚠️ **Pre-eclampsia** (dangerous high blood pressure)
- 🩸 **Gestational Diabetes** (diabetes during pregnancy)
- 👶 **Preterm Birth** (baby born too early)
- 🏥 **Maternal Deterioration** (sudden health worsening)

The model is designed specifically for **low-resource rural settings** – lightweight, offline-capable, and interpretable for non-specialist clinicians.

---

## 🎯 Why This Matters

In the **Western North Region of Ghana**:
- Long distances to healthcare facilities
- Limited access to specialists
- Late detection of pregnancy complications

**Our solution:** A simple, offline mobile/web app that health workers can use to screen patients using basic wearable sensors and routine clinical data.Disclaimer:for educational purpose  not standard to deploy in real-world processes, are correct to follow and practice 

---

## 🏗️ Project Structure
aternal-health-risk-predictor/
│
├── app.py # Streamlit web application
├── train_model.py # Train the AI model (run once)
├── generate_dataset.py # Generate synthetic dataset
├── maternal_risk_models.pkl # Trained model file (created after training)
├── users.json # User credentials (auto-created)
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── data/
│ ├── maternal_health_synthetic_dataset.csv # Full dataset (5000 patients)
│ └── maternal_health_sample_100.csv # Sample dataset (100 patients)
│
└── screenshots/
├── dashboard.png # App interface
└── shap_explanation.png # SHAP force plot example



---

## 🔬 Model Architecture

The system uses an **ensemble of 3 models** combined by a meta-learner:

| Component | Role | Example |
|-----------|------|---------|
| **1D-CNN + Transformer** | Finds patterns in wearable time-series | "Heart rate spiked at 6 AM and 9 PM" |
| **XGBoost** | Learns rules from summary statistics | "High BMI + low sleep → higher risk" |
| **Logistic Regression** | Simple linear baseline | "Age + blood pressure linear combination" |
| **Meta-Learner** | Combines the 3 models optimally | "Trust XGBoost for diabetes, trust NN for pre-eclampsia" |

**Explainability:** SHAP values show WHY each patient received their risk score (e.g., "High risk because of hypertension history and low hemoglobin").

---

## 📊 Performance Metrics

| Risk Target | Precision | Recall | F1-Score | AUROC |
|-------------|-----------|--------|----------|-------|
| Pre-eclampsia | 0.76 | 0.71 | 0.73 | 0.83 |
| Gestational Diabetes | 0.81 | 0.68 | 0.74 | 0.86 |
| Preterm Birth | 0.72 | 0.78 | 0.75 | 0.81 |
| Maternal Deterioration | 0.74 | 0.73 | 0.73 | 0.79 |
| **Macro Average** | **0.76** | **0.72** | **0.74** | **0.82** |

**Ensemble improvement:** +3% AUROC over single neural network.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for first run (to download packages)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/maternal-health-risk-predictor.git
cd maternal-health-risk-predictor
