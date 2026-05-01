# 🚀 AI-Based Alternative Credit Scoring System

An AI-driven credit scoring system designed to evaluate borrowers with **no traditional credit history** using alternative data such as behavioral, financial, and psychometric signals.

---

## 📌 Problem Statement

Many individuals and MSMEs lack formal credit history, making it difficult for them to access loans. Traditional banking systems struggle to assess their risk, leading to financial exclusion.

---

## 💡 Solution

We built an **AI-powered credit scoring system** that uses alternative data to:
- Assess borrower risk
- Provide explainable predictions
- Enable fair and inclusive lending

---

## 🧠 Key Features

- Multi-model ensemble (Logistic, LightGBM, XGBoost, Neural Network)
- Multi-level stacking for improved accuracy
- Feature engineering from behavioral & financial data
- Consent-based data usage (toggle system)
- Privacy protection (hashed user IDs)
- Explainable predictions (reason-based output)
- FastAPI-based real-time inference API
- Saved models for fast predictions (no retraining)

---

## 📊 Data Sources

- Phone payment behavior  
- E-commerce activity  
- Geolocation stability  
- Psychometric questionnaire  
- Merchant ratings  
- Financial/cash flow patterns  
- Synthetic data (for simulation)

> ⚠️ Synthetic features were used for experimentation and can be removed in real-world deployment.

---

## ⚙️ Tech Stack

- Python  
- Scikit-learn  
- LightGBM  
- XGBoost  
- FastAPI  
- Pandas  
- NumPy  

---

## 🏗️ Model Architecture

```
Raw Data
↓
Feature Engineering
↓
Base Models
├── Logistic Regression
├── LightGBM
├── XGBoost
└── Neural Network
↓
Level 1 Stacking (K-Fold)
↓
Level 2 Meta Models (LGB + XGB)
↓
Final Prediction

```

