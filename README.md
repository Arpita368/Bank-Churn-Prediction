🏦 Banking Customer Churn Prediction System
<div align="center">








🚀 End-to-End Machine Learning System for Predicting Banking Customer Churn
</div>
📌 Project Overview

This project was developed for the ChurnZero 26 Data Science Hackathon.

The goal is to predict whether a banking customer will churn using behavioral, financial, and engagement-based features.

Unlike traditional ML projects focused only on accuracy, this system emphasizes:

🎯 Recall & F2-score optimization
💼 Business-driven feature insights
🔍 Threshold tuning for early churn detection
⚙️ End-to-end ML pipeline
📊 Deep exploratory data analysis
🎯 Business Problem

Customer churn leads to major revenue loss for banks due to:

Reduced customer lifetime value
Loss of high-value accounts
Increased acquisition costs
Reduced cross-sell opportunities

This system helps banks:

Identify at-risk customers early and take proactive retention actions before they leave.

🧠 Key Insights from EDA
📊 Behavioral Drivers of Churn
Low digital engagement strongly increases churn risk
High account inactivity (90+ days) signals disengagement
Low credit utilization indicates inactive customers
More complaints = higher probability of churn
Lower satisfaction scores directly correlate with churn
💰 Financial Patterns
Low account balance customers are more likely to churn
High-balance customers contribute disproportionately to revenue
Credit utilization is a strong behavioral indicator
👤 Demographics
Age shows no meaningful impact on churn
Customer base is concentrated in the 35–55 age group
🔗 Correlation Insights
Most features show weak linear correlation with churn
Churn depends on multi-factor behavioral interactions
Dataset has low multicollinearity overall
⚠️ Model Performance Insight (Important)
🚨 Warning: Possible Data Leakage

The model achieved:

ROC-AUC ≈ 1.0
Near-perfect confusion matrix results

This indicates:

Potential data leakage or overly predictive features were present.

A strict feature audit is required to ensure only pre-decision variables are used.

🏗️ Project Pipeline
Data Collection
      ↓
EDA & Business Insights
      ↓
Data Cleaning & Encoding
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Threshold Optimization
      ↓
Evaluation (Recall, F2, ROC-AUC)
      ↓
Feature Importance Analysis
      ↓
Prediction Output
🤖 Models Used
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost / Ensemble Methods (if applicable)

Final selection prioritized:

High Recall
High F2-score
Stable generalization
📈 Key Feature Drivers
🔥 Top Predictors
Customer Lifetime Value
Campaign Response Time
Satisfaction Score
Account Inactivity
Complaint Count
⚙️ Mid-Level Drivers
Service requests
Relationship manager interactions
Transaction behavior
📉 Low Impact Features
Demographics (Age, Gender)
Marketing exposure signals
📊 Core Business Findings
Churn is driven mainly by behavior, not demographics
Engagement drop is the earliest churn indicator
Financial inactivity strongly correlates with churn
High-value customers must be prioritized for retention
Churn is a non-linear multi-factor problem
📌 Model Evaluation Summary
ROC-AUC: ~1.0 (flagged for leakage risk)
Very low false positives and negatives
Extremely high separation performance

Despite strong metrics, real-world validation is required due to potential overfitting.

💼 Business Impact

This system enables banks to:

🔍 Detect churn risk early
📉 Reduce customer attrition
💰 Protect high-value customers
📊 Improve customer engagement strategy
📞 Enable proactive support interventions
🚀 Future Scope
Real-time churn prediction API
Streamlit dashboard for business teams
XGBoost / LightGBM optimization
SHAP-based explainability
Time-series customer behavior tracking
NLP sentiment analysis from support data
AI-driven personalized retention campaigns
🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Joblib
📂 Project Structure
churn_project/
│
├── data/
├── notebooks/
├── models/
├── outputs/
├── graphs/
└── README.md
▶️ How to Run
python notebooks/eda.py
python notebooks/model_training.py
📌 Conclusion

This project demonstrates a complete end-to-end churn prediction system with strong business alignment.

It highlights that:

Customer churn is not driven by one factor, but by combined behavioral signals that require advanced machine learning models to capture effectively.

<div align="center">
⭐ If you like this project, consider starring it ⭐
</div>
