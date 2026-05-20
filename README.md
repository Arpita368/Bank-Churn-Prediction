🏦 Banking Customer Churn Prediction System
<div align="center">












🚀 End-to-End Machine Learning System for Predicting Banking Customer Churn
<img src="https://img.shields.io/badge/Business%20Impact-High%20Revenue%20Retention-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Goal-Customer%20Retention%20Optimization-important?style=for-the-badge" /> </div>
📌 Project Overview

This project was built for the ChurnZero 26 Data Science Hackathon.

It is a production-style machine learning pipeline designed to predict whether a banking customer is likely to churn based on:

Behavioral patterns
Financial activity
Engagement metrics
Service interactions

Unlike standard ML projects, this system prioritizes:

🎯 Recall over Accuracy
💼 Business impact over metrics
🔍 Early churn detection
⚙️ Real-world deployment readiness
🎯 Business Problem Statement

Customer churn leads to major financial losses for banks:

📉 Loss of revenue streams
💳 Drop in product usage
🧾 Reduced customer lifetime value
📊 Increased acquisition cost
💡 Objective:

Predict churn early so banks can take proactive retention actions before customers leave.

📊 Dataset Overview
Attribute	Value
Total Customers	10,000
Features	14
Target	Churn (Exited)
Type	Binary Classification
🧠 Key Business Insights (EDA Results)
📉 Behavioral Insights
Low digital engagement → Higher churn risk
High inactivity (>90 days) → Strong churn signal
Low credit utilization → Passive customers likely to churn
High complaints → Direct churn indicator
Low satisfaction scores → Early warning signal
💰 Financial Insights
Low balance customers churn more frequently
High-balance customers contribute disproportionately to revenue
Credit usage patterns strongly influence retention
👤 Demographic Insights
Age distribution is NOT a strong churn factor
Majority customers are 35–55 years old
Demographics alone cannot predict churn
🔗 Correlation Insights
Weak linear correlation with churn
No single dominant feature
Churn depends on multi-variable interaction patterns
📊 Visual Insights
📌 Churn Distribution
Imbalanced dataset (~20% churn rate)
📌 Feature Relationships
Satisfaction ↓ → Churn ↑
Complaints ↑ → Churn ↑
Balance ↓ → Churn ↑
📌 Correlation Heatmap
Low multicollinearity
Strong feature clusters (income, balance, transactions)
⚠️ Critical Insight: Model Leakage Warning
🚨 Model Performance Concern

The model achieved:

ROC-AUC ≈ 1.0
Near-perfect classification
Extremely low error rate
⚠️ Interpretation:

This is a strong indicator of data leakage or overfitting

Possible causes:

Target-related features included accidentally
Post-churn information leakage
Overly predictive engineered features
📌 Action Required:

Feature audit must ensure:

Only pre-churn variables are used
No future-state information is included
🏗️ Machine Learning Pipeline
📥 Data Loading
      ↓
📊 Exploratory Data Analysis
      ↓
🧹 Data Cleaning & Preprocessing
      ↓
🔧 Feature Encoding & Scaling
      ↓
🤖 Model Training (Multiple Models)
      ↓
⚙️ Hyperparameter Tuning
      ↓
🎯 Threshold Optimization (F2 Focus)
      ↓
📈 Evaluation (Recall, ROC-AUC, F2)
      ↓
📊 Feature Importance Analysis
      ↓
📦 Model Export
🤖 Models Used
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting (Final Model)
Ensemble Methods
🏆 Final Model
🌟 Gradient Boosting Classifier

Selected based on:

Highest Recall
Best F2-score
Strong ROC-AUC
Stability across validation
📈 Model Performance
Metric	Score
Accuracy	~0.75
Precision	~0.44
Recall	~0.83
F1 Score	~0.57
F2 Score	~0.70
ROC-AUC	~0.87
🔥 Key Feature Drivers
🚀 Top Predictors
Customer Lifetime Value
Campaign Response Time
Satisfaction Score
Account Inactivity
Complaint Count
⚙️ Medium Impact Features
Service Requests
Relationship Manager Interaction
Transaction Behavior
📉 Low Impact Features
Age
Gender
Geography (low predictive power alone)
💼 Business Impact

This system enables banks to:

🔍 Identify churn risk early
📉 Reduce customer attrition
💰 Protect high-value customers
📞 Improve retention campaigns
📊 Optimize marketing targeting
📌 Core Takeaways
Churn is behavior-driven, not demographic-driven
Engagement drop is the earliest warning signal
Financial inactivity strongly correlates with churn
No single feature defines churn → multi-factor problem
High-value customers require priority retention
🚀 Future Scope
🔄 Real-time churn prediction API (FastAPI)
📊 Interactive Streamlit dashboard
⚡ XGBoost / LightGBM optimization
🧠 SHAP explainability integration
📈 Time-series customer behavior modeling
💬 NLP sentiment analysis from support data
🤖 AI-driven personalized retention engine
🛠️ Tech Stack
Python 🐍
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Joblib
📂 Project Structure
churn_project/
│
├── data/
├── notebooks/
├── models/
├── graphs/
├── outputs/
└── README.md
▶️ How to Run
# Step 1
python notebooks/eda.py

# Step 2
python notebooks/model_training.py
📊 Example Prediction
Prediction: 1 (Churn)
Probability: 0.31
🧾 Conclusion

This project demonstrates a real-world, end-to-end churn prediction system that combines:

Strong machine learning pipeline
Business-driven insights
Early warning detection system
Production-style workflow

The key insight: churn is not random — it is behaviorally predictable when engagement signals are tracked properly.

<div align="center">
⭐ If you like this project, don’t forget to star it ⭐
</div>
