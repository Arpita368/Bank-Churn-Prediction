🏦 Customer Churn Prediction System (End-to-End ML Pipeline)

A complete machine learning pipeline for predicting customer churn in a banking environment. The project includes exploratory data analysis (EDA), feature engineering, model training, evaluation, and prediction generation with a strong focus on recall and F2-score optimization to correctly identify high-risk churn customers.

📌 Problem Statement

Banks lose significant revenue when customers stop using their services. The goal of this project is to:

Predict whether a customer will churn (1) or stay (0)
Identify high-risk customers early
Reduce churn by improving retention strategies
📊 Dataset Overview
Dataset Name: ChurnZero Dataset v1
Records: ~10,000 customers
Target Variable: churn
0 → Retained Customer
1 → Churned Customer
Key Features
Demographics (Age, etc.)
Financial behavior (Balance, Credit Utilization)
Engagement metrics (Login activity, inactive days)
Complaints & satisfaction scores
Product ownership flags
🧠 Project Workflow
1️⃣ Exploratory Data Analysis (EDA)

Performed in eda.py

Data inspection (shape, missing values, duplicates)
Target distribution analysis
Feature-wise churn comparison using:
Boxplots (Age, Balance, Engagement, etc.)
Histograms (Age, Balance distributions)
Correlation heatmap using encoded features
Saved all visualizations in /graphs
2️⃣ Feature Engineering

Performed in model_training.py

New engineered features:

total_loan_products
total_products_owned
engagement_gap
credit_utilization_change
complaint_severity
transaction_drop_indicator

These features improve churn signal detection by capturing behavioral patterns.

3️⃣ Data Preprocessing Pipeline
Numerical features:
Median imputation
Standard scaling
Categorical features:
Most frequent imputation
One-hot encoding
Combined using ColumnTransformer
Fully wrapped inside a Scikit-learn Pipeline
4️⃣ Model Training

Multiple models evaluated:

Logistic Regression
Decision Tree
K-Nearest Neighbors
Gaussian Naive Bayes
SVM
MLP Neural Network
Random Forest
Bagging Classifier
Extra Trees
Gradient Boosting
AdaBoost
Histogram Gradient Boosting
5️⃣ Evaluation Strategy

Instead of accuracy, the project focuses on:

🎯 Recall (catch all churners)
🎯 F2-Score (recall-weighted metric)
ROC-AUC
PR-AUC
Precision (secondary metric)

A custom threshold of:

Threshold = 0.35

is used to improve churn detection sensitivity.

6️⃣ Best Model Selection
Models are compared using F2-score
Best-performing model is automatically selected
Saved as:
models/best_model.pkl
📈 Outputs Generated
📊 Graphs (/graphs)
Churn distribution
Age vs churn
Balance vs churn
Engagement vs churn
Complaints vs churn
Satisfaction vs churn
Credit utilization vs churn
Inactive days vs churn
Correlation heatmap
ROC curve
Confusion matrix
Feature importance plot
📁 Files (/outputs)
model_comparison.csv → Model performance comparison
feature_importance.csv → Feature ranking
final_metrics.json → Final evaluation metrics
classification_report.txt → Full report
ChurnZero_Team_Predictions.csv → Test predictions
🤖 Model Artifact
models/best_model.pkl → Trained production-ready model
🏆 Final Metrics (Best Model)

Tracked metrics:

Accuracy
Precision
Recall
F1 Score
F2 Score (primary metric)
ROC-AUC
PR-AUC
🧪 Prediction Output Format
customer_id	churn_prediction	churn_probability
1001	1	0.87
1002	0	0.12
⚙️ Tech Stack
Programming Language
Python 3.10+
Libraries
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
🏗️ Project Structure
project-root/
│
├── data/
│   ├── ChurnZero_Dataset_v1.csv
│   └── ChurnZero_Test_v1.csv
│
├── graphs/
│   ├── churn_distribution.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── final_metrics.json
│   ├── classification_report.txt
│   └── ChurnZero_Team_Predictions.csv
│
├── eda.py
├── model_training.py
└── README.md
🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run EDA
python eda.py
3. Train model & generate predictions
python model_training.py
💡 Key Highlights
End-to-end ML pipeline (EDA → Training → Prediction)
Strong focus on recall optimization for churn detection
Feature engineering for behavioral intelligence
Automated model comparison system
Production-ready pipeline with saved artifacts
Threshold tuning for real-world business impact
📌 Business Impact

This system helps banks:

Identify potential churners early
Improve customer retention strategies
Reduce revenue loss
Target high-risk customers with offers and interventions
