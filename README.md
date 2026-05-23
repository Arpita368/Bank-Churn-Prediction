# 🏦 Customer Churn Prediction System (End-to-End Machine Learning Project)

An advanced end-to-end machine learning system developed to predict **customer churn in a banking environment** using customer behavioral, financial, engagement, and complaint-related data.

This project covers the complete ML lifecycle including:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Multiple Model Training
- Model Evaluation & Comparison
- Business Cost Analysis
- Prediction Generation
- Reporting & Visualization

The system is optimized not only for high accuracy but also for **real-world business impact**, prioritizing **Recall** and **F2-score** to maximize churn detection and minimize customer revenue loss.

---

# 📌 Problem Statement

Customer churn is one of the most critical business problems in the banking sector. Losing customers directly affects:

- Revenue
- Customer trust
- Long-term business growth
- Retention costs

The objective of this project is to:

- Predict whether a customer will churn (`1`) or stay (`0`)
- Identify high-risk customers early
- Support proactive retention strategies
- Reduce financial losses due to churn

---

# 📊 Dataset Overview

The dataset contains structured banking customer information with behavioral, transactional, financial, and engagement-related features.

### Dataset Information

- **Dataset Name:** ChurnZero Dataset v1
- **Training Records:** 8,101 customers
- **Test Records:** 2,026 customers
- **Features:** 97 attributes
- **Target Variable:** `churn`

### Target Definition

- `0 → Retained Customer`
- `1 → Churned Customer`

---

# 🧠 Project Workflow

```text
Raw Dataset
↓
EDA & Visualization
↓
Data Cleaning
↓
Feature Engineering
↓
Encoding + Scaling
↓
Train-Test Split
↓
Multiple ML Models Training
↓
Model Evaluation & Comparison
↓
Best Model Selection
↓
Prediction Generation
↓
Business Reports & Outputs
```

---

# 🔍 1. Exploratory Data Analysis (EDA)

Performed in:

```bash
notebooks/eda.py
```

### 📌 EDA Tasks

- Dataset overview
- Shape & column inspection
- Missing value analysis
- Duplicate row detection
- Statistical summary
- Churn distribution analysis

### 📊 Visualizations Generated

- Customer churn distribution
- Age vs churn
- Average balance vs churn
- Digital engagement vs churn
- Complaints vs churn
- Satisfaction score vs churn
- Retention offer acceptance vs churn
- Last login days vs churn
- Credit utilization vs churn
- Inactive days vs churn
- Age distribution
- Balance distribution
- Correlation heatmap

All graphs are automatically saved in:

```bash
graphs/
```

---

# 🧹 2. Data Cleaning & Preprocessing

### Data Cleaning

- Missing values handled using median imputation
- Duplicate records checked and removed
- Irrelevant identifiers like `customer_id` excluded
- Dataset consistency validation performed

### Preprocessing

- One-hot encoding for categorical features
- Binary encoding for gender
- StandardScaler used for feature normalization
- Stratified train-test split applied

### Outcome

A clean and machine-learning-ready dataset prepared for robust model training and evaluation.

---

# ⚙️ 3. Feature Engineering

Advanced behavioral and banking-related features were created to improve predictive performance.

### 🧩 Engineered Features

### `total_loan_products`
Total loan products used by the customer.

### `total_products_owned`
Total banking products owned.

### `engagement_gap`
Difference between login activity and inactivity.

### `complaint_severity`
Combined impact of complaints and unresolved complaints.

These engineered features helped improve churn detection performance significantly.

---

# 🤖 4. Machine Learning Models Used

Multiple models were trained and evaluated.

## 📌 Models Implemented

### Linear Model
- Logistic Regression

### Tree-Based Models
- Decision Tree
- Random Forest
- Hist Gradient Boosting

---

# 🏆 5. Best Model Selected

## ✅ Hist Gradient Boosting Classifier

The best-performing model selected based on:

- Recall
- F2-score
- ROC-AUC
- Business impact

### Why This Model?

Hist Gradient Boosting achieved:

- Excellent churn detection
- Near-perfect class separation
- Strong recall-focused performance
- High real-world reliability

---

# 📏 6. Evaluation Strategy

Instead of focusing only on accuracy, the project prioritizes **business-oriented evaluation metrics**.

## 🎯 Key Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- F2 Score
- ROC-AUC
- PR-AUC

---

# 🎯 Why Recall & F2 Score Matter

In churn prediction:

> Missing a churn customer is much more costly than incorrectly predicting churn.

Therefore:

- Recall was prioritized
- F2-score used as the main optimization metric
- False negatives minimized aggressively

---

# 📊 7. Final Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.9975 |
| Precision | 1.0000 |
| Recall | 0.9847 |
| F1-Score | 0.9923 |
| F2-Score | 0.9877 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9998 |

---

# 📉 8. Confusion Matrix Interpretation

| | Predicted Stayed | Predicted Churned |
|---|---|---|
| Actual Stayed | 1360 | 0 |
| Actual Churned | 4 | 257 |

### Key Observations

- Only **4 churn customers were missed**
- **0 false positives** generated
- The model captures almost all churners successfully
- Extremely strong separation capability observed

---

# ⚠️ 9. Business Cost Analysis

### Cost Assumptions

- False Negative → ₹40,000 loss
- False Positive → ₹500 retention cost

### Business Outcome

- False Negatives = 4 → ₹1,60,000 potential loss
- False Positives = 0 → ₹0 unnecessary retention spending

### Why Recall Was Prioritized

The financial impact of missing a churn customer is significantly higher than wrongly targeting a retained customer.

Therefore, maximizing Recall becomes critical for reducing revenue leakage.

---

# 📤 10. Prediction Generation System

Performed in:

```bash
notebooks/generate_predictions.py
```

### Generated Output

The system generates predictions for all test customers.

### Output Columns

| customer_id | churn_prediction | churn_probability |
|---|---|---|
| 1001 | 1 | 0.92 |
| 1002 | 0 | 0.08 |

### Final Result

- Predictions generated for **2026 customers**
- Risk probabilities assigned to each customer

Outputs saved in:

```bash
outputs/predictions.csv
```

---

# 📊 11. Outputs Generated

## 📁 Graphs (`graphs/`)

- churn_distribution.png
- balance_vs_churn.png
- digital_engagement_vs_churn.png
- complaints_vs_churn.png
- satisfaction_vs_churn.png
- confusion_matrix.png
- roc_curve.png
- feature_importance.png
- heatmap.png
- age_distribution.png
- age_vs_churn.png
- balance_distribution.png
- credit_utilization_vs_churn
- inactive_days_vs_churn
- last_login_vs_churn
- retention_offer_vs_churn

---

## 📁 Outputs (`outputs/`)

- Predictions.csv

---

## 📁 Models (`models/`)

- best_model.pkl
- scaler.pkl

---

# 📈 12. Business Insights

### Customer Behavior Insights

- Customers with low engagement are more likely to churn
- Complaint history strongly correlates with churn
- Financial inactivity is a major churn indicator
- Declining balances increase churn probability

### Strategic Value

The system helps banks:

- Identify at-risk customers early
- Improve retention campaigns
- Reduce customer attrition
- Improve long-term revenue stability

---

# 🚀 13. Future Improvements

### Planned Enhancements

- Deep Learning models
- Real-time churn prediction
- Explainable AI integration
- Time-series behavioral analysis
- Cloud deployment pipeline

---

# ⚙️ 14. Tech Stack

## Programming Language
- Python 3.10+

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

# 🏗️ 15. Project Structure

```bash
project-root/
│
├── data/
│   ├── ChurnZero_dataset_v1.csv
│   ├── ChurnZero_test_v1.csv
│
├── graphs/
│
├── models/
│
├── outputs/
│
├── notebooks/
│   ├── eda.py
│   ├── model_training.py
│   ├── generate_predictions.py
│
├── README.md
```

---

# ▶️ 16. How to Run the Project

## STEP 1 — Open Project Folder

```bash
cd "C:\Users\rootuser\OneDrive\Desktop\churn_project"
```

---

## STEP 2 — Run EDA & Preprocessing

```bash
python notebooks\eda.py
```

This will:
- Perform EDA
- Generate graphs
- Clean & preprocess data
- Save processed dataset

---

## STEP 3 — Run Model Training

```bash
python notebooks\model_training.py
```

This will:
- Train multiple ML models
- Compare performance
- Select best model
- Generate reports & graphs
- Save trained model

---

## STEP 4 — Generate Predictions

```bash
python notebooks\generate_predictions.py
```

This will:
- Load saved best model
- Predict churn for test customers
- Generate probability scores
- Save prediction reports

---

## STEP 5 — Check Final Outputs

### View Predictions

```bash
type outputs\Predictions.csv
```

---

# 🔥 17. Key Highlights

✅ End-to-end ML pipeline  
✅ Advanced EDA & visualization  
✅ Feature engineering  
✅ Multi-model comparison system  
✅ Recall & F2-score optimized  
✅ Business-oriented evaluation  
✅ Automated prediction generation  
✅ Production-style workflow  
✅ Reproducible ML architecture  
✅ Banking-focused churn analytics  

---

# 📌 18. Conclusion

This project demonstrates a complete production-oriented machine learning pipeline for banking customer churn prediction.

The system combines:

- Strong predictive performance
- Business-driven evaluation
- Automated workflows
- Advanced preprocessing
- Real-world churn analytics

By prioritizing Recall and minimizing false negatives, the project effectively supports proactive customer retention strategies and helps reduce long-term revenue leakage in banking environments.
