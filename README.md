# 🏦 Customer Churn Prediction System (End-to-End Machine Learning Project)

An advanced end-to-end machine learning system designed to predict **customer churn in a banking environment**. This project covers the complete ML lifecycle including **data analysis, feature engineering, model training, evaluation, threshold tuning, and prediction generation**.

The system is optimized not just for accuracy, but for **business impact**, focusing heavily on **recall and F2-score** to ensure maximum identification of potential churn customers.

---

# 📌 Problem Statement

Customer churn is one of the most critical problems in the banking sector. Losing customers directly impacts revenue, trust, and long-term growth.

This project aims to:

- Predict whether a customer will churn (1) or stay (0)
- Identify high-risk customers early
- Help banks take preventive retention actions
- Reduce revenue loss due to customer attrition

---

# 📊 Dataset Overview

The dataset contains structured banking customer information.

- **Dataset Name:** ChurnZero Dataset v1  
- **Total Records:** ~10,000 customers  
- **Features:** ~15–30 behavioral + financial attributes  
- **Target Variable:** `churn`

### Target Definition
- `0 → Retained Customer`
- `1 → Churned Customer`

---

# 🧠 Project Architecture


Raw Dataset
↓
EDA (eda.py)
↓
Data Cleaning + Visualization
↓
Feature Engineering
↓
Preprocessing Pipeline (Scaling + Encoding)
↓
Multiple ML Models Training
↓
Model Evaluation (Recall, F2, ROC-AUC)
↓
Best Model Selection
↓
Threshold Optimization (0.35)
↓
Final Predictions
↓
Saved Outputs + Reports + Model Artifact


---

# 🔍 1. Exploratory Data Analysis (EDA)

Performed in `eda.py`

### 📌 Steps:
- Dataset overview (shape, info, statistics)
- Missing value analysis
- Duplicate detection
- Target class distribution (churn vs non-churn)

### 📊 Visualizations Generated:
- Customer churn distribution
- Age vs churn (boxplot)
- Balance vs churn
- Digital engagement vs churn
- Complaints vs churn
- Satisfaction score vs churn
- Credit utilization vs churn
- Inactive days vs churn
- Age distribution
- Balance distribution
- Correlation heatmap

All graphs are saved in:


/graphs


---

# ⚙️ 2. Feature Engineering

Advanced behavioral features were created to improve model performance:

### 🧩 Engineered Features:

- `total_loan_products`
  → Total number of loan products used by customer

- `total_products_owned`
  → Total banking products owned

- `engagement_gap`
  → Difference between login activity and inactivity

- `credit_utilization_change`
  → Change in credit usage over time

- `complaint_severity`
  → Impact of unresolved complaints

- `transaction_drop_indicator`
  → Decline in transactions combined with inactivity

---

# 🧹 3. Data Preprocessing Pipeline

A production-grade pipeline using `sklearn`:

### Numeric Features:
- Missing values → Median Imputation
- Scaling → StandardScaler

### Categorical Features:
- Missing values → Most frequent value
- Encoding → OneHotEncoder

### Pipeline System:
- ColumnTransformer used to combine both transformations
- Fully integrated into ML pipeline

---

# 🤖 4. Machine Learning Models Used

Multiple models were trained and compared:

### Linear Models
- Logistic Regression

### Tree-Based Models
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- HistGradient Boosting

### Other Models
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine
- Multi-Layer Perceptron (Neural Network)
- Bagging Classifier

---

# 📏 5. Evaluation Strategy

Instead of focusing only on accuracy, this project is optimized for **business impact**.

### 🎯 Key Metrics:
- Recall (MOST IMPORTANT)
- F2 Score (highest priority metric)
- Precision
- F1 Score
- ROC-AUC
- PR-AUC

### 🎯 Why F2 Score?
F2 gives more weight to **recall**, which is critical because:

> Missing a churn customer is more costly than incorrectly predicting churn.

---

# 🎯 Threshold Optimization

Instead of default 0.5 threshold:


Optimal Threshold = 0.35


This helps:
- Catch more churn customers
- Improve recall significantly
- Reduce false negatives

---

# 🏆 6. Best Model Selection

- All models are evaluated on F2-score
- Best performing model is automatically selected
- Saved as:


models/best_model.pkl


---

# 📊 7. Outputs Generated

## 📁 Graphs (`/graphs`)
- churn_distribution.png
- age_vs_churn.png
- balance_vs_churn.png
- engagement plots
- confusion_matrix.png
- roc_curve.png
- feature_importance.png

---

## 📁 Outputs (`/outputs`)

- model_comparison.csv → All model metrics
- feature_importance.csv → Top features
- final_metrics.json → Final evaluation results
- classification_report.txt → Precision/Recall report
- ChurnZero_Team_Predictions.csv → Final predictions

---

## 🤖 Model Artifact


models/best_model.pkl


---

# 📈 8. Final Prediction System

The trained model generates predictions:

### Output Format:

| customer_id | churn_prediction | churn_probability |
|-------------|------------------|-------------------|
| 101         | 1                | 0.87              |
| 102         | 0                | 0.12              |

---

# 🧠 9. Business Impact

This system enables banks to:

- Identify customers likely to leave
- Reduce churn rate significantly
- Improve customer retention strategy
- Target customers with personalized offers
- Increase long-term revenue stability

---

# ⚙️ 10. Tech Stack

### Programming Language:
- Python 3.10+

### Libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

# 🏗️ 11. Project Structure

```bash
project-root/
│
├── data/
│ ├── ChurnZero_Dataset_v1.csv
│ ├── ChurnZero_Test_v1.csv
│
├── graphs/
├── models/
├── outputs/
├── notebooks/
│ ├── eda.py
│ ├── model_training.py
├── README.md
```

---

# 🚀 12. How to Run


```bash
STEP 1: First go to the project folder
cd "C:\Users\rootuser\OneDrive\Desktop\churn_project"

Wait until the terminal shows the project path again, then run STEP 2.

STEP 2: Run preprocessing and EDA first
python notebooks\eda.py

Wait until it finishes and prints: Saved processed dataset
Then run STEP 3.

STEP 3: Run model training
python notebooks\model_training.py

Wait until it finishes and saves: models/best_model.pkl
Then run STEP 4.

STEP 4: Check final best model metrics
type outputs\final_metrics.json

Then run STEP 5.

STEP 5: Check all model comparison results
type outputs\model_comparison.csv

Then run STEP 6.

STEP 6: Test if saved model loads and predicts correctly
python -c "import joblib,pandas as pd; model=joblib.load('models/best_model.pkl'); df=pd.read_csv('data/ChurnZero_dataset_v1.csv').drop(columns=['customer_id','churn']).head(1); df['total_loan_products']=df['personal_loan_flag']+df['home_loan_flag']+df['auto_loan_flag']; df['total_products_owned']=df['savings_account_flag']+df['current_account_flag']+df['credit_card_flag']+df['investment_product_flag']+df['insurance_product_flag']; df['engagement_gap']=df['mobile_app_login_count']-df['last_login_days']; df['credit_utilization_change']=df['credit_utilization_6m_avg']-df['credit_utilization_3m_avg']; df['complaint_severity']=df['total_complaints']*df['unresolved_complaint_count']; df['transaction_drop_indicator']=df['balance_decline_percentage']*df['account_inactive_days']; print('Prediction:',model.predict(df)); print('Probability:',model.predict_proba(df)[:,1])"

STEP 7: Confirm model, graph, and output files exist
dir models
dir graphs
dir outputs
```
# 🔥 13. Key Highlights
End-to-end ML pipeline (EDA → Training → Prediction)
Feature engineering for behavioral intelligence
Multi-model training system
F2-score optimized evaluation strategy
Automated best model selection
Production-ready ML workflow
Threshold tuning for real-world performance


# 📌 14. Conclusion

This project demonstrates a complete machine learning system for customer churn prediction with strong emphasis on:

Real-world business impact
Model interpretability
Performance optimization for recall
Scalable ML pipeline design
