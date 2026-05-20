# 🏦 Banking Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge\&logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/ChurnZero26-Hackathon-purple?style=for-the-badge)

### 🚀 End-to-End Machine Learning Pipeline for Predicting Banking Customer Churn

</div>

---

# 📌 Project Overview

This project was developed for the **ChurnZero 26 Data Science Hackathon**.

The objective was to build a high-performance machine learning system capable of predicting whether a bank customer is likely to leave the bank (customer churn).

Unlike traditional beginner-level ML projects that focus only on accuracy, this system was specifically designed around:

* 📈 High Recall
* 🎯 F2-score Optimization
* 💼 Business Impact
* 🔍 Threshold Tuning
* ⚙️ Production-Ready Pipeline

The final system helps banks identify high-risk customers early so retention strategies can be applied before customer loss occurs.

---

# 🎯 Business Problem

Customer churn is one of the biggest challenges faced by banks.

When customers leave:

* Revenue decreases
* Cross-selling opportunities reduce
* Customer acquisition cost increases
* Lifetime customer value drops

Banks lose approximately **15–25% customers annually** due to churn.

The goal of this project is to:

> Predict churn probability using customer banking behavior and demographic information.

---

# 🧠 Key Features of the Project

✅ Complete EDA Pipeline
✅ Data Cleaning & Preprocessing
✅ Multiple ML Models Trained
✅ Hyperparameter Optimization
✅ Threshold Optimization using F2-score
✅ Recall-Focused Evaluation
✅ Model Comparison System
✅ Feature Importance Analysis
✅ Prediction CSV Generation
✅ Saved Production-Ready Model
✅ Reproducible End-to-End Workflow

---

# 📂 Dataset Information

### Dataset Used

`Churn_Modelling.csv`

### Dataset Size

| Attribute       | Value  |
| --------------- | ------ |
| Total Customers | 10,000 |
| Features        | 14     |
| Target Variable | Exited |

---

# 📊 Features Used

| Feature         | Description              |
| --------------- | ------------------------ |
| CreditScore     | Customer credit score    |
| Geography       | Country of customer      |
| Gender          | Male/Female              |
| Age             | Customer age             |
| Tenure          | Years with bank          |
| Balance         | Bank balance             |
| NumOfProducts   | Number of products used  |
| HasCrCard       | Credit card ownership    |
| IsActiveMember  | Active membership status |
| EstimatedSalary | Estimated yearly salary  |

### 🎯 Target Variable

| Value | Meaning          |
| ----- | ---------------- |
| 0     | Customer Stayed  |
| 1     | Customer Churned |

---

# 🏗️ Project Architecture

```text
churn_project/
│
├── data/
│   └── Churn_Modelling.csv
│
├── notebooks/
│   ├── eda.py
│   └── model_training.py
│
├── models/
│   └── best_model.pkl
│
├── graphs/
│   ├── churn_distribution.png
│   ├── age_vs_churn.png
│   ├── balance_vs_churn.png
│   ├── geography_vs_churn.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── outputs/
│   ├── processed_data.csv
│   ├── predictions.csv
│   ├── model_comparison.csv
│   ├── final_metrics.json
│   ├── classification_report.txt
│   └── model_health_report.md
│
└── README.md
```

---

# 🔬 Exploratory Data Analysis (EDA)

The EDA pipeline performs:

* Dataset inspection
* Statistical analysis
* Missing value checking
* Duplicate detection
* Correlation analysis
* Customer behavior analysis
* Churn visualization

### 📈 Generated Visualizations

* Customer Churn Distribution
* Age vs Churn
* Balance vs Churn
* Geography vs Churn
* Active Member vs Churn
* Credit Score Distribution
* Correlation Heatmap

---

# ⚙️ Data Preprocessing Pipeline

The preprocessing workflow includes:

## ✅ Data Cleaning

Removed unnecessary columns:

* RowNumber
* CustomerId
* Surname

## ✅ Feature Encoding

* Gender → Label Encoding
* Geography → One Hot Encoding

## ✅ Feature Scaling

Implemented using:

```python
StandardScaler()
```

---

# 🤖 Machine Learning Models Used

The project trained and compared multiple machine learning algorithms.

| Model                  | Status |
| ---------------------- | ------ |
| Logistic Regression    | ✅      |
| Decision Tree          | ✅      |
| K-Nearest Neighbors    | ✅      |
| Gaussian Naive Bayes   | ✅      |
| Support Vector Machine | ✅      |
| MLP Neural Network     | ✅      |
| Random Forest          | ✅      |
| Bagging Trees          | ✅      |
| Extra Trees            | ✅      |
| Gradient Boosting      | ✅      |
| AdaBoost               | ✅      |
| Hist Gradient Boosting | ✅      |

---

# 🧪 Hyperparameter Tuning

To improve model performance, the project used:

* `GridSearchCV`
* `RandomizedSearchCV`

Evaluation metrics during tuning:

* ROC-AUC
* Recall
* F1-score
* F2-score

---

# 🎯 Threshold Optimization

Instead of using the default probability threshold of **0.5**, this project optimized thresholds using:

```python
TunedThresholdClassifierCV
```

### Why?

In churn prediction:

> Missing a real churn customer is more expensive than investigating false positives.

Therefore, the project optimized for:

# ✅ High Recall

This significantly improved churn detection capability.

---

# 🏆 Final Selected Model

## ✅ Gradient Boosting Classifier

### Selection Rule

The final model was selected using:

1. Highest F2-score
2. Highest ROC-AUC
3. Highest Recall

---

# 📊 Final Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.7500 |
| Precision | 0.4397 |
| Recall    | 0.8329 |
| F1-score  | 0.5756 |
| F2-score  | 0.7065 |
| ROC-AUC   | 0.8726 |

---

# 📌 Model Interpretation

### 🔍 ROC-AUC = 0.8726

The model can successfully distinguish churners from non-churners approximately **87% of the time**.

### 🔍 Recall = 83.29%

The system successfully identifies more than **83% of customers likely to churn**.

This makes the model highly effective for:

* Early customer retention
* Risk monitoring
* Banking CRM systems
* Retention campaigns

---

# 📈 Feature Importance Analysis

Permutation Feature Importance was used to determine the most influential churn factors.

### Most Important Features

* Age
* Balance
* Geography
* Number of Products
* Active Membership Status

---

# 📤 Generated Outputs

## 📦 Saved Model

```text
models/best_model.pkl
```

---

## 📊 Graphs

* Confusion Matrix
* ROC Curve
* Feature Importance
* Model Comparison

---

## 📁 Prediction Files

* `predictions.csv`
* `all_agent_predictions.csv`

---

## 📄 Reports

* `final_metrics.json`
* `classification_report.txt`
* `model_health_report.md`
* `api_usage_notes.md`

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

# 🚀 How the Project Was Developed

The development workflow followed a structured machine learning lifecycle:

```text
Dataset Loading
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Feature Encoding
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Threshold Optimization
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
        ↓
Prediction Generation
        ↓
Model Serialization
```

---

# ▶️ How to Run the Project

## STEP 1: Go to the project folder

```bash
cd "C:\Users\root\OneDrive\Desktop\churn_project"
```

Wait until the terminal shows the project path again.

---

## STEP 2: Run preprocessing and EDA

```bash
python notebooks\eda.py
```

Wait until the script prints:

```text
Saved processed dataset
```

---

## STEP 3: Run model training

```bash
python notebooks\model_training.py
```

Wait until the script saves:

```text
models/best_model.pkl
```

---

## STEP 4: Check final best model metrics

```bash
type outputs\final_metrics.json
```

---

## STEP 5: Check model comparison results

```bash
type outputs\model_comparison.csv
```

---

## STEP 6: Test model prediction

```bash
python -c "import joblib, pandas as pd; model=joblib.load('models/best_model.pkl'); df=pd.read_csv('data/Churn_Modelling.csv').drop(columns=['RowNumber','CustomerId','Surname','Exited']).head(1); print('Prediction:', model.predict(df)); print('Churn probability:', model.predict_proba(df)[:,1])"
```

---

## STEP 7: Confirm generated files exist

```bash
dir models
dir graphs
dir outputs
```

---

# 🧪 Example Prediction Output

```text
Prediction: [1]
Churn probability: [0.31330392]
```

### Interpretation

* `1` → Customer likely to churn
* `31.33%` → Estimated churn probability

The optimized threshold allows earlier churn detection compared to traditional systems.

---

# 💡 Business Impact

The system can help banks:

✅ Identify high-risk customers
✅ Improve customer retention
✅ Reduce revenue loss
✅ Trigger proactive retention campaigns
✅ Support relationship managers with risk insights

---

# 🔮 Future Improvements

Potential future enhancements:

* SHAP Explainability
* Streamlit Dashboard Deployment
* Flask/FastAPI API Deployment
* Real-time Prediction System
* Ensemble Stacking Models
* Deep Learning Architectures
* Customer Risk Segmentation

---

# 📌 Conclusion

This project demonstrates a complete end-to-end machine learning solution for banking customer churn prediction.

The final Gradient Boosting model achieved:

* ⭐ Strong ROC-AUC
* ⭐ High Recall
* ⭐ Business-focused Optimization
* ⭐ Production-Ready Pipeline

The project successfully balances technical machine learning performance with real-world banking business requirements.

---

<div align="center">

# ⭐ If you found this project useful, consider giving it a star ⭐

</div>
