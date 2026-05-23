# =====================================
# model_training.py
# =====================================

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    average_precision_score
)

# =====================================
# SETTINGS
# =====================================

warnings.filterwarnings("ignore")

Path("graphs").mkdir(exist_ok=True)

Path("models").mkdir(exist_ok=True)

Path("outputs").mkdir(exist_ok=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPHS_DIR = PROJECT_ROOT / "graphs"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# =====================================
# LOAD DATA
# =====================================

print("Loading dataset...\n")

df = pd.read_csv(
    "data/ChurnZero_dataset_v1.csv"
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================
# TARGET + FEATURES
# =====================================

y = df['churn']

X = df.drop(
    ['customer_id', 'churn'],
    axis=1
)

# =====================================
# FEATURE ENGINEERING
# =====================================

X['total_loan_products'] = (
    X['personal_loan_flag'] +
    X['home_loan_flag'] +
    X['auto_loan_flag']
)

X['total_products_owned'] = (
    X['savings_account_flag'] +
    X['current_account_flag'] +
    X['credit_card_flag'] +
    X['investment_product_flag'] +
    X['insurance_product_flag']
)

X['engagement_gap'] = (
    X['mobile_app_login_count'] -
    X['last_login_days']
)

X['complaint_severity'] = (
    X['total_complaints'] *
    X['unresolved_complaint_count']
)

# =====================================
# ENCODING
# =====================================

X = pd.get_dummies(
    X,
    drop_first=True
)

print("\nEncoded Shape:")
print(X.shape)

# =====================================
# HANDLE MISSING VALUES
# =====================================

X = X.fillna(
    X.median(
        numeric_only=True
    )
)

print("\nRemaining Missing Values:")
print(X.isnull().sum().sum())

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTrain Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =====================================
# MODELS
# =====================================

models = {

    "Random Forest": RandomForestClassifier(

        n_estimators=200,

        class_weight='balanced',

        random_state=42,

        n_jobs=-1
    ),

    "Hist Gradient Boosting":
    HistGradientBoostingClassifier(
        random_state=42
    ),

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    )
}

# =====================================
# TRAIN MODELS
# =====================================

results = []

best_model = None

best_model_name = ""

best_f2 = 0

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"\n========== {name} ==========\n")

    model.fit(
        X_train_scaled,
        y_train
    )

    pred = model.predict(
        X_test_scaled
    )

    prob = model.predict_proba(
        X_test_scaled
    )[:,1]

    accuracy = accuracy_score(
        y_test,
        pred
    )

    precision = precision_score(
        y_test,
        pred
    )

    recall = recall_score(
        y_test,
        pred
    )

    f1 = f1_score(
        y_test,
        pred
    )

    f2 = fbeta_score(
        y_test,
        pred,
        beta=2
    )

    roc = roc_auc_score(
        y_test,
        prob
    )

    pr_auc = average_precision_score(
        y_test,
        prob
    )

    results.append([

        name,

        accuracy,

        precision,

        recall,

        f1,

        f2,

        roc,

        pr_auc
    ])

    print(
        "Accuracy:",
        round(accuracy,4)
    )

    print(
        "Precision:",
        round(precision,4)
    )

    print(
        "Recall:",
        round(recall,4)
    )

    print(
        "F1:",
        round(f1,4)
    )

    print(
        "F2:",
        round(f2,4)
    )

    print(
        "ROC AUC:",
        round(roc,4)
    )

    print(
        "PR AUC:",
        round(pr_auc,4)
    )

    print("\nClassification Report:\n")

    print(

        classification_report(
            y_test,
            pred
        )
    )

    if f2 > best_f2:

        best_f2 = f2

        best_model = model

        best_model_name = name

# =====================================
# MODEL COMPARISON
# =====================================

comparison_df = pd.DataFrame(

    results,

    columns=[

        'Model',

        'Accuracy',

        'Precision',

        'Recall',

        'F1',

        'F2',

        'ROC_AUC',

        'PR_AUC'
    ]
)

comparison_df = comparison_df.sort_values(

    'F2',

    ascending=False
)

print("\n========== MODEL COMPARISON ==========\n")

print(comparison_df)

comparison_df.to_csv(

    "outputs/model_comparison.csv",

    index=False
)

# =====================================
# BEST MODEL
# =====================================

print(
f"\nBest Model: {best_model_name}"
)

pred = best_model.predict(
    X_test_scaled
)

prob = best_model.predict_proba(
    X_test_scaled
)[:,1]

# =====================================
# CONFUSION MATRIX
# =====================================

matrix = confusion_matrix(
    y_test,
    pred
)

fig, ax = plt.subplots(figsize=(6,5))

display = ConfusionMatrixDisplay(

    confusion_matrix=matrix,

    display_labels=['Stayed','Churned']
)

display.plot(
    cmap='Blues',
    ax=ax
)

plt.title(
f"Confusion Matrix - {best_model_name}"
)

plt.savefig(
"graphs/confusion_matrix.png"
)

plt.show()

# =====================================
# ROC CURVE
# =====================================

fpr, tpr, _ = roc_curve(
    y_test,
    prob
)

plt.figure(figsize=(7,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc_score(y_test,prob):.4f}"
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel(
"False Positive Rate"
)

plt.ylabel(
"True Positive Rate"
)

plt.title(
"ROC Curve"
)

plt.legend()

plt.savefig(
"graphs/roc_curve.png"
)

plt.show()

# =====================================
# FEATURE IMPORTANCE
# =====================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        'Feature':X.columns,

        'Importance':best_model.feature_importances_
    })

    importance = importance.sort_values(

        'Importance',

        ascending=False
    )

    print("\nTop 15 Important Features:\n")

    print(
        importance.head(15)
    )

    importance.to_csv(

        "outputs/feature_importance.csv",

        index=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(

        data=importance.head(10),

        x='Importance',

        y='Feature'
    )

    plt.title(
    "Top 10 Important Features"
    )

    plt.savefig(
    "graphs/feature_importance.png"
    )

    plt.show()

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(

    best_model,

    "models/best_model.pkl"
)

print(
"\nBest model saved!"
)

# =====================================
# SAVE SCALER
# =====================================

joblib.dump(

    scaler,

    "models/scaler.pkl"
)

print(
"Scaler saved!"
)

# =====================================
# FINAL METRICS
# =====================================

metrics = {

    "Best_Model":best_model_name,

    "Accuracy":float(
        accuracy_score(y_test,pred)
    ),

    "Precision":float(
        precision_score(y_test,pred)
    ),

    "Recall":float(
        recall_score(y_test,pred)
    ),

    "F1":float(
        f1_score(y_test,pred)
    ),

    "F2":float(
        fbeta_score(y_test,pred,beta=2)
    ),

    "ROC_AUC":float(
        roc_auc_score(y_test,prob)
    )
}

with open(
    "outputs/final_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nFinal metrics saved!")

# =====================================
# MODEL HEALTH REPORT
# =====================================

health_report = f"""

MODEL HEALTH REPORT
==============================

Best Model:
{best_model_name}

--------------------------------

Accuracy  : {metrics['Accuracy']:.4f}
Precision : {metrics['Precision']:.4f}
Recall    : {metrics['Recall']:.4f}
F1 Score  : {metrics['F1']:.4f}
F2 Score  : {metrics['F2']:.4f}
ROC-AUC   : {metrics['ROC_AUC']:.4f}

--------------------------------

Business Interpretation:

- Recall is prioritized because
  churn prediction is an
  imbalanced classification task.

- F2 score gives more importance
  to Recall.

- ROC-AUC shows strong
  class separation ability.

- The model can help identify
  high-risk customers early.

==============================
"""

with open(
    "outputs/model_health_report.txt",
    "w"
) as f:

    f.write(
        health_report
    )

print(
"\nHealth report saved!"
)

# =====================================
# LOAD TEST DATA
# =====================================

print("\nLoading test dataset...\n")

test = pd.read_csv(
    "data/ChurnZero_test_v1.csv"
)

test_ids = test['customer_id']

# =====================================
# FEATURE ENGINEERING ON TEST
# =====================================

test['total_loan_products'] = (
    test['personal_loan_flag'] +
    test['home_loan_flag'] +
    test['auto_loan_flag']
)

test['total_products_owned'] = (
    test['savings_account_flag'] +
    test['current_account_flag'] +
    test['credit_card_flag'] +
    test['investment_product_flag'] +
    test['insurance_product_flag']
)

test['engagement_gap'] = (
    test['mobile_app_login_count'] -
    test['last_login_days']
)

test['complaint_severity'] = (
    test['total_complaints'] *
    test['unresolved_complaint_count']
)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

try:

    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    importance_df = pd.DataFrame({

        "feature": X_test.columns,

        "importance": result.importances_mean

    })

    importance_df.sort_values(
        by="importance",
        ascending=False,
        inplace=True
    )

    importance_df.to_csv(
        OUTPUTS_DIR / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df.head(10),
        x="importance",
        y="feature"
    )

    plt.title("Top 10 Important Features")

    plt.savefig(
        GRAPHS_DIR / "feature_importance.png",
        dpi=300
    )
    plt.show()
    plt.close()

except Exception as e:

    print(f"Feature importance failed: {e}")
    
# =====================================
# TEST PREPROCESSING
# =====================================

test_X = test.drop(
    ['customer_id'],
    axis=1
)

test_X = pd.get_dummies(
    test_X,
    drop_first=True
)

test_X = test_X.reindex(
    columns=X.columns,
    fill_value=0
)

test_X = test_X.fillna(
    test_X.median(
        numeric_only=True
    )
)

# =====================================
# SCALE TEST DATA
# =====================================

test_scaled = scaler.transform(
    test_X
)

# =====================================
# PREDICT TEST DATA
# =====================================

submission_prob = best_model.predict_proba(
    test_scaled
)[:,1]

submission_pred = (
    submission_prob >= 0.5
).astype(int)

# =====================================
# SAVE PREDICTIONS
# =====================================

submission = pd.DataFrame({

    'customer_id':test_ids,

    'churn_prediction':submission_pred,

    'churn_probability':submission_prob
})

submission.to_csv(

    "outputs/predictions.csv",

    index=False
)

print(
"\nPredictions saved!"
)

print(
submission.head()
)

# =====================================
# DETAILED BUSINESS REPORT
# =====================================

submission['risk_level'] = submission[
    'churn_probability'
].apply(

    lambda x:

    "High Risk" if x >= 0.75 else

    "Medium Risk" if x >= 0.45 else

    "Low Risk"
)

submission['business_action'] = submission[
    'risk_level'
].map({

    "High Risk":
    "Immediate retention campaign required",

    "Medium Risk":
    "Monitor customer engagement closely",

    "Low Risk":
    "Regular customer relationship maintenance"
})

submission.to_csv(

    "outputs/detailed_predictions_report.csv",

    index=False
)

print(
"\nDetailed prediction report saved!"
)

# =====================================
# COMPLETED
# =====================================

print("\n========== ALL TASKS COMPLETED ==========\n")

print("Saved Files:")

print("- models/best_model.pkl")

print("- models/scaler.pkl")

print("- outputs/model_comparison.csv")

print("- outputs/final_metrics.json")

print("- outputs/predictions.csv")

print("- outputs/detailed_predictions_report.csv")

print("- outputs/model_health_report.txt")

print("- outputs/feature_importance.csv")

print("- graphs/confusion_matrix.png")

print("- graphs/roc_curve.png")

print("- graphs/feature_importance.png")
