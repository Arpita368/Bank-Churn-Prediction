# model_training.py

from pathlib import Path
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_selector

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "ChurnZero_Dataset_v1.csv"

TEST_PATH = PROJECT_ROOT / "data" / "ChurnZero_Test_v1.csv"

GRAPHS_DIR = PROJECT_ROOT / "graphs"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


GRAPHS_DIR.mkdir(exist_ok=True)

MODELS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)


print("\nTrain Shape:")
print(train_df.shape)

print("\nTest Shape:")
print(test_df.shape)


# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------

TARGET_COLUMN = "churn"

ID_COLUMN = "customer_id"


X = train_df.drop(columns=[TARGET_COLUMN])

y = train_df[TARGET_COLUMN]

test_ids = test_df[[ID_COLUMN]]

X_submission = test_df.copy()


# ---------------------------------------------------
# DROP ID
# ---------------------------------------------------

X.drop(columns=[ID_COLUMN], inplace=True)

X_submission.drop(columns=[ID_COLUMN], inplace=True)


# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

def feature_engineering(df):

    df = df.copy()

    df["total_loan_products"] = (
        df["personal_loan_flag"] +
        df["home_loan_flag"] +
        df["auto_loan_flag"]
    )

    df["total_products_owned"] = (
        df["savings_account_flag"] +
        df["current_account_flag"] +
        df["credit_card_flag"] +
        df["investment_product_flag"] +
        df["insurance_product_flag"]
    )

    df["engagement_gap"] = (
        df["mobile_app_login_count"] -
        df["last_login_days"]
    )

    df["credit_utilization_change"] = (
        df["credit_utilization_6m_avg"] -
        df["credit_utilization_3m_avg"]
    )

    df["complaint_severity"] = (
        df["total_complaints"] *
        df["unresolved_complaint_count"]
    )

    df["transaction_drop_indicator"] = (
        df["balance_decline_percentage"] *
        df["account_inactive_days"]
    )

    return df


X = feature_engineering(X)

X_submission = feature_engineering(X_submission)


# ---------------------------------------------------
# FEATURE TYPES
# ---------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(len(numeric_features))


# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        ),
    ]
)


# ---------------------------------------------------
# PIPELINE
# ---------------------------------------------------

def build_pipeline(model):

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

models = {

    "Logistic Regression": build_pipeline(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Decision Tree": build_pipeline(
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        )
    ),

    "KNN": build_pipeline(
        KNeighborsClassifier()
    ),

    "GaussianNB": build_pipeline(
        GaussianNB()
    ),

    "SVM": build_pipeline(
        SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    ),

    "MLP": build_pipeline(
        MLPClassifier(
            max_iter=600,
            early_stopping=True,
            random_state=42
        )
    ),

    "Random Forest": build_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    ),

    "Bagging": build_pipeline(
        BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=200,
            random_state=42,
            n_jobs=1
        )
    ),

    "Extra Trees": build_pipeline(
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    ),

    "Gradient Boosting": build_pipeline(
        GradientBoostingClassifier(
            random_state=42
        )
    ),

    "AdaBoost": build_pipeline(
        AdaBoostClassifier(
            random_state=42
        )
    ),

    "Hist Gradient Boosting": build_pipeline(
        HistGradientBoostingClassifier(
            random_state=42
        )
    ),
}


# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

results = []

best_model = None

best_model_name = ""

best_f2 = 0


print("\nTraining Models...\n")


for name, model in models.items():

    print(f"Training {name}...")

    try:

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        threshold = 0.35

        y_pred = (y_prob >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f2 = fbeta_score(
            y_test,
            y_pred,
            beta=2,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        results.append([
            name,
            accuracy,
            precision,
            recall,
            f1,
            f2,
            roc_auc,
            pr_auc
        ])

        print(
            f"Recall={recall:.4f} | "
            f"F2={f2:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )

        if f2 > best_f2:

            best_f2 = f2

            best_model = model

            best_model_name = name

    except Exception as e:

        print(f"{name} failed: {e}")


# ---------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------

comparison_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "ROC_AUC",
        "PR_AUC"
    ]
)

comparison_df.sort_values(
    by="F2",
    ascending=False,
    inplace=True
)

comparison_df.to_csv(
    OUTPUTS_DIR / "model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(comparison_df)


# ---------------------------------------------------
# FINAL MODEL
# ---------------------------------------------------

print(f"\nBest Model: {best_model_name}")

y_prob = best_model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= 0.35).astype(int)


# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

report = classification_report(
    y_test,
    y_pred
)

print("\nClassification Report:")
print(report)

with open(
    OUTPUTS_DIR / "classification_report.txt",
    "w"
) as f:

    f.write(report)


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(figsize=(6, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Retained", "Churned"]
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.savefig(
    GRAPHS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}"
)

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    GRAPHS_DIR / "roc_curve.png",
    dpi=300
)

plt.close()


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

    plt.close()

except Exception as e:

    print(f"Feature importance failed: {e}")


# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    best_model,
    MODELS_DIR / "best_model.pkl"
)


# ---------------------------------------------------
# GENERATE TEST PREDICTIONS
# ---------------------------------------------------

print("\nGenerating Predictions...")

submission_prob = best_model.predict_proba(
    X_submission
)[:, 1]

submission_pred = (
    submission_prob >= 0.35
).astype(int)


submission = pd.DataFrame({

    "customer_id": test_ids["customer_id"],

    "churn_prediction": submission_pred,

    "churn_probability": submission_prob.round(6)

})


submission.to_csv(
    OUTPUTS_DIR / "ChurnZero_Team_Predictions.csv",
    index=False
)


# ---------------------------------------------------
# FINAL METRICS
# ---------------------------------------------------

metrics = {

    "Best_Model": best_model_name,

    "Accuracy": float(
        accuracy_score(y_test, y_pred)
    ),

    "Precision": float(
        precision_score(y_test, y_pred)
    ),

    "Recall": float(
        recall_score(y_test, y_pred)
    ),

    "F1": float(
        f1_score(y_test, y_pred)
    ),

    "F2": float(
        fbeta_score(y_test, y_pred, beta=2)
    ),

    "ROC_AUC": float(
        roc_auc_score(y_test, y_prob)
    ),

    "PR_AUC": float(
        average_precision_score(y_test, y_prob)
    )
}


with open(
    OUTPUTS_DIR / "final_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nAll Outputs Saved Successfully.")

print("\nSaved Files:")

print("- graphs/confusion_matrix.png")
print("- graphs/roc_curve.png")
print("- graphs/feature_importance.png")
print("- outputs/model_comparison.csv")
print("- outputs/feature_importance.csv")
print("- outputs/final_metrics.json")
print("- outputs/classification_report.txt")
print("- outputs/ChurnZero_Team_Predictions.csv")
print("- models/best_model.pkl")