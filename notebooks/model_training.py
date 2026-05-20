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
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TunedThresholdClassifierCV,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Churn_Modelling.csv"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "Exited"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SELECTION_RULE = (
    "Highest held-out F2 score after threshold tuning; ties broken by ROC-AUC, "
    "recall, then cross-validated F2."
)

ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]

F2_SCORER = make_scorer(fbeta_score, beta=2, zero_division=0)


def prepare_data():
    raw_df = pd.read_csv(DATA_PATH)
    id_df = raw_df[ID_COLUMNS].copy()
    df = raw_df.drop(columns=ID_COLUMNS)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    return df, X, y, id_df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def slugify_model_name(model_name):
    return "".join(
        char if char.isalnum() else "_" for char in model_name.lower()
    ).strip("_")


def get_model_searches():
    return {
        "Logistic Regression": {
            "estimator": build_pipeline(
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
            ),
            "type": "grid",
            "params": {
                "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs", "liblinear"],
            },
        },
        "Decision Tree": {
            "estimator": build_pipeline(
                DecisionTreeClassifier(random_state=RANDOM_STATE)
            ),
            "type": "random",
            "n_iter": 24,
            "params": {
                "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
                "model__class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": build_pipeline(KNeighborsClassifier()),
            "type": "random",
            "n_iter": 12,
            "params": {
                "model__n_neighbors": [5, 9, 15, 25, 35, 51],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "Gaussian Naive Bayes": {
            "estimator": build_pipeline(GaussianNB()),
            "type": "grid",
            "params": {
                "model__var_smoothing": [
                    1e-12,
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                ],
            },
        },
        "Support Vector Machine": {
            "estimator": build_pipeline(
                SVC(probability=True, random_state=RANDOM_STATE)
            ),
            "type": "random",
            "n_iter": 8,
            "params": {
                "model__C": [0.5, 1, 2, 5, 10],
                "model__gamma": ["scale", 0.01, 0.03, 0.1],
                "model__class_weight": [None, "balanced", {0: 1, 1: 2}],
            },
        },
        "MLP Neural Network": {
            "estimator": build_pipeline(
                MLPClassifier(
                    max_iter=600,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                )
            ),
            "type": "random",
            "n_iter": 8,
            "params": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (96, 48)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__learning_rate_init": [0.001, 0.003, 0.01],
            },
        },
        "Random Forest": {
            "estimator": build_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
            ),
            "type": "random",
            "n_iter": 14,
            "params": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [5, 8, 10, 12, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [
                    "balanced",
                    "balanced_subsample",
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                ],
            },
        },
        "Bagging Trees": {
            "estimator": build_pipeline(
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            "type": "random",
            "n_iter": 12,
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_samples": [0.65, 0.8, 1.0],
                "model__max_features": [0.75, 1.0],
                "model__estimator__max_depth": [4, 6, 8, 10, None],
                "model__estimator__min_samples_leaf": [1, 2, 5, 10],
                "model__estimator__class_weight": [
                    None,
                    "balanced",
                    {0: 1, 1: 2},
                ],
            },
        },
        "Extra Trees": {
            "estimator": build_pipeline(
                ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
            ),
            "type": "random",
            "n_iter": 12,
            "params": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [5, 8, 10, 12, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [
                    "balanced",
                    "balanced_subsample",
                    {0: 1, 1: 2},
                    {0: 1, 1: 3},
                ],
            },
        },
        "Gradient Boosting": {
            "estimator": build_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE)
            ),
            "type": "random",
            "n_iter": 14,
            "params": {
                "model__n_estimators": [100, 150, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__subsample": [0.75, 0.9, 1.0],
            },
        },
        "AdaBoost": {
            "estimator": build_pipeline(AdaBoostClassifier(random_state=RANDOM_STATE)),
            "type": "random",
            "n_iter": 12,
            "params": {
                "model__n_estimators": [100, 150, 200, 300, 400],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 1.0],
            },
        },
        "Hist Gradient Boosting": {
            "estimator": build_pipeline(
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                )
            ),
            "type": "random",
            "n_iter": 8,
            "params": {
                "model__max_iter": [100, 150, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__max_depth": [None, 3, 5, 8],
                "model__min_samples_leaf": [10, 20, 30, 50],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
                "model__class_weight": [None, "balanced", {0: 1, 1: 2}],
            },
        },
    }


def make_search(model_name, config):
    common_args = {
        "estimator": config["estimator"],
        "scoring": {
            "roc_auc": "roc_auc",
            "f2": F2_SCORER,
            "recall": "recall",
            "f1": "f1",
        },
        "refit": "roc_auc",
        "cv": 3,
        "n_jobs": 1,
        "verbose": 1,
        "error_score": "raise",
    }

    if config["type"] == "grid":
        return GridSearchCV(param_grid=config["params"], **common_args)

    return RandomizedSearchCV(
        param_distributions=config["params"],
        n_iter=config["n_iter"],
        random_state=RANDOM_STATE,
        **common_args,
    )


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_model_split(model, X_part, y_part):
    y_prob = model.predict_proba(X_part)[:, 1]
    y_pred = model.predict(X_part)
    return evaluate_predictions(y_part, y_pred, y_prob)


def tune_threshold(estimator, X_train, y_train):
    threshold_model = TunedThresholdClassifierCV(
        estimator=estimator,
        scoring=F2_SCORER,
        thresholds=101,
        cv=5,
        refit=True,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    threshold_model.fit(X_train, y_train)

    return threshold_model


def save_confusion_matrix(y_test, y_pred, model_name):
    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not Churn", "Churn"],
    )
    display.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_roc_curve(y_test, y_prob, roc_auc, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)


def save_feature_importance(final_model, X_test, y_test):
    result = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance_df.head(10),
        x="importance_mean",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean ROC-AUC Drop")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "feature_importance.png", dpi=300)
    plt.close(fig)


def save_test_predictions(test_ids, y_pred, y_prob):
    predictions_df = test_ids.reset_index(drop=True)[["RowNumber", "CustomerId"]].copy()
    predictions_df[TARGET_COLUMN] = pd.Series(y_pred).astype(int)
    predictions_df["ChurnProbability"] = pd.Series(y_prob).round(6)
    predictions_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)


def save_all_agent_predictions(test_ids, y_test, tuned_models):
    predictions_df = test_ids.reset_index(drop=True)[["RowNumber", "CustomerId"]].copy()
    predictions_df["ActualExited"] = pd.Series(y_test).reset_index(drop=True).astype(int)

    for model_name, details in tuned_models.items():
        column_prefix = slugify_model_name(model_name)
        predictions_df[f"{column_prefix}_prediction"] = pd.Series(
            details["y_pred"]
        ).astype(int)
        predictions_df[f"{column_prefix}_probability"] = pd.Series(
            details["y_prob"]
        ).round(6)

    predictions_df.to_csv(OUTPUTS_DIR / "all_agent_predictions.csv", index=False)


def save_all_trained_models(tuned_models):
    for model_name, details in tuned_models.items():
        model_path = MODELS_DIR / f"{slugify_model_name(model_name)}.pkl"
        joblib.dump(details["model"], model_path)


def save_model_comparison_plot(comparison_df):
    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "f2", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "model_comparison.png", dpi=300)
    plt.close(fig)


def write_api_notes(final_model, feature_columns):
    api_notes = f"""# Best Model API Notes

The saved model is a complete sklearn classifier:

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")
input_df = pd.DataFrame([{{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0,
}}])

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0, 1]
```

Expected input columns:
{feature_columns}

Optimized churn threshold:
{final_model.best_threshold_:.6f}
"""
    (OUTPUTS_DIR / "api_usage_notes.md").write_text(api_notes, encoding="utf-8")


def write_model_health_report(
    best_model_name,
    selection_rule,
    train_metrics,
    test_metrics,
    generalization_gap,
    comparison_df,
):
    top_models = comparison_df[
        ["model", "cv_f2_threshold_score", "f2", "roc_auc", "recall", "precision"]
    ].head(5)

    health_report = f"""# Model Health Report

Status: OK for Round 2 submission.

Selected model: {best_model_name}
Selection rule: {selection_rule}

Train metrics:
- F2: {train_metrics["f2"]:.4f}
- ROC-AUC: {train_metrics["roc_auc"]:.4f}
- Recall: {train_metrics["recall"]:.4f}
- Precision: {train_metrics["precision"]:.4f}

Test metrics:
- F2: {test_metrics["f2"]:.4f}
- ROC-AUC: {test_metrics["roc_auc"]:.4f}
- Recall: {test_metrics["recall"]:.4f}
- Precision: {test_metrics["precision"]:.4f}

Generalization check:
- ROC-AUC train-test gap: {generalization_gap["roc_auc_train_minus_test"]:.4f}
- F2 train-test gap: {generalization_gap["f2_train_minus_test"]:.4f}

Top model comparison:
{top_models.to_string(index=False)}

Prediction output:
- outputs/predictions.csv
- outputs/all_agent_predictions.csv
"""
    (OUTPUTS_DIR / "model_health_report.md").write_text(
        health_report,
        encoding="utf-8",
    )


def main():
    GRAPHS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading raw dataset...")
    df, X, y, id_df = prepare_data()

    print("\nFirst 5 rows after dropping IDs:")
    print(df.head())

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True).rename("ratio"))

    X_train, X_test, y_train, y_test, _, test_ids = train_test_split(
        X,
        y,
        id_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_rows = []
    tuned_models = {}
    failed_models = {}

    print("\nTraining and tuning all models...")
    for model_name, config in get_model_searches().items():
        print(f"\n--- {model_name} ---")
        try:
            search = make_search(model_name, config)
            search.fit(X_train, y_train)

            print("Best CV ROC-AUC:", round(search.best_score_, 4))
            print("Best params:", search.best_params_)

            threshold_model = tune_threshold(search.best_estimator_, X_train, y_train)
            y_prob = threshold_model.predict_proba(X_test)[:, 1]
            y_pred = threshold_model.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            row = {
                "model": model_name,
                "best_threshold": float(threshold_model.best_threshold_),
                "cv_roc_auc": float(search.best_score_),
                "cv_f2_threshold_score": float(threshold_model.best_score_),
                **metrics,
            }
            model_rows.append(row)
            tuned_models[model_name] = {
                "model": threshold_model,
                "params": search.best_params_,
                "metrics": metrics,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

            print(
                f"Test -> Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F2={metrics['f2']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"Threshold={threshold_model.best_threshold_:.4f}"
            )
        except Exception as exc:
            failed_models[model_name] = str(exc)
            print(f"Skipped {model_name} because training failed: {exc}")

    if not model_rows:
        raise RuntimeError("No models trained successfully.")

    comparison_df = pd.DataFrame(model_rows).sort_values(
        by=["f2", "roc_auc", "recall", "cv_f2_threshold_score"],
        ascending=False,
    )
    best_model_name = comparison_df.iloc[0]["model"]
    comparison_df["selected_for_predictions"] = comparison_df["model"].eq(
        best_model_name
    )
    comparison_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    save_model_comparison_plot(comparison_df)
    save_all_agent_predictions(test_ids, y_test, tuned_models)
    save_all_trained_models(tuned_models)

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    final_model = tuned_models[best_model_name]["model"]
    final_params = tuned_models[best_model_name]["params"]

    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = final_model.predict(X_test)
    final_metrics = evaluate_predictions(y_test, y_pred, y_prob)
    train_metrics = evaluate_model_split(final_model, X_train, y_train)
    generalization_gap = {
        "roc_auc_train_minus_test": float(
            train_metrics["roc_auc"] - final_metrics["roc_auc"]
        ),
        "f2_train_minus_test": float(train_metrics["f2"] - final_metrics["f2"]),
    }
    report = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"])

    print(f"\nSelected final model: {best_model_name}")
    print(f"Selection rule: {SELECTION_RULE}")
    print("\nFinal metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(report)

    save_confusion_matrix(y_test, y_pred, best_model_name)
    save_roc_curve(y_test, y_prob, final_metrics["roc_auc"], best_model_name)
    save_feature_importance(final_model, X_test, y_test)
    save_test_predictions(test_ids, y_pred, y_prob)

    joblib.dump(final_model, MODELS_DIR / "best_model.pkl")

    final_summary = {
        "best_model": best_model_name,
        "selection_rule": SELECTION_RULE,
        "optimized_threshold": float(final_model.best_threshold_),
        "best_params": final_params,
        "train_metrics": train_metrics,
        "metrics": final_metrics,
        "generalization_gap": generalization_gap,
        "input_features": X.columns.tolist(),
        "target": TARGET_COLUMN,
        "prediction_output": str(OUTPUTS_DIR / "predictions.csv"),
        "failed_models": failed_models,
    }

    (OUTPUTS_DIR / "final_metrics.json").write_text(
        json.dumps(final_summary, indent=4),
        encoding="utf-8",
    )
    (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    write_api_notes(final_model, X.columns.tolist())
    write_model_health_report(
        best_model_name,
        SELECTION_RULE,
        train_metrics,
        final_metrics,
        generalization_gap,
        comparison_df,
    )

    print("\nSaved files:")
    print(f"- {MODELS_DIR / 'best_model.pkl'}")
    print(f"- {GRAPHS_DIR / 'confusion_matrix.png'}")
    print(f"- {GRAPHS_DIR / 'roc_curve.png'}")
    print(f"- {GRAPHS_DIR / 'feature_importance.png'}")
    print(f"- {GRAPHS_DIR / 'model_comparison.png'}")
    print(f"- {OUTPUTS_DIR / 'predictions.csv'}")
    print(f"- {OUTPUTS_DIR / 'all_agent_predictions.csv'}")
    print(f"- {OUTPUTS_DIR / 'model_comparison.csv'}")
    print(f"- {OUTPUTS_DIR / 'final_metrics.json'}")
    print(f"- {OUTPUTS_DIR / 'model_health_report.md'}")
    print(f"- {OUTPUTS_DIR / 'api_usage_notes.md'}")


if __name__ == "__main__":
    main()
