# eda.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "ChurnZero_Dataset_v1.csv"

GRAPHS_DIR = PROJECT_ROOT / "graphs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


GRAPHS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nStatistics:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ---------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------

print("\nChurn Distribution:")
print(df["churn"].value_counts())

print("\nChurn Ratio:")
print(df["churn"].value_counts(normalize=True))


# ---------------------------------------------------
# DROP CUSTOMER ID
# ---------------------------------------------------

df.drop(columns=["customer_id"], inplace=True)

print("\nRemaining Columns:")
print(df.columns.tolist())


# ---------------------------------------------------
# CHURN DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    x="churn",
    data=df
)

plt.title("Customer Churn Distribution")

plt.xlabel("Churn (0 = Retained, 1 = Churned)")

plt.ylabel("Customer Count")

plt.savefig(
    GRAPHS_DIR / "churn_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# AGE VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="age",
    data=df
)

plt.title("Age vs Churn")

plt.savefig(
    GRAPHS_DIR / "age_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# BALANCE VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="avg_monthly_balance",
    data=df
)

plt.title("Average Monthly Balance vs Churn")

plt.savefig(
    GRAPHS_DIR / "balance_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# DIGITAL ENGAGEMENT VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="digital_engagement_index",
    data=df
)

plt.title("Digital Engagement Index vs Churn")

plt.savefig(
    GRAPHS_DIR / "digital_engagement_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# COMPLAINTS VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="total_complaints",
    data=df
)

plt.title("Complaints vs Churn")

plt.savefig(
    GRAPHS_DIR / "complaints_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# SATISFACTION SCORE VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="satisfaction_score",
    data=df
)

plt.title("Satisfaction Score vs Churn")

plt.savefig(
    GRAPHS_DIR / "satisfaction_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# CREDIT UTILIZATION VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="credit_utilization_ratio",
    data=df
)

plt.title("Credit Utilization Ratio vs Churn")

plt.savefig(
    GRAPHS_DIR / "credit_utilization_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# INACTIVE DAYS VS CHURN
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="churn",
    y="account_inactive_days",
    data=df
)

plt.title("Inactive Days vs Churn")

plt.savefig(
    GRAPHS_DIR / "inactive_days_vs_churn.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# AGE DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.histplot(
    df["age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")

plt.savefig(
    GRAPHS_DIR / "age_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# BALANCE DISTRIBUTION
# ---------------------------------------------------

plt.figure(figsize=(7, 5))

sns.histplot(
    df["avg_monthly_balance"],
    bins=30,
    kde=True
)

plt.title("Average Monthly Balance Distribution")

plt.savefig(
    GRAPHS_DIR / "balance_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

heatmap_df = df.copy()

label_encoders = {}

for column in heatmap_df.select_dtypes(include=["object"]).columns:

    encoder = LabelEncoder()

    heatmap_df[column] = encoder.fit_transform(
        heatmap_df[column].astype(str)
    )

    label_encoders[column] = encoder


plt.figure(figsize=(22, 18))

sns.heatmap(
    heatmap_df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    GRAPHS_DIR / "heatmap.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------
# SAVE PROCESSED DATA
# ---------------------------------------------------

heatmap_df.to_csv(
    OUTPUTS_DIR / "processed_data.csv",
    index=False
)

print("\nProcessed dataset saved.")

print("\nEDA Completed Successfully.")