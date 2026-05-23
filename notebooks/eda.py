import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "data/ChurnZero_dataset_v1.csv"
)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(
    df['churn']
    .value_counts(normalize=True)
)
plt.figure(figsize=(6,4))

sns.countplot(
    x='churn',
    data=df
)

plt.title(
"Customer Churn Distribution"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Customers"
)

plt.savefig(
"graphs/churn_distribution.png"
)

plt.show()

# -----------------------------------
# Digital Engagement vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='digital_engagement_index',
    data=df
)

plt.title(
"Digital Engagement vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Digital Engagement Index"
)

plt.savefig(
"graphs/digital_engagement_vs_churn.png"
)

plt.show()
# -----------------------------------
# Complaints vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='total_complaints',
    data=df
)

plt.title(
"Total Complaints vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Total Complaints"
)

plt.savefig(
"graphs/complaints_vs_churn.png"
)

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='satisfaction_score',
    data=df
)

plt.title(
"Satisfaction Score vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Satisfaction Score"
)

plt.savefig(
"graphs/satisfaction_vs_churn.png"
)

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    x='retention_offer_accepted',
    hue='churn',
    data=df
)

plt.title(
"Retention Offer Accepted vs Churn"
)

plt.xlabel(
"Retention Offer Accepted"
)

plt.ylabel(
"Customers"
)

plt.legend(
title="Churn"
)

plt.savefig(
"graphs/retention_offer_vs_churn.png"
)

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='last_login_days',
    data=df
)

plt.title(
"Last Login Days vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Days Since Last Login"
)

plt.savefig(
"graphs/last_login_vs_churn.png"
)

plt.show()

# -----------------------------------
# Encoding categorical columns
# -----------------------------------

df['gender'] = df['gender'].map(
{
'Male':1,
'Female':0
}
)

df = pd.get_dummies(
df,

columns=[

'marital_status',

'education_level',

'occupation_type',

'income_band',

'income_category',

'city_tier',

'region',

'customer_segment',

'onboarding_channel',

'relationship_type',

'primary_account_type',

'card_category',

'competitor_bank_offer_awareness',

'customer_feedback_sentiment'

],

drop_first=True
)

print(
"\nAfter Encoding:"
)

print(
df.head()
)

# -----------------------------------
# Missing value handling
# -----------------------------------

df['app_rating_given'] = (
df['app_rating_given']
.fillna(
df['app_rating_given']
.median()
)
)

print(
"\nRemaining Missing Values:"
)

print(
df.isnull()
.sum()
.sum()
)
# -----------------------------------
# Scaling
# -----------------------------------

# Remove target and ID
X = df.drop(
[
'customer_id',
'churn'
],
axis=1
)

y = df['churn']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
X
)

print(
"\nScaled shape:"
)

print(
X_scaled.shape
)

print(
"\nFirst scaled row:"
)

print(
X_scaled[0]
)
# -----------------------------------
# Train Test Split
# -----------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

X_scaled,

y,

test_size=0.2,

random_state=42,

stratify=y

)

print(
"\nTrain shape:"
)

print(
X_train.shape
)

print(
"\nTest shape:"
)

print(
X_test.shape
)

# -----------------------------------
# Age vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='age',
    data=df
)

plt.title(
"Age vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Age"
)

plt.savefig(
"graphs/age_vs_churn.png"
)

plt.show()

# -----------------------------------
# Average Monthly Balance vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='avg_monthly_balance',
    data=df
)

plt.title(
"Average Monthly Balance vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Average Monthly Balance"
)

plt.savefig(
"graphs/balance_vs_churn.png"
)

plt.show()

# -----------------------------------
# Credit Utilization Ratio vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='credit_utilization_ratio',
    data=df
)

plt.title(
"Credit Utilization Ratio vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Credit Utilization Ratio"
)

plt.savefig(
"graphs/credit_utilization_vs_churn.png"
)

plt.show()

# -----------------------------------
# Account Inactive Days vs Churn
# -----------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='account_inactive_days',
    data=df
)

plt.title(
"Account Inactive Days vs Churn"
)

plt.xlabel(
"Churn (0=Stayed,1=Left)"
)

plt.ylabel(
"Inactive Days"
)

plt.savefig(
"graphs/inactive_days_vs_churn.png"
)

plt.show()

# -----------------------------------
# Age Distribution
# -----------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df['age'],
    bins=30,
    kde=True
)

plt.title(
"Age Distribution"
)

plt.xlabel(
"Age"
)

plt.ylabel(
"Frequency"
)

plt.savefig(
"graphs/age_distribution.png"
)

plt.show()

# -----------------------------------
# Balance Distribution
# -----------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df['avg_monthly_balance'],
    bins=30,
    kde=True
)

plt.title(
"Average Monthly Balance Distribution"
)

plt.xlabel(
"Average Monthly Balance"
)

plt.ylabel(
"Frequency"
)

plt.savefig(
"graphs/balance_distribution.png"
)

plt.show()

# -----------------------------------
# Correlation Heatmap
# -----------------------------------

heatmap_df = df.copy()

plt.figure(figsize=(22,18))

sns.heatmap(
    heatmap_df.corr(
        numeric_only=True
    ),
    cmap='coolwarm'
)

plt.title(
"Correlation Heatmap"
)

plt.savefig(
"graphs/heatmap.png"
)

plt.show()


# -----------------------------------
# Save processed data
# -----------------------------------

processed = pd.DataFrame(
X_scaled
)

processed['churn'] = y.values

processed.to_csv(

"outputs/processed_data.csv",

index=False

)

print(
"\nProcessed data saved!"
)
