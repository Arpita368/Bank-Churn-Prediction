import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/Churn_Modelling.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())


print("\nStatistics:")
print(df.describe())

print("\nExited Count:")
print(df['Exited'].value_counts())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

df.drop(
    ['RowNumber','CustomerId','Surname'],
    axis=1,
    inplace=True
)

print("\nRemaining Columns:")
print(df.columns)


plt.figure(figsize=(6,4))

sns.countplot(
    x='Exited',
    data=df
)

plt.title("Customer Churn Distribution")

plt.xlabel("Exited (0 = Stayed, 1 = Churned)")

plt.ylabel("Number of Customers")

plt.savefig(
    "graphs/churn_distribution.png"
)

plt.show()



plt.figure(figsize=(7,5))

sns.boxplot(
    x='Exited',
    y='Age',
    data=df
)

plt.title("Age vs Customer Churn")

plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Customer Age")

plt.savefig(
    "graphs/age_vs_churn.png"
)

plt.show()



plt.figure(figsize=(7,5))

sns.boxplot(
    x='Exited',
    y='Balance',
    data=df
)

plt.title("Balance vs Customer Churn")

plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Account Balance")

plt.savefig(
    "graphs/balance_vs_churn.png"
)

plt.show()

plt.figure(figsize=(7,5))

sns.countplot(
    x='IsActiveMember',
    hue='Exited',
    data=df
)

plt.title("Active Member vs Customer Churn")

plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.savefig(
    "graphs/active_member_vs_churn.png"
)

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    x='Geography',
    hue='Exited',
    data=df
)

plt.title("Geography vs Customer Churn")

plt.xlabel("Country")
plt.ylabel("Customers")

plt.savefig(
"graphs/geography_vs_churn.png"
)

plt.show()


plt.figure(figsize=(7,5))

sns.histplot(
    df['Age'],
    bins=30,
    kde=True
)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.savefig(
"graphs/age_distribution.png"
)

plt.show()



plt.figure(figsize=(7,5))

sns.histplot(
    df['Balance'],
    bins=30,
    kde=True
)

plt.title(
"Balance Distribution"
)

plt.xlabel(
"Account Balance"
)

plt.savefig(
"graphs/balance_distribution.png"
)

plt.show()



plt.figure(figsize=(7,5))

sns.histplot(
    df['CreditScore'],
    bins=30,
    kde=True
)

plt.title(
"Credit Score Distribution"
)

plt.xlabel(
"Credit Score"
)

plt.savefig(
"graphs/creditscore_distribution.png"
)

plt.show()

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title(
"Correlation Heatmap"
)

plt.savefig(
"graphs/heatmap.png"
)

plt.show()

df['Gender'] = df['Gender'].map(
{
    'Male':0,
    'Female':1
})

df = pd.get_dummies(
    df,
    columns=['Geography'],
    drop_first=True
)

print(df.head())


X = df.drop(
    'Exited',
    axis=1
)

y = df['Exited']

print(
"\nX shape:",
X.shape
)

print(
"\ny shape:",
y.shape
)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

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
processed = pd.DataFrame(X_scaled)

processed['Exited'] = y.values

processed.to_csv(
"outputs/processed_data.csv",
index=False
)

print("Saved processed dataset")