import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# ====================
# LOAD DATA
# ====================

train = pd.read_csv(
"data/ChurnZero_dataset_v1.csv"
)

test = pd.read_csv(
"data/ChurnZero_test_v1.csv"
)


# save customer ids
test_ids = test['customer_id']


# ====================
# TRAIN DATA
# ====================

y = train['churn']

X = train.drop(
['customer_id','churn'],
axis=1
)

X = pd.get_dummies(
X,
drop_first=True
)

X = X.fillna(
X.median(
numeric_only=True
)
)


# ====================
# TEST DATA
# ====================

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


# ====================
# SCALE
# ====================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

test_scaled = scaler.transform(
test_X
)


# ====================
# TRAIN BEST MODEL
# ====================

model = HistGradientBoostingClassifier(
random_state=42
)

model.fit(
X_scaled,
y
)


# ====================
# PREDICT
# ====================

prob = model.predict_proba(
test_scaled
)[:,1]

pred = (
prob > 0.5
).astype(int)


# ====================
# SAVE CSV
# ====================

submission = pd.DataFrame({

'customer_id':test_ids,

'churn_prediction':pred,

'churn_probability':prob

})

submission.to_csv(

"outputs/Predictions.csv",

index=False

)

print(
"Predictions saved!"
)

print(
submission.head()
)