# Best Model API Notes

The saved model is a complete sklearn classifier:

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")
input_df = pd.DataFrame([{
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
}])

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0, 1]
```

Expected input columns:
['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

Optimized churn threshold:
0.156618
