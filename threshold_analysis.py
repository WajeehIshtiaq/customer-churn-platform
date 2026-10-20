from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

project_folder = Path(__file__).resolve().parent

df = pd.read_csv(
    project_folder / "data" / "processed" / "telco_churn_features.csv"
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Wohi original split; test data threshold selection mein use nahi hoga
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

saved_model = joblib.load(
    project_folder / "models" / "gradient_boosting_pipeline.joblib"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Training data par out-of-fold predictions started...", flush=True)

# Har customer ki prediction us fold ke model se,
# jiski fitting mein woh customer shamil nahi tha
probabilities = cross_val_predict(
    clone(saved_model),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

rows = []

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train, predictions, labels=[0, 1]
    ).ravel()

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(y_train, predictions),
        "F1": f1_score(y_train, predictions),
        "FlaggedCustomers": int(predictions.sum()),
        "FalseAlarms": int(fp),
        "MissedChurners": int(fn),
    })

results = pd.DataFrame(rows)

print("\nThreshold comparison:")
print(results.round(4).to_string(index=False))

output_folder = project_folder / "reports" / "tables"
output_folder.mkdir(parents=True, exist_ok=True)

results.to_csv(output_folder / "threshold_comparison.csv", index=False)

# Predictions save karo, taake analysis dobara training ke baghair ho sake
oof_results = pd.DataFrame({
    "customerID": df.loc[X_train.index, "customerID"].to_numpy(),
    "ActualChurn": y_train.to_numpy(),
    "ChurnProbability": probabilities,
})
oof_results.to_csv(
    output_folder / "training_oof_predictions.csv",
    index=False,
)

print("\nThreshold comparison aur training predictions saved.")