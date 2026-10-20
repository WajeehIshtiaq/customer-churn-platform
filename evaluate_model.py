from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

project_folder = Path(__file__).resolve().parent
threshold = 0.35

df = pd.read_csv(
    project_folder / "data" / "processed" / "telco_churn_features.csv"
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Original split reproduce karo
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = joblib.load(
    project_folder / "models" / "gradient_boosting_pipeline.joblib"
)

# Threshold decision record karo
config = {
    "model_file": "gradient_boosting_pipeline.joblib",
    "threshold": threshold,
    "selection_rule": "Highest training OOF F1 among 9 tested thresholds",
    "selection_f1": 0.6369,
    "split_random_state": 42,
}

config_path = project_folder / "models" / "prediction_config.json"
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

# Saved model se predictions; koi retraining nahi
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions),
    "f1": f1_score(y_test, predictions),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "average_precision": average_precision_score(y_test, probabilities),
}

print("Test customers:", len(X_test))
print("Selected threshold:", threshold)
print("\nFinal test scores:")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\nCorrectly identified churners:", tp)
print("Missed churners:", fn)
print("False alarms:", fp)
print("Correctly identified non-churners:", tn)

results_folder = project_folder / "reports" / "tables"
results_folder.mkdir(parents=True, exist_ok=True)

pd.DataFrame([{
    "Model": "Tuned Gradient Boosting",
    "Threshold": threshold,
    "TestCustomers": len(X_test),
    **metrics,
    "TruePositives": int(tp),
    "FalseNegatives": int(fn),
    "FalsePositives": int(fp),
    "TrueNegatives": int(tn),
}]).to_csv(results_folder / "final_test_metrics.csv", index=False)

pd.DataFrame({
    "customerID": df.loc[X_test.index, "customerID"].to_numpy(),
    "ActualChurn": y_test.to_numpy(),
    "ChurnProbability": probabilities,
    "PredictedChurn": predictions,
}).to_csv(results_folder / "test_predictions.csv", index=False)

print("\nTest results saved:", results_folder)
print("Prediction config saved:", config_path)