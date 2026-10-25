from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

project = Path(__file__).resolve().parent
figures = project / "reports" / "figures"
tables = project / "reports" / "tables"
figures.mkdir(parents=True, exist_ok=True)
tables.mkdir(parents=True, exist_ok=True)

# Saved pipeline load karo; model dobara train nahi hoga.
pipeline = joblib.load(
    project / "models" / "gradient_boosting_pipeline.joblib"
)

df = pd.read_csv(
    project / "data" / "processed" / "telco_churn_features.csv"
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Training ke waqt wala same split.
X_train, X_test, _, _ = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = pipeline.named_steps["preprocessing"]
classifier = pipeline.named_steps["classifier"]

# Background sirf training data se.
background = X_train.sample(n=100, random_state=42)
sample = X_test.sample(n=500, random_state=42)


def transform(frame):
    values = preprocessor.transform(frame)
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)


background_values = transform(background)
sample_values = transform(sample)
feature_names = preprocessor.get_feature_names_out().tolist()

print("SHAP explanations started...", flush=True)

explainer = shap.TreeExplainer(
    classifier,
    data=background_values,
    feature_names=feature_names,
    feature_perturbation="interventional",
    model_output="raw"
)

explanations = explainer(sample_values)

# Confirm explanations reconstruct the model's raw scores.
reconstructed = (
    explanations.base_values + explanations.values.sum(axis=1)
)
np.testing.assert_allclose(
    reconstructed,
    classifier.decision_function(sample_values),
    atol=1e-5,
    rtol=1e-5
)
print("PASS: SHAP contributions match model scores.")

# Global importance across these 500 customers.
importance = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsoluteSHAP": np.abs(explanations.values).mean(axis=0)
}).sort_values("MeanAbsoluteSHAP", ascending=False)

importance.to_csv(tables / "shap_feature_importance.csv", index=False)

print("\nTop 10 transformed features:")
print(importance.head(10).round(4).to_string(index=False))

shap.plots.bar(explanations, max_display=15, show=False)
plt.savefig(
    figures / "shap_global_importance.png",
    dpi=160, bbox_inches="tight"
)
plt.close()

# Explain the first customer in the reproducible random sample.
customer_id = df.loc[sample.index[0], "customerID"]
churn_index = list(pipeline.classes_).index(1)
probability = pipeline.predict_proba(sample.iloc[[0]])[0, churn_index]

shap.plots.waterfall(explanations[0], max_display=12, show=False)
plt.savefig(
    figures / "shap_customer_waterfall.png",
    dpi=160, bbox_inches="tight"
)
plt.close()

local = pd.DataFrame({
    "Feature": feature_names,
    "TransformedValue": sample_values[0],
    "SHAPContribution": explanations.values[0]
})
local.insert(0, "customerID", customer_id)
local.to_csv(tables / "shap_customer_explanation.csv", index=False)

pd.DataFrame([{
    "customerID": customer_id,
    "ChurnProbability": probability,
    "BaseLogOdds": float(explanations.base_values[0]),
    "PredictionLogOdds": float(reconstructed[0])
}]).to_csv(tables / "shap_customer_summary.csv", index=False)

print("\nCustomers explained:", len(sample))
print("Example customer:", customer_id)
print(f"Churn probability: {probability:.4f}")
print("Graphs saved:", figures)
print("SHAP tables saved:", tables)
print("SHAP contributions are in log-odds, not percentage points.")