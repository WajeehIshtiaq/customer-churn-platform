from pathlib import Path
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent / "reports" / "tables"

test = pd.read_csv(folder / "test_predictions.csv")
batch = pd.read_csv(folder / "batch_predictions_demo.csv")

matched = test.merge(
    batch,
    on="customerID",
    suffixes=("_test", "_batch"),
    validate="one_to_one",
)

assert len(matched) == len(test), "Kuch test customers batch mein missing hain."

np.testing.assert_allclose(
    matched["ChurnProbability_test"],
    matched["ChurnProbability_batch"],
    rtol=0,
    atol=1e-10,
)

assert (
    matched["PredictedChurn_test"] == matched["PredictedChurn_batch"]
).all(), "Risk flags match nahi kar rahe."

print(f"PASS: All {len(matched)} test customers match.")
print("Churn probabilities aur risk flags consistent hain.")