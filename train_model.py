from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_folder = Path(__file__).resolve().parent
file_path = project_folder / "data" / "processed" / "telco_churn_features.csv"

df = pd.read_csv(file_path)

# Inputs aur answer alag karo
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Test customers:", len(X_test))
print("Input columns:", X_train.shape[1])

print("\nTraining churn percentage:", round(y_train.mean() * 100, 2))
print("Test churn percentage:", round(y_test.mean() * 100, 2))