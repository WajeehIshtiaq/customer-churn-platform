from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent
processed_folder = project_folder / "data" / "processed"

df = pd.read_csv(processed_folder / "telco_churn_clean.csv")

service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

# Har "Yes" ko ek subscribed service count karo
df["ServiceCount"] = df[service_columns].eq("Yes").sum(axis=1)

print("\nServiceCount ki examples:")
print(
    df[["customerID"] + service_columns + ["ServiceCount"]]
    .head()
    .to_string(index=False)
)

print("\nServiceCount ke hisaab se customer counts:")
print(df["ServiceCount"].value_counts().sort_index())

output_path = processed_folder / "telco_churn_features.csv"
df.to_csv(output_path, index=False)

print("\nFeature dataset saved:", output_path)