from pathlib import Path
import pandas as pd

# Data folder main.py ke paas banega
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data" / "raw"
data_folder.mkdir(parents=True, exist_ok=True)

file_path = data_folder / "telco_churn.csv"

# Pehli baar download; baad mein saved file use hogi
if file_path.exists():
    df = pd.read_csv(file_path)
else:
    url = (
        "https://raw.githubusercontent.com/IBM/"
        "telco-customer-churn-on-icp4d/master/data/"
        "Telco-Customer-Churn.csv"
    )
    df = pd.read_csv(url)
    df.to_csv(file_path, index=False)

print("Total customers:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nPehle 5 customers:")
print(df.head())

print("\nChurn counts:")
print(df["Churn"].value_counts())
print("\nColumn names aur data types:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate customer IDs:")
print(df["customerID"].duplicated().sum())

print("\nTotalCharges mein blank entries:")
print(df["TotalCharges"].astype(str).str.strip().eq("").sum())
# Text charges ko numbers mein convert karo.
# Blank ya invalid text NaN (missing value) ban jayega.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges ka naya data type:")
print(df["TotalCharges"].dtype)

print("\nTotalCharges missing count:")
print(df["TotalCharges"].isna().sum())

print("\nMissing charges walay customers:")
print(
    df.loc[
        df["TotalCharges"].isna(),
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)
# Assumption: zero-tenure customers ke missing total charges = 0
zero_tenure_missing = (
    df["TotalCharges"].isna()
    & df["tenure"].eq(0)
)

df.loc[zero_tenure_missing, "TotalCharges"] = 0.0

print("\nTotalCharges values filled:", zero_tenure_missing.sum())
print("Remaining missing TotalCharges:", df["TotalCharges"].isna().sum())

# Cleaned data alag save karo; raw file preserve rahegi
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)

clean_file = processed_folder / "telco_churn_clean.csv"
df.to_csv(clean_file, index=False)

print("Total customers:", len(df))
print("Cleaned file saved:", clean_file)
from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent
file_path = project_folder / "data" / "processed" / "telco_churn_clean.csv"

df = pd.read_csv(file_path)

# Yes = 1, No = 0
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

summary = df.groupby("Contract").agg(
    TotalCustomers=("customerID", "count"),
    ChurnedCustomers=("ChurnFlag", "sum"),
    ChurnRate=("ChurnFlag", "mean")
)

summary["ChurnRate"] = (summary["ChurnRate"] * 100).round(2)
summary = summary.rename(columns={"ChurnRate": "ChurnRatePercent"})

print("\nContract ke hisaab se churn:")
print(summary.to_string())