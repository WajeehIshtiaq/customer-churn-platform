from pathlib import Path
import pandas as pd

project_folder = Path(__file__).resolve().parent
tables_folder = project_folder / "reports" / "tables"

predictions = pd.read_csv(tables_folder / "test_predictions.csv")

customers = pd.read_csv(
    project_folder / "data" / "processed" / "telco_churn_features.csv"
)

# Customer details aur predictions customerID par join karo
report = predictions.merge(
    customers[["customerID", "MonthlyCharges", "Contract", "tenure"]],
    on="customerID",
    how="left",
    validate="one_to_one",
)

if report["MonthlyCharges"].isna().any():
    raise ValueError("Kuch customers ki MonthlyCharges missing hain.")

# Illustrative exposure score; confirmed revenue loss nahi
report["RevenueExposureScore"] = (
    report["ChurnProbability"] * report["MonthlyCharges"]
)

report["RiskFlag"] = report["PredictedChurn"].map({
    1: "Flagged",
    0: "Not flagged",
})

# Sabse zyada exposure walay customers pehle
report = report.sort_values(
    "RevenueExposureScore", ascending=False
).reset_index(drop=True)

report.insert(0, "ExposureRank", range(1, len(report) + 1))

output_path = tables_folder / "test_revenue_exposure.csv"
report.to_csv(output_path, index=False)

summary = report.groupby("RiskFlag").agg(
    Customers=("customerID", "size"),
    TotalMonthlyCharges=("MonthlyCharges", "sum"),
    TotalExposureScore=("RevenueExposureScore", "sum"),
    AverageChurnProbability=("ChurnProbability", "mean"),
)

summary.to_csv(tables_folder / "test_revenue_summary.csv")

print("Customers in report:", len(report))

print("\nRevenue-exposure summary:")
print(summary.round(2).to_string())

print("\nTop 10 customers by exposure score:")
print(
    report[[
        "customerID",
        "MonthlyCharges",
        "ChurnProbability",
        "RiskFlag",
        "RevenueExposureScore",
    ]].head(10).round(2).to_string(index=False)
)

print("\nReport saved:", output_path)
print("\nNote: Historical test-data demo; exposure scores are not confirmed losses.")