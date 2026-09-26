from pathlib import Path
import pandas as pd

# Cleaned dataset load karo
project_folder = Path(__file__).resolve().parent
file_path = project_folder / "data" / "processed" / "telco_churn_clean.csv"

df = pd.read_csv(file_path)

# Churn: Yes = 1, No = 0
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# Har contract type ke customers aur churn calculate karo
summary = df.groupby("Contract").agg(
    TotalCustomers=("customerID", "count"),
    ChurnedCustomers=("ChurnFlag", "sum"),
    ChurnRate=("ChurnFlag", "mean")
)

summary["ChurnRate"] = (summary["ChurnRate"] * 100).round(2)
summary = summary.rename(
    columns={"ChurnRate": "ChurnRatePercent"}
)

print("\nContract ke hisaab se churn:")
print(summary.to_string())
customer_comparison = df.groupby("Churn").agg(
    TotalCustomers=("customerID", "count"),
    AverageTenureMonths=("tenure", "mean"),
    AverageMonthlyCharges=("MonthlyCharges", "mean")
).round(2)

print("\nService chhorne aur rehne walay customers ka comparison:")
print(customer_comparison.to_string())
# Customers ko tenure ke mutabiq groups mein divide karo
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]
)

tenure_summary = df.groupby("TenureGroup", observed=True).agg(
    TotalCustomers=("customerID", "count"),
    ChurnedCustomers=("ChurnFlag", "sum"),
    ChurnRatePercent=("ChurnFlag", "mean")
)

tenure_summary["ChurnRatePercent"] = (
    tenure_summary["ChurnRatePercent"] * 100
).round(2)

print("\nTenure ke hisaab se churn:")
print(tenure_summary.to_string())
import matplotlib.pyplot as plt

# Graphs save karne ka folder
figures_folder = project_folder / "reports" / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)

# Contract aur tenure ke churn rates
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

summary["ChurnRatePercent"].plot.bar(
    ax=axes[0], color="steelblue", rot=0
)
axes[0].set_title("Churn Rate by Contract")
axes[0].set_xlabel("Contract")
axes[0].set_ylabel("Churn rate (%)")

tenure_summary["ChurnRatePercent"].plot.bar(
    ax=axes[1], color="darkorange", rot=0
)
axes[1].set_title("Churn Rate by Tenure")
axes[1].set_xlabel("Tenure")
axes[1].set_ylabel("Churn rate (%)")

for ax in axes:
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

fig.tight_layout()

chart_path = figures_folder / "churn_patterns.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)

print("\nGraphs saved:", chart_path)
# Summary tables save karne ka folder
tables_folder = project_folder / "reports" / "tables"
tables_folder.mkdir(parents=True, exist_ok=True)

summary.to_csv(tables_folder / "churn_by_contract.csv")
customer_comparison.to_csv(tables_folder / "churn_customer_comparison.csv")
tenure_summary.to_csv(tables_folder / "churn_by_tenure.csv")

# Payment method ke hisaab se churn
payment_summary = df.groupby("PaymentMethod").agg(
    TotalCustomers=("customerID", "count"),
    ChurnedCustomers=("ChurnFlag", "sum"),
    ChurnRatePercent=("ChurnFlag", "mean")
)

payment_summary["ChurnRatePercent"] = (
    payment_summary["ChurnRatePercent"] * 100
).round(2)

payment_summary = payment_summary.sort_values(
    "ChurnRatePercent", ascending=False
)

payment_summary.to_csv(tables_folder / "churn_by_payment_method.csv")

print("\nPayment method ke hisaab se churn:")
print(payment_summary.to_string())

print("\n4 summary tables saved:", tables_folder)
payment_contract = df.groupby(
    ["PaymentMethod", "Contract"]
).agg(
    TotalCustomers=("customerID", "count"),
    ChurnedCustomers=("ChurnFlag", "sum"),
    ChurnRatePercent=("ChurnFlag", "mean")
)

payment_contract["ChurnRatePercent"] = (
    payment_contract["ChurnRatePercent"] * 100
).round(2)

print("\nPayment method aur contract ka combined comparison:")
print(payment_contract.to_string())

payment_contract.to_csv(
    tables_folder / "churn_by_payment_and_contract.csv"
)

print("\nCombined comparison saved.")