from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent

SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def main():
    parser = argparse.ArgumentParser(description="Score a customer CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if input_path == output_path:
        raise ValueError("Input aur output files alag honi chahiye.")

    config = json.loads(
        (PROJECT_FOLDER / "models" / "prediction_config.json")
        .read_text(encoding="utf-8")
    )

    model = joblib.load(
        PROJECT_FOLDER / "models" / config["model_file"]
    )
    threshold = float(config["threshold"])

    df = pd.read_csv(input_path, dtype={"customerID": "str"})

    if df.empty:
        raise ValueError("Input CSV mein customer rows nahi hain.")

    # Model ke original input columns
    model_columns = list(model.feature_names_in_)
    required = ["customerID"] + [
        column for column in model_columns if column != "ServiceCount"
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Text values ke extra spaces remove karo
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip().replace("", pd.NA)

    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs missing ya duplicate hain.")

    numeric_columns = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Training wali documented assumption
    zero_tenure_missing = (
        df["TotalCharges"].isna() & df["tenure"].eq(0)
    )
    df.loc[zero_tenure_missing, "TotalCharges"] = 0.0

    if df[required].isna().any().any():
        raise ValueError("Required columns mein missing values hain.")

    numeric_values = df[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numeric values finite honi chahiye.")

    if (numeric_values < 0).any():
        raise ValueError("Numeric values negative nahi honi chahiye.")

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen sirf 0 ya 1 hona chahiye.")

    if (df["tenure"] % 1 != 0).any():
        raise ValueError("Tenure whole months mein hona chahiye.")

    # Categorical spelling/category mistakes detect karo
    preprocessing = model.named_steps["preprocessing"]
    encoder = preprocessing.named_transformers_["categorical"]
    categorical_columns = next(
        columns for name, _, columns in preprocessing.transformers_
        if name == "categorical"
    )

    for column, allowed in zip(categorical_columns, encoder.categories_):
        invalid = ~df[column].isin(allowed)
        if invalid.any():
            values = df.loc[invalid, column].unique().tolist()
            raise ValueError(f"{column}: unknown categories {values}")

    # ServiceCount dobara calculate karo for consistent inference
    df["ServiceCount"] = df[SERVICE_COLUMNS].eq("Yes").sum(axis=1)

    # Churn target aur customerID model ko nahi diye jaate
    X = df[model_columns]
    churn_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, churn_index]
    flagged = probabilities >= threshold

    report = df[["customerID", "MonthlyCharges"]].copy()
    report["ChurnProbability"] = probabilities
    report["PredictedChurn"] = flagged.astype(int)
    report["RiskFlag"] = np.where(flagged, "Flagged", "Not flagged")
    report["RevenueExposureScore"] = (
        probabilities * df["MonthlyCharges"]
    )

    report = report.sort_values(
        "RevenueExposureScore", ascending=False
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print("Customers scored:", len(report))
    print("Threshold:", threshold)
    print("Flagged customers:", int(flagged.sum()))
    print("\nTop 5 by exposure:")
    print(report.head().round(4).to_string(index=False))
    print("\nPredictions saved:", output_path)
    print("Exposure scores are illustrative, not confirmed revenue losses.")


if __name__ == "__main__":
    main()