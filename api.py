from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROJECT_FOLDER = Path(__file__).resolve().parent

config = json.loads(
    (PROJECT_FOLDER / "models" / "prediction_config.json")
    .read_text(encoding="utf-8")
)

model = joblib.load(
    PROJECT_FOLDER / "models" / config["model_file"]
)

threshold = float(config["threshold"])
model_columns = list(model.feature_names_in_)
churn_index = list(model.classes_).index(1)

preprocessing = model.named_steps["preprocessing"]
encoder = preprocessing.named_transformers_["categorical"]
categorical_columns = next(
    columns for name, _, columns in preprocessing.transformers_
    if name == "categorical"
)

SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Churn prediction and illustrative revenue exposure.",
)


class CustomerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        str_min_length=1,
        allow_inf_nan=False,
    )

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1, strict=True)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, strict=True)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(customer: CustomerInput):
    record = customer.model_dump()

    # Training wali missing-billing assumption
    if record["TotalCharges"] is None:
        if record["tenure"] == 0:
            record["TotalCharges"] = 0.0
        else:
            raise HTTPException(
                status_code=422,
                detail="TotalCharges is required when tenure is above zero.",
            )

    # Batch script ki tarah unknown categories reject karo
    for column, allowed in zip(
        categorical_columns, encoder.categories_
    ):
        if record[column] not in allowed:
            raise HTTPException(
                status_code=422,
                detail={
                    "column": column,
                    "message": "Unknown category",
                    "allowed_values": allowed.tolist(),
                },
            )

    record["ServiceCount"] = sum(
        record[column] == "Yes" for column in SERVICE_COLUMNS
    )

    features = pd.DataFrame([record])[model_columns]
    probability = float(
        model.predict_proba(features)[0, churn_index]
    )
    flagged = probability >= threshold

    return {
        "customerID": customer.customerID,
        "ChurnProbability": probability,
        "Threshold": threshold,
        "PredictedChurn": int(flagged),
        "RiskFlag": "Flagged" if flagged else "Not flagged",
        "RevenueExposureScore": (
            probability * customer.MonthlyCharges
        ),
        "Note": "Illustrative exposure score, not confirmed revenue loss.",
    }