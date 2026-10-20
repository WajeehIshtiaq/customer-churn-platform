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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# Numeric aur text columns alag karo
numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = [
    column for column in X_train.columns
    if column not in numeric_columns
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)

# Training data ke andar 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nLogistic Regression cross-validation started...", flush=True)

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    error_score="raise"
)

print("\nAverage validation scores:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"{metric}: {scores[f'test_{metric}'].mean():.4f}")

# Validation ke baad saaray training customers par fit karo
model.fit(X_train, y_train)
print("\nFirst model trained successfully.")
# Har validation fold ke scores save karo
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

fold_results = pd.DataFrame({
    metric: scores[f"test_{metric}"]
    for metric in metrics
})
fold_results.index = range(1, len(fold_results) + 1)
fold_results.index.name = "Fold"

results_folder = project_folder / "reports" / "tables"
results_folder.mkdir(parents=True, exist_ok=True)

output_path = results_folder / "logistic_regression_cv.csv"
fold_results.to_csv(output_path)

print("\nValidation results saved:", output_path)

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# Logistic Regression ke existing scores reuse karo
comparison_rows = [{
    "Model": "Logistic Regression",
    **{
        metric: scores[f"test_{metric}"].mean()
        for metric in metrics
    }
}]

other_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

for name, classifier in other_models.items():
    print(f"\n{name} validation started...", flush=True)

    candidate = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", classifier)
    ])

    candidate_scores = cross_validate(
        candidate,
        X_train,
        y_train,
        cv=cv,
        scoring={metric: metric for metric in metrics},
        error_score="raise"
    )

    comparison_rows.append({
        "Model": name,
        **{
            metric: candidate_scores[f"test_{metric}"].mean()
            for metric in metrics
        }
    })

    # Har model ke individual fold scores save karo
    candidate_folds = pd.DataFrame({
        metric: candidate_scores[f"test_{metric}"]
        for metric in metrics
    })
    candidate_folds.index = range(1, len(candidate_folds) + 1)
    candidate_folds.index.name = "Fold"

    filename = name.lower().replace(" ", "_") + "_cv.csv"
    candidate_folds.to_csv(results_folder / filename)

comparison = pd.DataFrame(comparison_rows)

print("\nModel comparison - average validation scores:")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    results_folder / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")
from sklearn.model_selection import GridSearchCV

# Gradient Boosting ki different settings compare karo
tuning_pipeline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

parameter_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_depth": [2, 3]
}

search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    verbose=1,
    error_score="raise",
    return_train_score=False
)

print("\nGradient Boosting tuning started...", flush=True)
search.fit(X_train, y_train)

print("\nBest settings:")
print(search.best_params_)

print(f"\nBest mean validation ROC-AUC: {search.best_score_:.4f}")

tuning_results = pd.DataFrame(search.cv_results_)
tuning_results = tuning_results.sort_values("rank_test_score")

tuning_path = results_folder / "gradient_boosting_tuning.csv"
tuning_results.to_csv(tuning_path, index=False)

print("\nTuning results saved:", tuning_path)

# Best pipeline training data par fitted hai
best_model = search.best_estimator_
import joblib

# Trained model aur preprocessing ek saath save karo
models_folder = project_folder / "models"
models_folder.mkdir(parents=True, exist_ok=True)

model_path = models_folder / "gradient_boosting_pipeline.joblib"
joblib.dump(best_model, model_path)

print("\nTrained pipeline saved:", model_path)