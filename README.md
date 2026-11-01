\# Customer Churn \& Revenue Risk Analytics Platform



An end-to-end machine learning project that predicts customer churn,

prioritizes customers using illustrative revenue exposure scores, and

serves predictions through a containerized API deployed on AWS.



\## Project highlights



\- Analyzed 7,043 customers from the IBM Telco Customer Churn dataset.

\- Resolved 11 missing TotalCharges values for customers with zero tenure,

&#x20; using a documented zero-billing assumption.

\- Engineered ServiceCount from six subscribed add-on services.

\- Compared Logistic Regression, Random Forest, and Gradient Boosting

&#x20; using five-fold stratified cross-validation.

\- Tuned Gradient Boosting across eight parameter combinations: 40 fits.

\- Selected a classification threshold using training out-of-fold predictions.

\- Generated SHAP explanations for 500 held-out customers.

\- Implemented batch scoring and a validated FastAPI prediction endpoint.

\- Deployed a Docker container through Amazon ECR to AWS Lambda.

\- Connected Lambda to an API Gateway HTTP API.

\- Used Amazon S3 for storage and Athena for SQL analysis.



\## Technology stack



Python, Pandas, NumPy, Scikit-learn, Matplotlib, SHAP, Joblib,

FastAPI, Pydantic, Mangum, Docker, Git, AWS CLI, Amazon S3,

Athena, ECR, AWS Lambda, API Gateway, and CloudWatch.



\## Model development



The dataset was split into 5,634 training customers and 1,409 test customers

using a stratified 80/20 split with random\_state=42.



The preprocessing pipeline applies numerical scaling and categorical

one-hot encoding inside cross-validation. There are 20 input columns

before encoding; customerID and Churn are excluded from model inputs.



\### Mean cross-validation results



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 0.8021 | 0.6527 | 0.5438 | 0.5926 | 0.8462 |

| Random Forest | 0.8033 | 0.6759 | 0.4997 | 0.5742 | 0.8445 |

| Gradient Boosting | 0.8060 | 0.6672 | 0.5378 | 0.5954 | 0.8482 |



Selected Gradient Boosting settings:



\- n\_estimators: 200

\- learning\_rate: 0.05

\- max\_depth: 2

\- Best mean tuning ROC-AUC: 0.8499



The decision threshold of 0.35 achieved the highest training out-of-fold

F1 among nine tested thresholds. The test set was not used to select

hyperparameters or the threshold.



\### Held-out test results



| Metric | Result |

|---|---:|

| Customers | 1,409 |

| Accuracy | 0.7807 |

| Precision | 0.5690 |

| Recall | 0.7166 |

| F1 | 0.6343 |

| ROC-AUC | 0.8457 |

| Average precision | 0.6645 |

| Correctly identified churners | 268 |

| Missed churners | 106 |

| False alarms | 203 |

| Correctly identified non-churners | 832 |



\## Revenue exposure



RevenueExposureScore = ChurnProbability × MonthlyCharges



This score supports customer prioritization. It is illustrative exposure,

not confirmed revenue loss, guaranteed savings, or a calibrated forecast

for a defined future period.



The full-dataset batch demonstration scored 7,043 customers and flagged

2,284 at threshold 0.35. This dataset includes training customers, so

these demonstration results are not an independent performance evaluation.



\## Explainability



SHAP explanations were generated for 500 held-out customers.

An additivity check confirmed that SHAP contributions reconstruct

the model scores.



Leading features included tenure, month-to-month contracts,

fiber-optic internet, and monthly charges.



SHAP contributions are expressed in log-odds, not percentage points.

Feature contributions describe model behavior, not causal effects.



\## AWS architecture



\- S3 stores raw data, processed data, model artifacts, predictions,

&#x20; and Athena query results.

\- Athena queries customer and prediction data stored in S3.

\- ECR stores the container image.

\- Lambda runs the FastAPI application through the Mangum adapter.

\- API Gateway exposes GET /health and POST /predict.

\- CloudWatch stores Lambda execution logs.



The deployed container includes the model and prediction configuration.

Inference does not download the model from S3 for each request.



\### Deployment configuration



| Setting | Value |

|---|---|

| Region | ap-southeast-2 |

| Lambda function | customer-churn-api |

| Architecture | x86\_64 |

| Memory | 1,024 MB |

| Timeout | 30 seconds |

| API stage | $default |

| Automatic deployment | Enabled |

| Default route throttling rate | 2 requests/second |

| Default route burst limit | 5 |

| Lambda log retention | 7 days |



The demo API currently has no authentication. Throttling is a

best-effort traffic control, not authentication or a guaranteed cost cap.



\## Verified deployment checks



\- Live GET /health returned status=ok and model\_loaded=true.

\- Live POST /predict matched the local Docker and direct Lambda result

&#x20; for the same demonstration customer.

\- Negative MonthlyCharges returned HTTP 422.

\- Batch predictions matched all 1,409 held-out test customer predictions.



For DEMO-001, the Docker, Lambda, and live API checks returned:



\- Churn probability: 0.6342058472831524

\- Threshold: 0.35

\- Risk flag: Flagged

\- Revenue exposure score: 18.931044541402102



One repeated Lambda invocation recorded 17.36 ms execution time and

283 MB maximum memory used. This is a single observation, not an

end-to-end latency benchmark.



\## Main project files



| File | Purpose |

|---|---|

| main.py | Data inspection and cleaning |

| EDA.py | Exploratory analysis and reports |

| features.py | ServiceCount feature engineering |

| train\_model.py | Model comparison, tuning, and pipeline saving |

| threshold\_analysis.py | Training out-of-fold threshold comparison |

| evaluate\_model.py | Held-out test evaluation |

| explain\_model.py | SHAP explanations |

| revenue\_report.py | Revenue exposure reporting |

| batch\_predict.py | CSV batch scoring |

| check\_predictions.py | Batch/test prediction consistency checks |

| api.py | Validated FastAPI endpoints |

| lambda\_handler.py | Mangum adapter for Lambda |

| Dockerfile | Lambda container packaging |

| requirements.txt | Local Python dependencies |

| requirements-lambda.txt | Container inference dependencies |



\## Local setup



The development environment uses Python 3.14 on Windows.



```powershell

py -3.14 -m venv .venv

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



Place the original dataset at:



data/raw/telco\_churn.csv



Data and trained model files are not included in Git. Generate the

artifacts locally or obtain the matching trusted project artifacts.



\### Data preparation and training



```powershell

.\\.venv\\Scripts\\python.exe main.py

.\\.venv\\Scripts\\python.exe EDA.py

.\\.venv\\Scripts\\python.exe features.py

.\\.venv\\Scripts\\python.exe train\_model.py

.\\.venv\\Scripts\\python.exe threshold\_analysis.py

.\\.venv\\Scripts\\python.exe evaluate\_model.py

.\\.venv\\Scripts\\python.exe explain\_model.py

.\\.venv\\Scripts\\python.exe revenue\_report.py

```



Training produces the saved pipeline. Evaluation saves the prediction

configuration used by batch scoring and the API.



\### Batch scoring



```powershell

.\\.venv\\Scripts\\python.exe batch\_predict.py --input data/processed/telco\_churn\_clean.csv --output reports/tables/batch\_predictions\_demo.csv

.\\.venv\\Scripts\\python.exe check\_predictions.py

```



\### Local API



```powershell

.\\.venv\\Scripts\\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

```



Open http://127.0.0.1:8000/docs to inspect and test the local API.



\### Build the Lambda image



Ensure the trained pipeline and prediction\_config.json exist in models/.



```powershell

docker build --platform linux/amd64 --provenance=false -t customer-churn-lambda:local .

```



\## Limitations and future work



\- Initial exploratory analysis used the full dataset before model splitting.

\- Hyperparameter tuning and threshold analysis reused the training data;

&#x20; the cross-validation procedure was not nested.

\- Predictive associations do not establish why customers leave.

\- Probability calibration and a defined prediction horizon need validation

&#x20; before interpreting exposure as expected financial loss.

\- Authentication, automated deployment, access logging, and alerting

&#x20; are future improvements.

\- Further extensions include data-drift monitoring, retraining workflows,

&#x20; business-cost-based threshold selection, and a user-facing dashboard.



\## Project status



The core analytics, model evaluation, explainability, batch scoring,

and cloud prediction workflow are implemented and demonstrated.

This is a portfolio project, not a production-ready customer system.

