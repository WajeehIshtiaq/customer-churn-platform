# Customer Churn: Initial EDA Findings

## Dataset and cleaning
- 7,043 customers and 21 columns.
- No duplicate customer IDs.
- Converted TotalCharges from text to numeric.
- Filled 11 missing TotalCharges values with zero because tenure was
  zero months. This is a project assumption, not verified billing data.
- Preserved the original raw dataset.

## Findings
- Overall churn rate: 26.54%.
- Monthly-contract churn: 42.71%, compared with 2.83% for two-year contracts.
- Churn among customers with 0–12 months of tenure: 47.44%.
- Churn among customers with 49+ months of tenure: 9.51%.
- Churned customers had higher average monthly charges:
  74.44 versus 61.27 for retained customers.
- Electronic-check customers had the highest churn rate within
  each contract category.
- Monthly contract + electronic check:
  994 churned out of 1,850 customers (53.73%).

## Interpretation limits
- These are historical associations, not proof of causes.
- Group churn rates are not individual model predictions.
- EDA used the full dataset. A later random test split should be
  described as a learning benchmark, not a completely untouched
  evaluation dataset.

## Next steps
- Resolve the remaining SciPy import block.
- Fit preprocessing only on training data.
- Compare models using cross-validation within training data.
- Reserve the test split for final evaluation.