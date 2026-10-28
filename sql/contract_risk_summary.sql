-- Demo scoring across the full dataset.
-- Exposure scores are illustrative, not confirmed revenue losses.

SELECT
    c.contract,
    COUNT(*) AS total_customers,
    SUM(p.predicted_churn) AS flagged_customers,
    ROUND(AVG(p.churn_probability) * 100, 2)
        AS average_churn_probability_percent,
    ROUND(SUM(p.revenue_exposure_score), 2)
        AS total_exposure_score
FROM customer_churn.customers c
JOIN customer_churn.predictions_demo p
    ON c.customer_id = p.customer_id
GROUP BY c.contract
ORDER BY total_exposure_score DESC;