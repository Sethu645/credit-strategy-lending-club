-- =====================================================================
-- portfolio_summary.sql
-- Portfolio view of loans_model: default rate, average loss and realized
-- return on capital by Lending Club grade and issue year.
-- Used for the EDA section and as a sanity check on the population.
-- =====================================================================

SELECT
    grade,
    issue_year,
    COUNT(*)                                                    AS n_loans,
    ROUND(AVG(is_default), 4)                                   AS default_rate,
    ROUND(AVG(int_rate), 2)                                     AS avg_int_rate,
    ROUND(SUM(loan_amnt) / 1e6, 2)                              AS funded_musd,
    ROUND(SUM(realized_net_profit) / 1e6, 3)                    AS net_profit_musd,
    ROUND(SUM(realized_net_profit) / SUM(loan_amnt), 4)         AS return_on_funded,
    -- loss given default: share of principal lost on charged-off loans
    ROUND(
        -SUM(CASE WHEN is_default = 1 THEN realized_net_profit END)
        / NULLIF(SUM(CASE WHEN is_default = 1 THEN loan_amnt END), 0), 4
    )                                                           AS lgd
FROM loans_model
GROUP BY grade, issue_year
ORDER BY grade, issue_year;
