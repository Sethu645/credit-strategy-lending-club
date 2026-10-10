-- =====================================================================
-- features.sql
-- Builds the modelling table from the raw Lending Club sample.
--
-- Input : raw_loans   (lc_sample.csv loaded as-is by src/load_data.py)
-- Output: loans_model (one row per loan, application-time features only,
--                      plus outcome and cash-flow fields for profit simulation)
--
-- Population rules
--   * 36-month loans only, issued 2007-2015. Every loan in this window had
--     reached maturity by the data snapshot, so outcomes are complete.
--     Later vintages only contain loans that already ended (early payoff or
--     charge-off), which biases the default rate.
--   * Individual applications only (joint apps use different income fields).
--   * Completed outcomes only: 'Fully Paid' or 'Charged Off'.
-- =====================================================================

DROP TABLE IF EXISTS loans_model;

CREATE TABLE loans_model AS
WITH parsed AS (
    SELECT
        r.*,
        -- 'Aug-2014' -> year 2014, month 8
        CAST(substr(issue_d, 5, 4) AS INTEGER)                                  AS issue_year,
        (instr('JanFebMarAprMayJunJulAugSepOctNovDec', substr(issue_d, 1, 3)) + 2) / 3
                                                                                AS issue_month,
        CAST(substr(earliest_cr_line, 5, 4) AS INTEGER)                         AS ecl_year,
        (instr('JanFebMarAprMayJunJulAugSepOctNovDec', substr(earliest_cr_line, 1, 3)) + 2) / 3
                                                                                AS ecl_month
    FROM raw_loans r
)
SELECT
    id,
    issue_year,
    issue_month,

    -- ---------------- target ----------------
    CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END                     AS is_default,

    -- ---------------- application-time features ----------------
    loan_amnt,
    purpose,
    home_ownership,
    verification_status,
    addr_state,
    annual_inc,
    dti,
    (fico_range_low + fico_range_high) / 2.0                                    AS fico,
    CASE
        WHEN emp_length IS NULL        THEN NULL
        WHEN emp_length = '< 1 year'   THEN 0
        WHEN emp_length = '10+ years'  THEN 10
        ELSE CAST(substr(emp_length, 1, instr(emp_length, ' ') - 1) AS INTEGER)
    END                                                                         AS emp_years,
    (issue_year * 12 + issue_month) - (ecl_year * 12 + ecl_month)               AS credit_history_months,
    delinq_2yrs,
    inq_last_6mths,
    open_acc,
    total_acc,
    pub_rec,
    pub_rec_bankruptcies,
    revol_bal,
    revol_util,
    mort_acc,

    -- engineered affordability ratios
    loan_amnt / NULLIF(annual_inc, 0)                                           AS loan_to_income,
    installment / NULLIF(annual_inc / 12.0, 0)                                  AS payment_to_income,
    revol_bal / NULLIF(annual_inc, 0)                                           AS revol_to_income,
    CAST(open_acc AS REAL) / NULLIF(total_acc, 0)                               AS open_acc_ratio,

    -- ---------------- external risk grade ----------------
    -- Lending Club's sub-grade (A1 = safest ... G5 = riskiest) is the output of
    -- LC's own risk model. We treat it like an external bureau score: available
    -- at decision time, and the benchmark our own features must add lift over.
    -- A1 -> 1, A2 -> 2, ..., B1 -> 6, ..., G5 -> 35
    (instr('ABCDEFG', substr(sub_grade, 1, 1)) - 1) * 5
        + CAST(substr(sub_grade, 2, 1) AS INTEGER)                              AS subgrade_rank,

    -- ---------------- pricing (strategy + profit only) ----------------
    grade,
    sub_grade,
    int_rate,
    installment,

    -- ---------------- post-outcome cash flows (profit simulation ONLY) ---------
    -- Known only after the loan ends. Never used as features (target leakage).
    total_pymnt,
    recoveries,
    collection_recovery_fee,
    total_pymnt - collection_recovery_fee - loan_amnt                           AS realized_net_profit

FROM parsed
WHERE TRIM(term) = '36 months'          -- raw values carry a leading space: ' 36 months'
  AND issue_year BETWEEN 2007 AND 2015
  AND application_type = 'Individual'
  AND loan_status IN ('Fully Paid', 'Charged Off')
  AND annual_inc > 0;
