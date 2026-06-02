# Credit Risk Modeling Report

## Project Summary

This project analyzes a loan application dataset to build a credit risk model that predicts borrower default. The analysis includes data cleaning, feature engineering, exploratory data analysis (EDA), and logistic regression modeling with progressive improvements.

## Dataset

- Source: `data/loan_book.csv`
- Observations: 120,358 before deduplication
- Variables: 25
- Target: `default_flag`
- Dataset split: `train` and `test` sets were already provided in the data

## Data Cleaning

Key preparation steps:

- Removed duplicate records from the dataset.
- Identified 4 variables with missing values.
- Imputed missing values for `annual_income` using the median income by `region`.
- Imputed missing values for `employment_length_years` and `num_open_accounts` using the overall median.
- Dropped `months_since_last_delinquency` due to excessive missingness.
- Converted float columns such as `age`, `employment_length_years`, `num_open_accounts`, and `months_since_oldest_account` to integer types.
- Converted `application_date` to datetime.
- Standardized inconsistent category labels in `home_ownership` and `loan_purpose`.

## Feature Engineering

The project used a combination of domain-driven transformations and credit risk modeling techniques:

- Removed forbidden or proxy variables such as `region`, `email_domain_type`, `branch_code_id`, `application_date`, and `application_dow`.
- Applied Weight of Evidence (WoE) binning for selected features: `age`, `num_open_accounts`, `credit_utilisation_pct`, and `employment_length_years`.
- Capped `annual_income` using IQR-based winsorization and created `log_annual_income_capped`.
- Log-transformed `total_revolving_balance` and `loan_amount` to reduce skew.

## Modeling Approach

A sequence of logistic regression models was trained to measure improvement from successive feature engineering steps:

1. Baseline model with scaled numerical features.
2. Model with WoE-transformed `age`.
3. Model adding capped and log-transformed annual income.
4. Model including log-transformed debt-related features.
5. Model adding WoE-transformed `num_open_accounts`.
6. Model adding WoE-transformed `employment_length_years`.
7. Final model trained on the full engineered feature set.

The models used:

- `LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42, solver='sag')`
- Evaluation metric: ROC AUC

## Results

- Baseline AUC: ~0.68
- Final model AUC: ~0.79
- Relative improvement: approximately 16%

The final logistic regression model produced improved discrimination while retaining interpretability through engineered features and WoE transformations.

## Model Interpretation

The final model coefficients were reviewed in terms of feature importance and odds ratios. The features with the largest coefficient magnitudes were identified to understand the strongest signals driving default risk.


