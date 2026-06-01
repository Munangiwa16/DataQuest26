# Credit Risk Modeling Project

This repository contains a credit risk modeling analysis built on a loan book dataset. The project includes data cleaning, exploratory analysis, feature engineering, and logistic regression modeling for predicting borrower default risk.

## Project Contents

- `notebooks/Credit_Modeling.ipynb` - Main Jupyter notebook with data preparation, feature engineering, model training, evaluation, and coefficient analysis.
- `data/loan_book.csv` - Loan application dataset used for the analysis.
- `reports/Modeling_Summary_Munangiwa.qmd` - Quarto summary report with credit risk concepts, data cleaning approach, EDA findings, and modeling rationale.
- `reports/Modeling_Summary_Munangiwa.pdf` - Rendered PDF version of the summary report.
- `requirements.txt` - Python package dependencies required to run the notebook and analysis.
- Images used in the report:
  - `images/Target.png`
  - `images/annual_income.png`
  - `images/age.png`
  - `images/employ.png`
  - `images/num_open.png`
  - `images/credit_util.png`
  - `images/interest.png`
  - `images/age_vs_older.png`
  - `images/sigmoid.png`
  - `images/trees.png`
  - `images/neural.png`

## Overview

The analysis focuses on credit risk modeling for predicting the probability of default (PD) over a 12-month horizon. It explains credit risk metrics such as Probability of Default (PD), Exposure at Default (EAD), Loss Given Default (LGD), and how these are used in banking risk assessment.

Key modeling techniques described in the report include:
- Logistic regression
- Tree-based models (decision trees, random forest, gradient boosting)
- Neural networks
- Other classifiers such as SVM, KNN, and Naïve Bayes

The analysis also covers data cleaning, missing value imputation, feature consistency, and prohibited features for credit models.

## Data Cleaning and Feature Engineering

The notebook performs the following data preparation steps:
- Remove duplicate records
- Impute missing values for `annual_income`, `employment_length_years`, and `num_open_accounts`
- Drop `months_since_last_delinquency` due to excessive missingness
- Convert numeric columns to appropriate integer types and parse `application_date` as datetime
- Standardize categorical labels for variables such as `home_ownership` and `loan_purpose`
- Remove forbidden or proxy variables like `region`, `email_domain_type`, `branch_code_id`, and `application_date`

Feature engineering includes:
- Weight of Evidence (WoE) transformation for variables such as `age`, `num_open_accounts`, `credit_utilisation_pct`, and `employment_length_years`
- Capping and log-transforming `annual_income`
- Log-transforming `total_revolving_balance` and `loan_amount`

## Modeling

The notebook builds a sequence of logistic regression models with progressively enhanced features and evaluates them using ROC AUC and classification metrics. The final model uses transformed features and balanced class weighting.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Open `notebooks/Credit_Modeling.ipynb` in Jupyter or VS Code and execute the cells sequentially. The notebook contains the full data cleaning, feature engineering, modeling, and evaluation workflow.

## Notes

- The `Modeling_Summary_Munangiwa.qmd` report includes the conceptual background for credit risk and the EDA findings used to inform feature engineering.
- The dataset appears to contain an imbalanced target distribution, which the notebook addresses using class weighting in logistic regression.
- The package list in `requirements.txt` includes tools for data analysis and Streamlit.
