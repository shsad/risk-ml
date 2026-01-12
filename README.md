# Explainable Credit Risk Scoring (Home Credit)

## Goal
Predict probability of default (PD) for loan applicants using application and credit bureau signals.

## Stakeholders
- Risk Management (model owner)
- Compliance / Model Risk Management (auditability)
- Business (approval policy)
- Ops (manual review queue)

## Constraints
- Avoid data leakage
- Explainability required (global + individual decisions)
- Stable performance, monitoring plan included
- Bias/fairness awareness (proxy features)

## Success Metrics
- ROC-AUC + PR-AUC
- Recall at fixed precision (policy-driven)
- Expected cost with cost ratio: FN=5, FP=1

## Deliverables
- Reproducible pipeline (data validation → features → training → evaluation)
- Explainability report (SHAP)
- Governance notes + monitoring plan
- 1-page executive summary

## Step 1 - Feature policy

- Exclude identifiers such as `SK_ID_CURR`
- Exclude target and any leakage-prone columns
- Use only application-time features (no future or derived info)
- Handle missing values explicitly (impute or flag)
- Avoid target encoding for the baseline to ensure governance compliance

This ensures transparent, reproducible, and governance-friendly feature design.

## Step 2 — Select feature groups
From `application_train.csv`, define feature groups:

**Numeric (examples)**
- AMT_INCOME_TOTAL
- AMT_CREDIT
- AMT_ANNUITY
- DAYS_BIRTH
- DAYS_EMPLOYED
- CNT_FAM_MEMBERS

**Categorical (examples)**
- NAME_CONTRACT_TYPE
- CODE_GENDER
- NAME_INCOME_TYPE
- NAME_EDUCATION_TYPE
- OCCUPATION_TYPE

## Step 3 — Build a clean feature pipeline

All preprocessing is handled in `src/features.py`.  
Design principles:
- No ad-hoc preprocessing in notebooks
- Everything reproducible
- Use `sklearn.Pipeline` and `ColumnTransformer`

The pipeline:
- Imputes missing numeric values with median
- Imputes missing categorical values with most frequent
- One-hot encodes categorical features
- Splits data into train and validation sets reproducibly

## Step 4 — Baseline model results

| Model                | ROC-AUC | PR-AUC | Notes |
|----------------------|---------|--------|-------|
| Logistic Regression  | 0.6341  | 0.1270 | Baseline, simple linear model |
| XGBoost              | 0.6679  | 0.1556 | Nonlinear model, better recall on minority class |

XGBoost improves precision-recall performance, confirming stronger signal capture.

## Step 5 — Decision policy (cost-based)


- Costs assumed: FN = 5, FP = 1  
- Selected threshold: t = 0.153  
- Precision = 0.203  
- Recall = 0.207  

Threshold is chosen by minimizing expected cost, aligning model output with business risk appetite.  
This produces an auditable decision rule rather than relying on an arbitrary 0.5 cutoff.

