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






