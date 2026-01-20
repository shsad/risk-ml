# Executive Summary — Credit Risk ML Prototype

---

## Objective
Develop an explainable ML-based credit risk scoring model to support lending decisions under regulatory and governance constraints.

---

## Approach
- Leakage-safe feature pipeline using application-time data  
- Baseline logistic regression for transparency  
- XGBoost for improved performance  
- Policy-driven threshold selection using asymmetric costs (FN=5, FP=1)  
- Explainability via permutation importance and local case analysis  

---

## Results
- **Baseline:** ROC-AUC 0.63 | PR-AUC 0.13  
- **XGBoost:** ROC-AUC 0.67 | PR-AUC 0.16  
- **Selected decision threshold:** 0.153  
- **Precision / Recall at policy threshold:** 0.20 / 0.21  

---

## Key Drivers
- Credit amount and annuity  
- Employment stability  
- Age proxy (`DAYS_BIRTH`)  
- Education and income type  

---

## Governance & Risk
- Explicit leakage controls implemented  
- Model explainability at both global and individual levels  
- Initial fairness assessment indicates performance differences across gender groups → recommend monitoring and calibration  

---

## Recommendation
Deploy as a decision-support tool with monitoring for data drift, performance decay, and fairness metrics.  
Review threshold quarterly based on business risk appetite.  

---
