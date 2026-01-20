# Governance and Monitoring Report  
Credit Risk PD Model — XGBoost (application_train)

---

## Governance notes
- Model built for credit risk PD estimation using `application_train` data.  
- Developed under internal governance with reproducible pipeline, versioned data, and audit-ready artifacts.  

---

## Data sources used
- Primary dataset: `application_train` from Home Credit.  
- No external or unlabeled data used.  

---

## Leakage controls
- Excluded identifiers (`SK_ID_CURR`), target (`TARGET`), and known post-application variables.  
- Heuristic leakage scan confirmed no high-correlation fields with target beyond expected behavior.  

---

## Feature policy
- Included demographic, income, credit, and employment features.  
- Excluded IDs, target, and proxy variables (e.g., `region_rating`).  
- One-hot encoding applied to categorical variables.  

---

## Model choice rationale
- Baseline logistic regression tested.  
- XGBoost selected for superior PR-AUC and interpretability via SHAP.  

---

## Metrics rationale
- PR-AUC chosen due to 8% positive base rate.  
- ROC-AUC used for secondary validation.  

---

## Threshold policy
- Business cost ratio FN:FP = 5:1.  
- Optimal threshold selected at **t = 0.153**, balancing recall and precision.  

---

## Limitations
- No temporal validation split.  
- Potential proxy bias from correlated socioeconomic features.  
- Missing SHAP visualizations for some examples.  
- Only one table (`application_train`) used; no bureau or credit card data.  

---

## Monitoring plan (production)

### Data quality
- Track missingness shift per key fields (`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_EMPLOYED`).  
- Monitor new categorical levels rate for `NAME_EDUCATION_TYPE`, `OCCUPATION_TYPE`.  
- Distribution checks for income/credit ratio and employment length.  

### Drift
- Compute PSI (Population Stability Index) monthly on top 10 features.  
- Monitor PR-AUC on labeled backtesting data.  
- Alert if PSI > 0.2 or PR-AUC drops >10% from baseline.  

### Decision monitoring
- Track approval and decline rates weekly.  
- Monitor manual review volume linked to threshold t = 0.153.  
- Review outcome feedback loop for delayed labels and retraining cadence.  

---

*Prepared for internal model governance and risk monitoring documentation.*
