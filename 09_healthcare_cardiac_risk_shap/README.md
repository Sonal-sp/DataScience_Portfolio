# 🫀 Project 09: Cardiovascular Disease Risk Engine with SHAP Interpretability

## 1. Executive Summary & Business Impact
Cardiovascular disease remains the leading cause of global mortality. While black-box ML models frequently achieve superior predictive accuracy over traditional clinical risk scores (e.g., Framingham), clinicians routinely reject them due to lack of explainability.

This project builds a **cost-sensitive gradient boosted diagnostic model** coupled with **SHAP (SHapley Additive exPlanations)** to generate both patient-level force plots and cohort-level beeswarm plots, bridging clinical trust and algorithmic power.

---

## 2. Comparative Analysis: Diagnostic Benchmarks

| Model Architecture | PR-AUC | ROC-AUC | Recall @ 90% Specificity | Clinician Interpretability |
|---|---|---|---|---|
| **Framingham Heuristic Score** | 0.542 | 0.741 | 48.2% | Transparent Scorecard |
| **Logistic Regression** | 0.621 | 0.812 | 58.6% | Linear Odds Ratios |
| **XGBoost Classifier + SHAP** | **0.748** | **0.884** | **78.4%** | **Individual Patient SHAP Force Plots** |

---

## 3. Implementation Guide
```bash
cd 09_healthcare_cardiac_risk_shap
jupyter notebook 09_cardiac_risk_shap.ipynb
```
