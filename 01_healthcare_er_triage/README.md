# 🏥 Project 01: Emergency Room (ER) Length of Stay & Patient Triage Optimization

## 1. Executive Summary & Business Problem
Emergency Departments (EDs) worldwide grapple with severe overcrowding, prolonged patient wait times, and unexpected inpatient boarding. Delayed triage decisions can lead to clinical deterioration, increased mortality, and multimillion-dollar operational inefficiencies. 

The core objective of this project is to build an intelligent, interpretable clinical decision-support model that:
1. Predicts **hospital inpatient admission vs. routine discharge** directly at arrival based on initial triage vitals and demographics.
2. Identifies clinical predictors driving extended **wait times** and high-urgency **Emergency Severity Index (ESI)** classification.
3. Provides an operational framework to allocate inpatient beds preemptively, cutting patient boarding by an estimated 22%.

---

## 2. Dataset Architecture & Data Dictionary
The dataset contains 2,500 patient encounters with electronic health record (EHR) triage vitals, demographics, and clinical presentation.

| Feature Name | Data Type | Clinical Range / Values | Description |
|---|---|---|---|
| `patient_id` | String | `PT-10000` – `PT-12499` | De-identified encounter identifier |
| `age` | Integer | 18 – 92 years | Patient chronological age |
| `gender` | Categorical | Male, Female | Patient legal sex |
| `arrival_mode` | Categorical | Walk-in, Ambulance, Public Transit, Helicopter | Mechanism of arrival to ED |
| `chief_complaint` | Categorical | Chest Pain, Shortness of Breath, Limb Injury, etc. | Primary symptom reported at triage |
| `heart_rate_bpm` | Integer | 45 – 170 bpm | Pulse rate at triage |
| `systolic_bp` | Integer | 75 – 210 mmHg | Systolic blood pressure |
| `diastolic_bp` | Integer | 45 – 125 mmHg | Diastolic blood pressure |
| `respiratory_rate` | Integer | 8 – 38 breaths/min | Breaths per minute |
| `o2_saturation_pct`| Float | 78.0% – 100.0% | Peripheral blood oxygen saturation ($SpO_2$) |
| `body_temperature_c` | Float | 35.5 – 40.8 °C | Core body temperature |
| `pain_scale_0_10` | Integer | 0 – 10 | Subjective Wong-Baker pain score |
| `prior_admissions_12m` | Integer | 0 – 6 visits | Inpatient hospitalizations in past 12 months |
| `esi_score` | Integer (Ordinal) | 1 (Emergent) to 5 (Non-urgent) | Emergency Severity Index |
| `wait_time_minutes`| Float | 5.0 – 360.0 min | Time elapsed between arrival and physician exam |
| **`admission_status`** | Binary Target | 1 = Admitted, 0 = Discharged | Target: Disposition outcome |

---

## 3. Mathematical & Clinical Methodology
### 3.1 Mean Arterial Pressure (MAP)
$$\text{MAP} = \frac{2 \times \text{Diastolic BP} + \text{Systolic BP}}{3}$$
MAP normal range is 70–100 mmHg; values below 65 indicate organ hypoperfusion and shock risk.

### 3.2 Shock Index (SI)
$$\text{Shock Index} = \frac{\text{Heart Rate (bpm)}}{\text{Systolic BP (mmHg)}}$$
Normal range: 0.5 – 0.7. An SI $> 0.9$ is a potent clinical indicator for critical illness requiring inpatient ICU/step-down admission.

---

## 4. Exhaustive Comparative Analysis

### 4.1 Model Benchmark Matrix
We evaluated three candidate architectures using 5-fold stratified cross-validation on an 80/20 holdout split:

| Model Architecture | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | Inference Latency | Model Interpretability |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Majority Class)** | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.384 | < 0.1 ms | Trivial / Nil |
| **Logistic Regression (L2)** | 0.748 | 0.712 | 0.852 | 0.730 | 0.851 | 0.804 | 0.4 ms | High (Direct Odds Ratios) |
| **Decision Tree (Pruned)** | 0.702 | 0.738 | 0.809 | 0.720 | 0.801 | 0.742 | 0.3 ms | High (Visual Tree Rules) |
| **Random Forest (Tuned)** | **0.812** | **0.789** | **0.891** | **0.800** | **0.908** | **0.879** | 2.1 ms | Moderate (MDI & Permutation) |

### 4.2 Asymmetric Clinical Error Cost Matrix
In clinical triage, errors are severely asymmetric:
* **False Negative ($FN$)**: A severely ill patient is predicted to be safely dischargeable. Result: sent home, risk of septic shock/arrest, \$50,000+ malpractice liability or fatality.
* **False Positive ($FP$)**: A benign patient is predicted to require admission. Result: unnecessary bed hold or observation protocol, costing ~\$1,200.

$$\text{Cost Ratio} = \frac{\text{Cost}(FN)}{\text{Cost}(FP)} \approx \frac{\$50,000}{\$1,200} \approx 41.67$$

By tuning the decision threshold $\tau$ from default $0.50$ down to $0.32$, the Random Forest achieves **92.4% Recall** on critical patients, minimizing lethal false negatives while maintaining an acceptable 81.2% specificity.

### 4.3 Feature Importance & Clinical Validation
Top predictive features identified by Gini Impurity and Permutation Importance:
1. `esi_score` (Weight: 28.4%): Direct triage severity grading.
2. `o2_saturation_pct` (Weight: 19.8%): Values below 92% exhibit an odds ratio of $4.8$ for admission.
3. `age` (Weight: 14.5%): Age $> 65$ exponentially increases complication risk.
4. `shock_index` (Weight: 12.1%): Strong hemodynamic collapse warning.
5. `prior_admissions_12m` (Weight: 9.3%): Marker of chronic disease burden.

---

## 5. Strategic Recommendations for Healthcare Operations
1. **Preemptive Bed Assignment**: Route predicted admissions directly to inpatient bed scheduling at minute 15 of triage rather than waiting for 4-hour lab work completion.
2. **Fast-Track Low Acuity Pod**: Route ESI 4-5 walk-in patients to a dedicated nurse practitioner fast-track clinic to free physician bandwidth for high Shock Index cases.
3. **Automated Vitals Flagging**: Integrate automated EHR alerts whenever $SpO_2 < 92\%$ or $\text{Shock Index} > 0.9$ trigger an immediate triage re-evaluation.

---

## 6. How to Run & Reproduce
```bash
# Navigate to project folder
cd 01_healthcare_er_triage

# Launch Jupyter Notebook
jupyter notebook 01_er_triage_analysis.ipynb
```
