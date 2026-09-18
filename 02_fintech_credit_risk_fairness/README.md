# 💳 Project 02: Consumer Credit Risk Scoring & Algorithmic Fairness Audit

## 1. Executive Summary & Business Impact
In credit risk underwriting, machine learning algorithms automate millions of loan approval decisions daily. However, deploying unchecked models poses massive compliance and financial risks:
1. **Underwriting Losses**: Approving bad borrowers results in severe credit write-offs ($FN$).
2. **Opportunity Losses**: Denying creditworthy applicants costs lucrative interest income ($FP$).
3. **Regulatory Fines & Disparate Impact**: Violations of the US Equal Credit Opportunity Act (ECOA) or the EU AI Act can incur nine-figure regulatory fines and reputational ruin if models systematically discriminate against protected classes (e.g., gender or younger demographics).

This project creates a **high-performing credit scoring engine** coupled with an **algorithmic fairness audit** that evaluates and mitigates disparate impact while preserving underwriting profitability.

---

## 2. Dataset Architecture & Data Dictionary
The dataset contains 3,000 personal loan applications with demographic, financial, and credit history variables.

| Column | Type | Range / Values | Description |
|---|---|---|---|
| `applicant_id` | String | `APP-20000` – `APP-22999` | De-identified applicant identifier |
| `age` | Integer | 21 – 72 years | Protected attribute: Applicant age |
| `gender` | Categorical | Male, Female | Protected attribute: Applicant gender |
| `annual_income` | Integer | $18,000 – $250,000 | Verified gross annual earnings |
| `credit_score` | Integer | 450 – 850 | FICO credit score |
| `debt_to_income_ratio` | Float | 0.05 – 0.65 | Total monthly debt obligations / Gross income |
| `employment_years` | Float | 0.0 – 35.0 years | Tenure at current employer |
| `open_credit_lines` | Integer | 0 – 15 | Active lines of credit |
| `delinquencies_2yr` | Integer | 0 – 3 | 30+ days past due incidents in past 2 years |
| `loan_amount` | Integer | $2,000 – $45,000 | Requested principal loan amount |
| `loan_purpose` | Categorical | Debt Consolidation, Home Impr, etc. | Stated loan objective |
| **`default_status`** | Binary Target | 1 = Defaulted, 0 = Fully Paid | Target label (Default incidence) |

---

## 3. Mathematical Foundations of Algorithmic Fairness
Under the US EEOC Four-Fifths Rule, a selection rate for any protected group that is less than 80% ($4/5$) of the highest selection group is evidence of **Disparate Impact**.

### 3.1 Disparate Impact Ratio (DIR)
$$\text{DIR} = \frac{P(\hat{Y}=0 \mid A=\text{Protected})}{P(\hat{Y}=0 \mid A=\text{Privileged})}$$
*(Where $\hat{Y}=0$ corresponds to loan approval).* Target: $\text{DIR} \ge 0.80$.

### 3.2 Equal Opportunity Difference (EOD)
$$\text{EOD} = |P(\hat{Y}=0 \mid Y=0, A=\text{Protected}) - P(\hat{Y}=0 \mid Y=0, A=\text{Privileged})|$$
Measures whether creditworthy individuals have equal chances of approval regardless of protected group membership. Target: $\text{EOD} < 0.05$.

---

## 4. Exhaustive Comparative Analysis

### 4.1 Model Benchmark & Fairness Tradeoff Matrix

| Model Architecture | ROC-AUC | PR-AUC | F1-Score | Default KS-Stat | Disparate Impact (Gender) | Equal Opp. Diff (EOD) | Regulatory Status |
|---|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.792 | 0.612 | 0.618 | 44.2% | 0.742 *(Violation)* | 0.098 | ❌ Non-Compliant |
| **Standard Random Forest** | **0.865** | **0.724** | **0.710** | **53.8%** | 0.761 *(Violation)* | 0.084 | ❌ Non-Compliant |
| **Fairness-Calibrated Ensemble** | 0.854 | 0.709 | 0.698 | 51.9% | **0.895** | **0.031** | ✅ Fully Compliant |

### 4.2 Financial Error Cost Matrix
* **Cost of False Negative ($FN$)**: Approving a borrower who defaults. Average charge-off loss: **$12,000**.
* **Cost of False Positive ($FP$)**: Denying a borrower who would have paid. Foregone net interest margin: **$2,500**.
* **Cost Ratio**: $FN : FP = 4.8 : 1$.

*Takeaway*: The Fairness-Calibrated model incurs a minor 1.1% dip in ROC-AUC but elevates Disparate Impact from 0.76 to 0.90, shielding the financial institution from millions in regulatory exposure while maintaining 98.7% of maximum portfolio profitability.

---

## 5. Implementation & Reproduction
```bash
cd 02_fintech_credit_risk_fairness
jupyter notebook 02_credit_risk_fairness.ipynb
```
