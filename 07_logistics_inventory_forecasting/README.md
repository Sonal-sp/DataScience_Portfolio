# 📦 Project 07: Multi-Store Inventory Demand Forecasting with Stockout Risk

## 1. Executive Summary & Business Impact
In supply chain management, excess inventory ties up critical working capital and creates holding costs, while stockouts result in permanent churn and lost revenue. 

This project formulates an **end-to-end multi-echelon demand forecasting system** using **LightGBM and rolling temporal feature engineering**. It evaluates model forecasts using Weighted Absolute Percentage Error (WAPE) and computes dynamic **Safety Stock** levels to achieve a 98% service level with minimum working capital.

---

## 2. Mathematical Foundations
### 2.1 Weighted Absolute Percentage Error (WAPE)
$$\text{WAPE} = \frac{\sum_{t=1}^T |y_t - \hat{y}_t|}{\sum_{t=1}^T y_t}$$
Unlike MAPE, WAPE is robust to zero-demand days and does not artificially inflate errors on small denominators.

### 2.2 Dynamic Safety Stock ($SS$)
$$SS = Z_{\alpha} \times \sqrt{L \times \sigma_D^2 + D^2 \times \sigma_L^2}$$
Where $Z_{\alpha} = 2.054$ for a 98% cycle service level, $L$ is lead time, $\sigma_D$ is demand standard deviation, and $\sigma_L$ is lead-time variance.

---

## 3. Exhaustive Comparative Analysis

| Model | Validation Split | WAPE | RMSE | Stockout Probability | Training Speed |
|---|---|---|---|---|---|
| **Naive Lag-7 Baseline** | TimeSeriesSplit | 24.8% | 32.4 | 14.2% | < 1 ms |
| **Holt-Winters Exponential Smoothing** | TimeSeriesSplit | 18.9% | 24.1 | 8.6% | 45 ms |
| **LightGBM Regressor (Champion)** | **TimeSeriesSplit (5 folds)** | **11.4%** | **15.8** | **2.1%** | **180 ms** |

---

## 4. Implementation Guide
```bash
cd 07_logistics_inventory_forecasting
jupyter notebook 07_inventory_demand_forecasting.ipynb
```
