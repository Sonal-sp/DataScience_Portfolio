# ⚙️ Project 13: Industrial Equipment Remaining Useful Life (RUL) & Predictive Maintenance

## 1. Executive Summary & Business Impact
Catastrophic failure of industrial rotating equipment causes massive downtime, safety hazards, and unscheduled maintenance costs. 

This project trains an **RUL regression engine** on multi-sensor degradation telemetry, penalizing late predictions harsher than early alerts via the **NASA Asymmetric Scoring Function**.

---

## 2. Comparative Analysis: RUL Model Benchmark

| Architecture | RMSE (Cycles) | MAE (Cycles) | NASA Asymmetric Penalty |
|---|---|---|---|
| **Linear Regression Baseline** | 24.8 | 19.5 | 1,480 |
| **Random Forest Regressor** | 16.2 | 12.4 | 640 |
| **Gradient Boosted Degradation (Champion)** | **12.1** | **9.2** | **310** |

---

## 3. Implementation Guide
```bash
cd 13_iot_predictive_maintenance_rul
jupyter notebook 13_predictive_maintenance_rul.ipynb
```
