# ⚽ Project 05: Player Market Valuation & Tactical Playstyle Role Clustering

## 1. Executive Summary & Business Impact
In professional sports analytics, traditional nominal positions (e.g., "Midfielder" or "Forward") fail to capture modern tactical nuances (e.g., inverted wingbacks, pressing forwards, deep-lying playmakers). Furthermore, transfer market valuations are frequently inflated by hype, domestic quotas, and media sentiment.

This project delivers:
1. **Unsupervised Playstyle Clustering via PCA**: Discovering true on-pitch tactical roles independent of listed positions.
2. **Regularized Valuation Modeling (OLS vs. Ridge vs. Lasso)**: Predicting objective market transfer values based purely on underlying per-90 metrics.
3. **Moneyball Transfer Arbitrage**: Pinpointing significantly undervalued talent for recruitment scout targeting.

---

## 2. Comparative Analysis: Regression Benchmark

| Model Architecture | Regularization | Test $R^2$ | RMSE (Mil €) | MAE (Mil €) | VIF Collinearity Control |
|---|---|---|---|---|---|
| **Ordinary Least Squares (OLS)** | None | 0.812 | 7.84 | 5.92 | Susceptible to multi-collinearity |
| **Ridge Regression ($L_2$)** | $\alpha=1.0$ | 0.824 | 7.61 | 5.75 | Shrinks correlated coefficients |
| **Lasso Regression ($L_1$)** | $\alpha=0.25$ | **0.829** | **7.51** | **5.66** | **Enforces sparsity / feature selection** |

---

## 3. Implementation Guide
```bash
cd 05_sports_player_valuation_roles
jupyter notebook 05_sports_player_valuation.ipynb
```
