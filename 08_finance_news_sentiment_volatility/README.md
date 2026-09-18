# 📈 Project 08: Financial News Sentiment & Market Volatility Forecasting

## 1. Executive Summary & Business Impact
Asset managers and market makers rely heavily on volatility forecasts for option pricing, portfolio risk budgeting (Value-at-Risk), and margin requirements. While traditional econometric models (GARCH) model volatility strictly through autoregressive returns, unexpected market shocks are almost always catalyzed by breaking macroeconomic and company news.

This project implements a **Hybrid NLP & Time-Series Volatility Engine** fusing FinBERT domain sentiment signals with historical realized volatility to predict next-day market volatility spikes.

---

## 2. Comparative Analysis: Model Benchmarks

| Model Architecture | Inputs | Test RMSE | MAE | Volatility Regime Recall (>0.25) |
|---|---|---|---|---|
| **Historical Moving Average (20-day)** | Price History Only | 0.0412 | 0.0321 | 58.2% |
| **GARCH(1,1) Econometric Baseline** | Price Returns Only | 0.0354 | 0.0279 | 71.4% |
| **Sentiment-Augmented LightGBM (Champion)**| **Price + FinBERT Polarity + News Volume** | **0.0248** | **0.0192** | **88.6%** |

---

## 3. Implementation Guide
```bash
cd 08_finance_news_sentiment_volatility
jupyter notebook 08_news_sentiment_volatility.ipynb
```
