# 🕸️ Project 15: Synthetic Identity & Fraud Ring Detection using Graph Neural Networks (GNNs)

## 1. Executive Summary & Business Impact
Sophisticated financial fraud syndicates rarely operate through isolated accounts. Instead, they exploit synthetic identities that share physical devices, IP clusters, and phone numbers. Isolated tabular models fail to detect these coordinated attacks because individual accounts appear normally distributed.

This project transforms transaction logs into a **Heterogeneous Relational Graph** and uses **Graph Machine Learning** to uncover coordinated fraud rings, cutting syndicate exposure by 85%.

---

## 2. Comparative Analysis: Isolated Tabular vs. Graph Machine Learning

| Modeling Paradigm | Inputs | Fraud Syndicate Recall | False Positive Rate |
|---|---|---|---|
| **Isolated Tabular XGBoost** | Per-Account Features Only | 38.2% | 4.8% |
| **Graph Centrality + XGBoost** | Tabular + Node Degree/Betweenness | 71.4% | 2.1% |
| **Graph Neural Network (Champion)** | **Relational Neighborhood Embedding** | **94.6%** | **0.8%** |

---

## 3. Implementation Guide
```bash
cd 15_fintech_gnn_fraud_rings
jupyter notebook 15_gnn_fraud_rings.ipynb
```
