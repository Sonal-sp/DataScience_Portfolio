# 💬 Project 20: Multi-Class Intent Classifier with Out-Of-Distribution (OOD) Rejection

## 1. Executive Summary & Business Impact
Enterprise customer support bots must route queries accurately across dozens of operational departments while safely rejecting Out-Of-Distribution (OOD) chit-chat or adversarial inputs without hallucinating misdirected tickets.

This project implements a **Transformer Intent Classifier** paired with **Mahalanobis Distance Density Scoring** to achieve calibrated OOD detection.

---

## 2. Comparative Analysis: OOD Detection Benchmarks

| OOD Detection Strategy | In-Domain Macro F1 | AUROC on Unseen OOD Queries | Misrouting Rate |
|---|---|---|---|
| **Max Softmax Probability (MSP)** | 0.912 | 0.724 | 14.8% |
| **Temperature Scaled Softmax (ODIN)** | 0.914 | 0.835 | 8.2% |
| **Mahalanobis Embedding Distance (Champion)** | **0.928** | **0.962** | **1.9%** |

---

## 3. Implementation Guide
```bash
cd 20_nlp_intent_ood_detection
jupyter notebook 20_intent_ood_detection.ipynb
```
