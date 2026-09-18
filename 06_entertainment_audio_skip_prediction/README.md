# 🎧 Project 06: Audio Feature Analysis & Song Skip Propensity

## 1. Executive Summary & Business Impact
User retention on digital audio platforms (Spotify, Apple Music) is governed by early listening satisfaction. When recommendation engines serve tracks that are skipped within the initial 30 seconds, session abandonment increases exponentially. 

This project formulates an **early song-skip prediction model** based on Spotify acoustic attributes and listening contexts to dynamically filter out high-probability skip tracks before they reach the user queue.

---

## 2. Comparative Analysis: Imbalance & Classification Benchmark

| Model Architecture | Handling Strategy | PR-AUC | ROC-AUC | F1-Score | Brier Score |
|---|---|---|---|---|---|
| **Majority Baseline** | None | 0.334 | 0.500 | 0.000 | 0.222 |
| **Logistic Regression** | Standard | 0.612 | 0.741 | 0.584 | 0.168 |
| **Random Forest** | Class Weights Balanced | 0.735 | 0.842 | 0.710 | 0.124 |
| **Random Forest + SMOTE** | Synthetic Oversampling | **0.751** | **0.856** | **0.728** | **0.118** |

---

## 3. Implementation Guide
```bash
cd 06_entertainment_audio_skip_prediction
jupyter notebook 06_audio_skip_prediction.ipynb
```
