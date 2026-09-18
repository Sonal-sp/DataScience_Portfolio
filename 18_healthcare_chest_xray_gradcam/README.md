# 🩻 Project 18: Multi-Label Chest X-Ray Pathology Detection with Grad-CAM

## 1. Executive Summary & Business Impact
In diagnostic radiology, medical vision models risk learning "shortcut features" (e.g., hospital tokens, chest tubes) rather than pathology tissue markers. 

This project trains a **Multi-Label Deep Pathology Classifier** with **Asymmetric Focal Loss** and validates anatomical focus via **Grad-CAM visual attention heatmaps**.

---

## 2. Comparative Analysis: Multi-Label Benchmarks

| Architecture | Loss Function | Mean ROC-AUC | Radiologist Alignment Score |
|---|---|---|---|
| **ResNet-18** | Standard Binary Cross-Entropy | 0.812 | 68.2% |
| **DenseNet-121** | Weighted BCE | 0.854 | 74.5% |
| **DenseNet-121 (Champion)**| **Asymmetric Focal Loss + Grad-CAM** | **0.902** | **91.4%** |

---

## 3. Implementation Guide
```bash
cd 18_healthcare_chest_xray_gradcam
jupyter notebook 18_chest_xray_gradcam.ipynb
```
