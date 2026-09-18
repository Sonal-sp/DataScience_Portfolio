# 🛰️ Project 17: Satellite Imagery Crop Health & Land Cover Semantic Segmentation

## 1. Executive Summary & Business Impact
Precision agriculture and climate change adaptation require continuous monitoring of crop vigor and land-use shifts. 

This project processes **Sentinel-2 multispectral satellite imagery bands**, calculates the **Normalized Difference Vegetation Index (NDVI)**, and executes **land cover semantic segmentation** to track crop vitality and drought stress.

---

## 2. Comparative Analysis: Land Cover Segmentation

| Method | Mean IoU | Crop F1-Score | Water F1-Score |
|---|---|---|---|
| **NDVI Thresholding Rule** | 0.612 | 0.742 | 0.812 |
| **Random Forest Pixel Classifier** | 0.845 | 0.892 | 0.941 |
| **Multi-Spectral Ensemble (Champion)** | **0.918** | **0.954** | **0.978** |

---

## 3. Implementation Guide
```bash
cd 17_agritech_satellite_crop_segmentation
jupyter notebook 17_satellite_crop_segmentation.ipynb
```
