# ⚡ Project 04: EV "Charging Deserts" & Spatial Infrastructure Analytics

## 1. Executive Summary & Business Impact
The global transition toward zero-emission electric mobility hinges directly on the accessibility of public charging infrastructure. In many urban centers, charging hubs are hyper-concentrated in affluent downtown cores, leaving commuter corridors and dense residential districts as **"charging deserts."**

This project applies **unsupervised spatial clustering (DBSCAN & K-Means)** and **gap-ratio analytics** to identify underserved zones with high EV adoption and traffic density, guiding public and private capital deployment for maximum charger utilization and social equity.

---

## 2. Mathematical Foundations
### 2.1 Charger Supply-to-Demand Deficit Index ($CDI$)
$$CDI_i = \frac{\text{Registered EVs}_i + 0.05 \times \text{Daily Traffic}_i}{\text{Existing Chargers}_i + 1}$$
High $CDI$ indicates severe infrastructure drought requiring urgent capital deployment.

### 2.2 Spatial Clustering: DBSCAN with Haversine Metric
DBSCAN clusters core zones within neighborhood radius $\varepsilon$ having at least $\text{MinPts}$ neighbors. Outlying zones labeled as noise ($-1$) with high EV adoption represent acute charging deserts.

---

## 3. Exhaustive Comparative Analysis

| Spatial Method | Cluster Detection Type | Noise / Outlier Isolation | Parameter Tuning Sensitivity | Business Utility |
|---|---|---|---|---|
| **Simple Grid Density** | Fixed spatial bins | None (Averages out pockets) | Low | Broad heatmaps, misses edge borders |
| **K-Means Clustering** | Spherical convex centroids | None (Forces every zone to cluster) | Moderate ($k$ selection) | Balanced regional servicing hubs |
| **DBSCAN (Champion)** | Arbitrary-shaped spatial corridors | **High (Directly flags isolated deserts)** | High ($\varepsilon$, MinPts) | Precise deployment prioritization |

---

## 4. Implementation Guide
```bash
cd 04_energy_ev_charging_deserts
jupyter notebook 04_ev_charging_deserts.ipynb
```
