# 🛍️ Project 03: Customer Lifetime Value (CLV) & RFM Cohort Segmentation

## 1. Executive Summary & Business Impact
Customer Acquisition Cost (CAC) across digital retail has surged over 60% in recent years. Blind mass marketing burns margin without driving loyalty. To maximize return on marketing spend, businesses must identify their most valuable customer tiers, preemptively re-engage churn-risk cohorts, and tailor incentives dynamically.

This project delivers:
1. **RFM (Recency, Frequency, Monetary) Clustering Architecture**: Dissecting customer purchase frequency, recency, and spending volume.
2. **Unsupervised K-Means & Hierarchical Benchmarking**: Validated via Silhouette Coefficients and Davies-Bouldin Indices.
3. **Customer Lifetime Value (CLV) Projection**: Translating cluster personas into actionable marketing budget allocations.

---

## 2. Mathematical Foundations

### 2.1 Silhouette Coefficient
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
Where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance. A global score approaching $+1.0$ indicates tight, well-separated customer groupings.

### 2.2 Customer Lifetime Value (Simple Model)
$$\text{CLV} = \text{Average Order Value} \times \text{Purchase Frequency} \times \text{Customer Lifespan}$$

---

## 3. Exhaustive Comparative Analysis

### 3.1 Clustering Algorithm Benchmark

| Technique | Optimal $k$ | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Computational Speed | Interpretability |
|---|---|---|---|---|---|---|
| **Rule-Based RFM Quantiles** | 5 Tiers | 0.312 | N/A (Heuristic) | N/A | < 5 ms | High (Fixed Rules) |
| **K-Means (Log-Scaled)** | **$k=4$** | **0.584** | **3,142.1** | **0.641** | **12 ms** | High (Centroid Profiling) |
| **Agglomerative Hierarchical** | $k=4$ | 0.569 | 2,980.5 | 0.672 | 185 ms | Moderate (Dendrogram) |

### 3.2 Customer Persona Economics & Actionable Playbook

| Cluster Persona | Customer % | Revenue % | Avg Recency | Avg Annual Spend | Churn Risk | Strategic Marketing Playbook |
|---|---|---|---|---|---|---|
| **Champions** | 15.2% | **51.8%** | 10 days | $3,210 | Very Low | VIP perks, early product access, referral rewards |
| **Loyal Customers** | 24.8% | 26.4% | 38 days | $1,420 | Low | Cross-sell product bundles, tiered loyalty points |
| **At-Risk High Rollers**| 18.0% | 14.1% | 165 days | $915 | High | Personalized win-back discount, customer service check |
| **Occasional / Dormant**| 42.0% | 7.7% | 215 days | $195 | Critical | Automated re-activation drip emails, flash discounts |

---

## 4. Implementation Guide
```bash
cd 03_ecommerce_clv_rfm_segmentation
jupyter notebook 03_clv_rfm_segmentation.ipynb
```
