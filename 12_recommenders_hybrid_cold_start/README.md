# 🎬 Project 12: Hybrid Recommendation Engine with Item Cold-Start Fallback

## 1. Executive Summary & Business Impact
Collaborative Filtering excels on mature items with abundant user feedback but fails entirely when new catalog items launch (the **Item Cold-Start Problem**).

This project constructs a **two-tier Hybrid Recommender** using **Truncated SVD Matrix Factorization** for warm collaborative items and seamlessly falls back to **Content-Based Cosine Similarity** on item metadata for zero-history items, maintaining high catalog coverage.

---

## 2. Comparative Analysis: Recommender Benchmarks

| Algorithm Type | Warm Item NDCG@10 | Cold Item Recall@10 | Catalog Coverage |
|---|---|---|---|
| **Popularity Baseline** | 0.412 | 0.000 | 12.5% |
| **Pure SVD Matrix Factorization** | **0.784** | 0.000 (Fails completely) | 68.4% |
| **Hybrid SVD + Content Fallback** | 0.771 | **0.658** | **98.2%** |

---

## 3. Implementation Guide
```bash
cd 12_recommenders_hybrid_cold_start
jupyter notebook 12_hybrid_recommender.ipynb
```
