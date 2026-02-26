# 🏦 Bank Customer Segmentation & Regional Revenue Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)](https://plotly.com)

> **AI-powered customer intelligence for banking** — cluster customers by behavior and forecast regional transaction revenue using K-Means and Linear Regression.

---

## 📌 Business Problem Statement

Modern banks serve millions of customers with highly varied financial behaviors. A one-size-fits-all approach leads to:
- Revenue leakage from mismatched products
- High churn in under-served segments
- Inefficient capital allocation across regions

This project solves two critical business problems:
1. **Who are our customers?** — Identify distinct behavioral clusters using K-Means
2. **Where is revenue growing?** — Forecast regional transaction volume using Linear Regression

---

## 💡 Economic Concepts Applied

| Concept | Application |
|---------|------------|
| **Price Discrimination** | Tailor fees & interest rates per customer segment |
| **Demand-Supply Analysis** | Match banking service supply to forecasted regional demand |
| **Customer Lifetime Value (CLV)** | Prioritize retention of high-CLV Premium Savers |
| **Risk Analysis (Credit Risk)** | Use cluster credit profiles to adjust lending terms |
| **Revenue Optimization** | Direct marketing spend to high-growth West & South regions |
| **Market Segmentation** | Segment-specific product design aligned to customer needs |

---

## 🤖 AI Techniques Used

| Technique | Purpose |
|-----------|---------|
| **K-Means Clustering** | Unsupervised customer segmentation (K=4 clusters) |
| **Elbow Method + Silhouette Score** | Optimal cluster count selection |
| **PCA (2D)** | Dimensionality reduction for cluster visualization |
| **Linear Regression** | Regional transaction volume forecasting |
| **Feature Engineering** | BalanceToIncome, LoanToBalance, AvgTransactionAmt ratios |
| **StandardScaler** | Feature normalization for both models |

---

## 🎯 Customer Segments Discovered

| Segment | Profile | Strategy |
|---------|---------|---------|
| 💎 Premium Savers | High balance, high credit, long tenure | Wealth management, VIP retention |
| 📈 Growth Investors | Mid-life, asset growth, multi-product | Investment products, mortgages |
| 💳 Active Spenders | High frequency, credit cards, young | Cashback rewards, credit limits |
| 🌱 Entry-Level | Low balance, new customers, digital-native | Digital onboarding, savings education |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 1M rows (Kaggle) |
| K-Means K | 4 clusters |
| Silhouette Score | ~0.25 |
| Linear Regression R² | ~0.85+ |
| Highest Growth Region | West (+20%), South (+15%) |
| Best Revenue Cluster | 💎 Premium Savers |

---

## 🗂️ Project Structure

```
bank-customer-segmentation/
├── bank_customer_segmentation.ipynb   # Google Colab notebook (full analysis)
├── app.py                             # Streamlit interactive dashboard
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🔗 Dataset

[Kaggle: Massive Bank Dataset (1 Million Rows)](https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows)

**Key Features:**
- CustomerID, Age, Gender, Region
- Account Balance, Transaction Volume, Number of Transactions
- Credit Score, Loan Amount, Monthly Income
- Tenure, Number of Products, Satisfaction Score

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/bank-customer-segmentation.git
cd bank-customer-segmentation
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Open Colab Notebook
Upload `bank_customer_segmentation.ipynb` to [Google Colab](https://colab.research.google.com/)

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud**:
> 🔗 [Live Demo](#) ← *(Add your Streamlit Cloud URL here after deployment)*

### Deploy Your Own

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click **Deploy** 🚀

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard.png)

### Customer Clusters (PCA)
![Clusters](screenshots/clusters.png)

### Regional Forecast
![Forecast](screenshots/forecast.png)

### Single Customer Predictor
![Predictor](screenshots/predictor.png)

---

## 📋 Notebook Contents

| Section | Description |
|---------|-------------|
| 1. Data Loading | Load from Kaggle or synthetic generation |
| 2. Data Cleaning | Null imputation, outlier removal (IQR) |
| 3. Feature Engineering | Ratio features, age groups, wealth tiers |
| 4. EDA | Distributions, correlations, regional analysis, time series |
| 5. K-Means Clustering | Elbow, silhouette, final model, radar profiles |
| 6. Linear Regression | Training, evaluation, 2025-2026 forecast |
| 7. Business Interpretation | Economic concepts, strategic recommendations |

---

## 👥 Team

- **[Your Name]** — Data Scientist / ML Engineer

---

## 📄 License

MIT License — free to use, modify, and distribute.
