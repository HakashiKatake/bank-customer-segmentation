# Bank Customer Segmentation & Regional Revenue Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)](https://plotly.com)

> **AI-powered customer intelligence for banking** — cluster customers by behavior and forecast regional transaction revenue using K-Means and Linear Regression.

---

## Business Problem Statement

Modern banks serve millions of customers with highly varied financial behaviors. A one-size-fits-all approach leads to:
- Revenue leakage from mismatched products
- High churn in under-served segments
- Inefficient capital allocation across regions

This project solves two critical business problems:
1. **Who are our customers?** — Identify distinct behavioral clusters using K-Means
2. **Where is revenue growing?** — Forecast regional transaction volume using Linear Regression

Google Collab link: https://colab.research.google.com/drive/1Kh7Jior64Y8o1eT6lE-At2MSWFzzlni7?usp=sharing

---

## Economic Concepts Applied

| Concept | Application |
|---------|------------|
| **Price Discrimination** | Tailor fees & interest rates per customer segment |
| **Demand-Supply Analysis** | Match banking service supply to forecasted regional demand |
| **Customer Lifetime Value (CLV)** | Prioritize retention of high-CLV Premium Savers |
| **Risk Analysis (Credit Risk)** | Use cluster credit profiles to adjust lending terms |
| **Revenue Optimization** | Direct marketing spend to high-growth West & South regions |
| **Market Segmentation** | Segment-specific product design aligned to customer needs |

---

## AI Techniques Used

| Technique | Purpose |
|-----------|---------|
| **K-Means Clustering** | Unsupervised customer segmentation (K=4 clusters) |
| **Elbow Method + Silhouette Score** | Optimal cluster count selection |
| **PCA (2D)** | Dimensionality reduction for cluster visualization |
| **Linear Regression** | Regional transaction volume forecasting |
| **Feature Engineering** | BalanceToIncome, LoanToBalance, AvgTransactionAmt ratios |
| **StandardScaler** | Feature normalization for both models |

---

## Customer Segments Discovered

| Segment | Profile | Strategy |
|---------|---------|---------|
| Premium Savers | High balance, high credit, long tenure | Wealth management, VIP retention |
| Growth Investors | Mid-life, asset growth, multi-product | Investment products, mortgages |
| Active Spenders | High frequency, credit cards, young | Cashback rewards, credit limits |
| Entry-Level | Low balance, new customers, digital-native | Digital onboarding, savings education |

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset Size | 1M rows (Kaggle) |
| K-Means K | 4 clusters |
| Silhouette Score | ~0.25 |
| Linear Regression R² | ~0.85+ |
| Highest Growth Region | West (+20%), South (+15%) |
| Best Revenue Cluster | Premium Savers |

---

## Project Structure

```
bank-customer-segmentation/
├── bank_customer_segmentation.ipynb   # Google Colab notebook (full analysis)
├── app.py                             # Streamlit interactive dashboard
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## Dataset

[Kaggle: Massive Bank Dataset (1 Million Rows)](https://www.kaggle.com/datasets/ksabishek/massive-bank-dataset-1-million-rows)
The local `bankdataset.csv` file, originally converted from the "Massive Bank Dataset" on Kaggle, is used for this project.


**Key Features:**
- CustomerID, Age, Gender, Region
- Account Balance, Transaction Volume, Number of Transactions
- Credit Score, Loan Amount, Monthly Income
- Tenure, Number of Products, Satisfaction Score

---

## Quick Start

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
Upload `bank_customer_segmentation.ipynb` to [Google Colab](https://colab.research.google.com/drive/1Kh7Jior64Y8o1eT6lE-At2MSWFzzlni7?usp=sharing)

---

## Deployment

The app is deployed on **Streamlit Cloud**:
> [Live Demo](#) ← *(Add your Streamlit Cloud URL here after deployment)*

### Deploy Your Own

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click **Deploy**

---

## Screenshots

### Dashboard Overview
<img width="1470" height="652" alt="Screenshot 2026-02-26 at 11 06 14 PM" src="https://github.com/user-attachments/assets/86dfda99-8e73-4e12-be40-f40c24d119c9" />


### Monthly Revenue Trend
<img width="970" height="426" alt="Screenshot 2026-02-26 at 11 07 45 PM" src="https://github.com/user-attachments/assets/54b40502-0ac3-42a1-b24d-9e7a6764c29a" />



### Top 10 Locations by Processing Volume
<img width="1067" height="505" alt="Screenshot 2026-02-26 at 11 08 05 PM" src="https://github.com/user-attachments/assets/bd6a5ee8-0a32-48a8-833a-f66e3b9ded49" />

### Domain Distribution
<img width="1040" height="480" alt="Screenshot 2026-02-26 at 11 08 56 PM" src="https://github.com/user-attachments/assets/8b39ffee-7c28-472a-81ed-62ad3bef7958" />

### Clusters using PCA
<img width="1052" height="473" alt="Screenshot 2026-02-26 at 11 10 10 PM" src="https://github.com/user-attachments/assets/467ab7eb-6c95-46d5-9b46-8fdeda81a9af" />

### Actual vs Predicted
<img width="813" height="461" alt="Screenshot 2026-02-26 at 11 10 48 PM" src="https://github.com/user-attachments/assets/d2002cf4-0970-4ed0-a9b7-7da25093f86b" />


---

## Notebook Contents

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

## Team

- Saurabh Yadav
- Sakshi Shingole
- Manthan Bhosale
- Antriksh Sandesh Manwadkar

---

## License

MIT License: free to use, modify, and distribute.
