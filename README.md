# Credit-Card-Fraud-Detection-project
# 🕵️‍♂️ Credit Card Fraud Detection using Machine Learning

A machine learning project to detect fraudulent credit card transactions using classification algorithms and techniques for handling class imbalance.

---

## 📌 Project Overview

Credit card fraud is a growing issue in the financial industry. This project aims to detect fraudulent transactions from a large set of anonymized credit card transaction data using machine learning techniques. Given the severe class imbalance, special attention is paid to sampling strategies and evaluation metrics.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Features**: 30 (V1 to V28 are PCA components + `Time`, `Amount`)
- **Target**: `Class` (0 = Legitimate, 1 = Fraudulent)

---

## 🧰 Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost
- SMOTE (Imbalanced-learn)

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
