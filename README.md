# 🛒 E-commerce Sales Predictor

This project predicts **monthly sales per product** using **time-series features** and a **Random Forest Regressor**.  
It cleans sales data, aggregates it by product and month, creates lag features, and evaluates predictive performance.

---

## 🚀 Features
- Load Excel/CSV files (supports multiple sheets in Excel).
- Clean data (remove cancellations, compute revenue).
- Aggregate to monthly revenue per product.
- Create lag & rolling mean features.
- Train and evaluate a `RandomForestRegressor`.
- Save trained model with `joblib`.

---


