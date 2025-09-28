"""
E-commerce Product Monthly Sales Predictor
- Load Excel/CSV (handles multiple sheets in Excel)
- Clean & preprocess (remove cancellations, compute revenue)
- Aggregate to monthly revenue per product
- Create lag features (lag-1, lag-3, rolling mean)
- Train RandomForestRegressor, evaluate (RMSE, MAE)
- Save model (joblib)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from math import sqrt

# ---------------------- CONFIG ----------------------

DATA_PATH = r"C:\Users\Sakshi Bhutekar\OneDrive\Desktop\Python\retail.xlsx"


MODEL_OUTPUT_PATH = "rf_sales_model.joblib"

# ---------------------- HELPERS ----------------------
def load_data(path):
    """Load Excel (all sheets concat) or CSV. Returns dataframe."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        xls = pd.read_excel(path, sheet_name=None)  
        if isinstance(xls, dict):
            df = pd.concat(xls.values(), ignore_index=True)
        else:
            df = xls
    elif ext == ".csv":

        df = pd.read_csv(path, encoding="latin1")
    else:
        raise ValueError("Unsupported file type. Provide .xlsx or .csv")
    return df

def find_col(df, candidates):
    """Find first matching column name in df from list of candidate substrings."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_l = cand.lower()
        
        if cand_l in cols_lower:
            return cols_lower[cand_l]
    
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return None

# ---------------------- CLEANING & PREP ----------------------
def clean_and_standardize(df):
    """Detect relevant columns, clean, compute Revenue, parse dates."""
    
    invoice_col = find_col(df, ["Invoice", "InvoiceNo", "InvoiceNo.", "Invoice Number"])
    stock_col = find_col(df, ["StockCode", "Stock Code", "Stock"])
    qty_col = find_col(df, ["Quantity", "Qty"])
    unitprice_col = find_col(df, ["UnitPrice", "Unit Price", "Price", "unit_price"])
    date_col = find_col(df, ["InvoiceDate", "Invoice Date", "Date", "Invoice_Date"])
    desc_col = find_col(df, ["Description", "ProductDescription", "ItemDescription"])
    cust_col = find_col(df, ["Customer ID", "CustomerID", "Customer Id", "Customer"])

   
    required = {"invoice": invoice_col, "stock": stock_col, "quantity": qty_col, "unitprice": unitprice_col, "date": date_col}
    missing = [k for k,v in required.items() if v is None]
    if missing:
        raise ValueError(f"Dataset missing required columns (candidates not found): {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df = df[~df[invoice_col].astype(str).str.startswith(('C','c'))]

    df = df.dropna(subset=[stock_col, qty_col, unitprice_col, date_col])

    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0).astype(int)
    df[unitprice_col] = pd.to_numeric(df[unitprice_col], errors='coerce').fillna(0.0)

    df['Revenue'] = df[qty_col] * df[unitprice_col]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])  

    df = df.rename(columns={
        invoice_col: 'Invoice',
        stock_col: 'ProductID',
        qty_col: 'Quantity',
        unitprice_col: 'UnitPrice',
        date_col: 'InvoiceDate'
    })

    df = df[df['Quantity'] > 0]

    return df

# ---------------------- AGGREGATE & FEATURES ----------------------
def aggregate_monthly(df):
    """Aggregate to monthly revenue per product and include avg unit price per month."""
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    
    monthly = df.groupby(['YearMonth', 'ProductID'], as_index=False).agg(
        MonthlyRevenue=('Revenue', 'sum'),
        AvgUnitPrice=('UnitPrice', 'mean'),
    )
 
    monthly['year'] = monthly['YearMonth'].dt.year
    monthly['month'] = monthly['YearMonth'].dt.month
    
    monthly = monthly.sort_values(['ProductID', 'year', 'month']).reset_index(drop=True)
    return monthly

def create_lag_features(monthly_df, lags=[1,2,3], rolling_windows=[3]):
    """For each product, create lag_k and rolling mean features."""
    df = monthly_df.copy()
    lag_cols = []
    for lag in lags:
        col = f'lag_{lag}'
        df[col] = df.groupby('ProductID')['MonthlyRevenue'].shift(lag)
        lag_cols.append(col)

    for w in rolling_windows:
        col = f'rolling_mean_{w}'
        df[col] = df.groupby('ProductID')['MonthlyRevenue'].shift(1).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
    
    df = df.dropna(subset=['lag_1']).reset_index(drop=True)
    return df

# ---------------------- MODELING ----------------------
def prepare_training_data(df_with_lags):
    """Select features and target. Return X,y and feature names."""
    feature_cols = ['month', 'year', 'AvgUnitPrice', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
    feature_cols = [c for c in feature_cols if c in df_with_lags.columns]
    X = df_with_lags[feature_cols].fillna(0)
    y = df_with_lags['MonthlyRevenue']
    return X, y, feature_cols

def train_and_evaluate(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    return model, (rmse, mae, r2), X_test, y_test, y_pred

# ---------------------- UTILS ----------------------
def plot_actual_vs_pred(y_test, y_pred, title="Actual vs Predicted (sample)"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel("Actual Monthly Revenue")
    plt.ylabel("Predicted Monthly Revenue")
    plt.title(title)
    lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.tight_layout()
    plt.show()

def top_products_plot(monthly_df, top_n=10):
    top = monthly_df.groupby('ProductID')['MonthlyRevenue'].sum().nlargest(top_n).reset_index()
    plt.figure(figsize=(8,4))
    plt.bar(top['ProductID'].astype(str), top['MonthlyRevenue'])
    plt.xticks(rotation=45)
    plt.title(f"Top {top_n} Products by Total Revenue")
    plt.tight_layout()
    plt.show()

# ---------------------- MAIN FLOW ----------------------
def main(data_path):
    print("Loading data...")
    df_raw = load_data(data_path)
    print("Raw shape:", df_raw.shape)
    print("Cleaning & standardizing...")
    df = clean_and_standardize(df_raw)
    print("After cleaning shape:", df.shape)

    print("Aggregating monthly revenue per product...")
    monthly = aggregate_monthly(df)
    print("Monthly rows:", len(monthly))

    print("Creating lag features...")
    monthly_lags = create_lag_features(monthly, lags=[1,2,3], rolling_windows=[3])
    print("After lags shape:", monthly_lags.shape)

    print("Preparing training data...")
    X, y, feat_names = prepare_training_data(monthly_lags)
    print("Features used:", feat_names)
    print("Training model...")
    model, metrics, X_test, y_test, y_pred = train_and_evaluate(X, y)
    rmse, mae, r2 = metrics
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

    print("Saving model to", MODEL_OUTPUT_PATH)
    joblib.dump({'model': model, 'features': feat_names}, MODEL_OUTPUT_PATH)

    print("Plotting results...")
    plot_actual_vs_pred(y_test, y_pred)
    top_products_plot(monthly)

    print("Done! Model saved. To predict next month, create a row with the same features and call model.predict([row])")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("ERROR: data file not found at:", DATA_PATH)
        print("Please update DATA_PATH at the top of the script to the correct file path.")
    else:
        main(DATA_PATH)
