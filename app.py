import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
linreg = joblib.load('linreg_expanded.pkl')
gbr    = joblib.load('gbr_tuned.pkl')
scaler = joblib.load('scaler.pkl')

st.title("EUR/USD Next-Day Price Forecast")

uploaded = st.file_uploader("Upload CSV with columns Date, Open, High, Low, Close[,Adj Close]", type="csv")
if uploaded:
    raw = pd.read_csv(uploaded, parse_dates=['Date'])
    df = raw.rename(columns={
        'Date':'date','Open':'open','High':'high','Low':'low','Close':'close',
        **({'Adj Close':'aclose'} if 'Adj Close' in raw.columns else {})
    })
    close = 'aclose' if 'aclose' in df.columns else 'close'
    for col in ['open','high','low','close', close]:
        if col in df: df[col] = 1/df[col]

    # Compute features exactly as in the notebook
    df['return'] = df[close] - df[close].shift(1)
    for i in range(1,61):
        df[f'return_lag_{i}'] = df['return'].shift(i)
    df['vol_14'] = df['return'].rolling(14).std()
    df['ma_30']  = df[close].rolling(30).mean()
    df = df.dropna().reset_index(drop=True)

    feature_cols = [f'return_lag_{i}' for i in range(1,61)] + ['vol_14','ma_30']
    X = df[feature_cols].to_numpy()
    X_scaled = scaler.transform(X)

    last = X_scaled[-1].reshape(1, -1)
    p_lr  = linreg.predict(last)[0]
    p_gbr = gbr.predict(last)[0]

    st.write(f"Linear Regression Forecast:   {p_lr:.4f} EUR/USD")
    st.write(f"Gradient Boosting Forecast:  {p_gbr:.4f} EUR/USD")
