import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Pages internes (sections) ---
from pages_internal import metrics as metrics_page
from pages_internal import strategy as strategy_page
from pages_internal import risk as risk_page
from pages_internal import predict as predict_page

st.set_page_config(page_title="Mini Dashboard", layout="wide")
st.title("📊 Mini Dashboard ETF")

# ----------------------------
# --- MENU HORIZONTAL AVEC BOUTONS ---
# ----------------------------
buttons = ["Dashboard", "Metrics", "Strategy", "Risk", "Predict"]
cols = st.columns(len(buttons))

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

# Gestion des clics sur boutons
for i, name in enumerate(buttons):
    if cols[i].button(name):
        st.session_state["page"] = name

page = st.session_state["page"]

# ----------------------------
# --- PAGE DASHBOARD ---
# ----------------------------
if page == "Dashboard":

    # --- Sidebar: Inputs ---
    tickers_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=180))
    end_date = st.date_input("End Date", datetime.now())

    # --- Data Load ---
    @st.cache_data(ttl=3600)
    def load_data(tickers, start, end):
        price_map = {}
        for t in tickers:
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    df = yf.download(t, start=start, end=end, progress=False)
                if not df.empty:
                    price_map[t] = df['Close']
            except:
                continue
        if len(price_map) >= 2:
            df_final = pd.concat(price_map.values(), axis=1)
            df_final.columns = ["Risk", "Safe"]
            df_final = df_final.ffill().dropna()
            return df_final
        return pd.DataFrame()

    df = load_data(tickers, start_date, end_date)

    if df.empty:
        st.error("❌ No data found. Check tickers or date range.")
    else:
        st.write("### Raw Prices")
        st.line_chart(df)

        # --- Performance Normalized ---
        perf = df / df.iloc[0] * 100
        st.write("### Normalized Performance (Base 100)")
        st.line_chart(perf)

        # --- Metrics ---
        def calc_metrics(series):
            total_ret = (series.iloc[-1] / series.iloc[0]) - 1
            days = len(series)
            cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 1 else 0
            drawdown = (series / series.cummax() - 1).min()
            return total_ret, cagr, drawdown

        metrics = {col: calc_metrics(df[col]) for col in df.columns}

        st.write("### Metrics")
        metrics_df = pd.DataFrame(metrics, index=["Total Return", "CAGR", "Max Drawdown"]).T
        st.table(metrics_df)

# ----------------------------
# --- PAGE METRICS ---
# ----------------------------
elif page == "Metrics":
    metrics_page.render()

# ----------------------------
# --- PAGE STRATEGY ---
# ----------------------------
elif page == "Strategy":
    strategy_page.render()

# ----------------------------
# --- PAGE RISK ---
# ----------------------------
elif page == "Risk":
    risk_page.render()

# ----------------------------
# --- PAGE PREDICT ---
# ----------------------------
elif page == "Predict":
    predict_page.render()
