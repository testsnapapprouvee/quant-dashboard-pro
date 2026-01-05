import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Mini Dashboard", layout="wide")

# ----------------------------
# SIDEBAR : NAVIGATION
# ----------------------------
st.sidebar.title("📂 Modules")

page = st.sidebar.radio(
    "Select a module",
    [
        "Dashboard",
        "Metrics",
        "Strategy",
        "Risk",
        "Predict"
    ]
)

# ----------------------------
# PAGE 1 : DASHBOARD (TON CODE)
# ----------------------------
if page == "Dashboard":

    st.title("📊 Mini Dashboard ETF")

    tickers_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=180))
    end_date = st.date_input("End Date", datetime.now())

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
        st.error("❌ No data found.")
    else:
        st.write("### Raw Prices")
        st.line_chart(df)

        perf = df / df.iloc[0] * 100
        st.write("### Normalized Performance (Base 100)")
        st.line_chart(perf)

# ----------------------------
# PAGE 2 : METRICS (FAKE)
# ----------------------------
elif page == "Metrics":
    st.title("📐 Metrics Module")
    st.info("This module will contain performance & risk metrics.")
    st.write("Coming soon…")

# ----------------------------
# PAGE 3 : STRATEGY (FAKE)
# ----------------------------
elif page == "Strategy":
    st.title("🧠 Strategy Module")
    st.info("This module will contain allocation / signals.")
    st.write("Coming soon…")

# ----------------------------
# PAGE 4 : RISK (FAKE)
# ----------------------------
elif page == "Risk":
    st.title("⚠️ Risk Module")
    st.info("This module will contain drawdowns, VaR, stress tests.")
    st.write("Coming soon…")

# ----------------------------
# PAGE 5 : PREDICT (FAKE)
# ----------------------------
elif page == "Predict":
    st.title("🔮 Predict Module")
    st.info("This module will contain forecasting models.")
    st.write("Coming soon…")
