import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ======================================
# CONFIG
# ======================================
st.set_page_config(
    page_title="Mini Data Dashboard",
    layout="wide"
)

st.title("📊 Mini ETF Data Dashboard")
st.caption("Étape 1 — Validation des données Yahoo")

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    st.header("Assets")

    risk = st.text_input("Risk ETF", "LQQ.PA")
    safe = st.text_input("Safe ETF", "PUST.PA")

    st.header("Period")
    period = st.selectbox(
        "Lookback",
        ["6M", "1Y", "3Y", "5Y", "Custom"],
        index=1
    )

    today = datetime.today()

    if period == "6M":
        start = today - timedelta(days=180)
    elif period == "1Y":
        start = today - timedelta(days=365)
    elif period == "3Y":
        start = today - timedelta(days=365 * 3)
    elif period == "5Y":
        start = today - timedelta(days=365 * 5)
    else:
        start = st.date_input("Start", today - timedelta(days=365))
        end = st.date_input("End", today)

    if period != "Custom":
        end = today

    run = st.button("LOAD DATA")

# ======================================
# DATA LOADER (NO CACHE ON PURPOSE)
# ======================================
def load_prices(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return pd.Series(dtype=float)

    return df["Close"].dropna()

# ======================================
# MAIN
# ======================================
if run:
    with st.spinner("Downloading data from Yahoo Finance..."):
        px_risk = load_prices(risk, start, end)
        px_safe = load_prices(safe, start, end)

    if px_risk.empty or px_safe.empty:
        st.error("❌ One or both tickers returned no data.")
        st.stop()

    # Align dates
    df = pd.concat([px_risk, px_safe], axis=1)
    df.columns = ["Risk", "Safe"]
    df = df.dropna()

    # Base 100 performance
    perf = df / df.iloc[0] * 100

    # ==================================
    # METRICS
    # ==================================
    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            f"{risk} Total Return",
            f"{(perf['Risk'].iloc[-1] - 100):.1f}%"
        )

    with c2:
        st.metric(
            f"{safe} Total Return",
            f"{(perf['Safe'].iloc[-1] - 100):.1f}%"
        )

    # ==================================
    # CHART
    # ==================================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=perf.index,
        y=perf["Risk"],
        name=risk,
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=perf.index,
        y=perf["Safe"],
        name=safe,
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Performance (Base 100)",
        yaxis_title="Index (100 = start)",
        xaxis_title="Date",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==================================
    # RAW DATA CHECK
    # ==================================
    with st.expander("🔍 Raw Prices (Debug)"):
        st.dataframe(df.tail(10))
