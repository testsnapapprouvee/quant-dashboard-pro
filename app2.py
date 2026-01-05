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

# ----------------------------
# --- CONFIG PAGE ---
# ----------------------------
st.set_page_config(page_title="Predict Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Fond global sombre */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    /* Sidebar sombre */
    .css-1d391kg {background-color: #1F1F1F;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# --- TITRE PREDICT ---
# ----------------------------
st.markdown(
    """
    <h1 style="font-family:sans-serif; font-size:48px; color:#E0E0E0;">
        Pred<span style="color:#8A2BE2;">i</span>ct
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# --- BOUTONS HORIZONTAUX ---
# ----------------------------
buttons = ["Dashboard", "Metrics", "Strategy", "Risk", "Predict"]
cols = st.columns(len(buttons))

if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

for i, name in enumerate(buttons):
    is_active = st.session_state["page"] == name
    btn_color = "#8A2BE2" if is_active else "#1F1F1F"
    btn_text = "#000000" if is_active else "#E0E0E0"
    cols[i].markdown(
        f"""
        <div style="
            background-color:{btn_color};
            color:{btn_text};
            text-align:center;
            padding:10px 20px;
            border-radius:8px;
            font-weight:bold;
            cursor:pointer;
            border:1px solid #333333;
        ">{name}</div>
        """,
        unsafe_allow_html=True
    )
    if cols[i].button("", key=f"hidden_{i}"):
        st.session_state["page"] = name

page = st.session_state["page"]

# ----------------------------
# --- SELECTEUR DE PERIODE ---
# ----------------------------
date_options = ["YTD", "1Y", "3Y", "5Y", "2022", "Custom"]
selected_period = st.selectbox("Select Period", date_options, index=0)

today = datetime.today()
if selected_period == "YTD":
    start_date = datetime(today.year, 1, 1)
elif selected_period == "1Y":
    start_date = today - timedelta(days=365)
elif selected_period == "3Y":
    start_date = today - timedelta(days=365*3)
elif selected_period == "5Y":
    start_date = today - timedelta(days=365*5)
elif selected_period == "2022":
    start_date = datetime(2022, 1, 1)
else:
    start_date = st.date_input("Start Date", today - timedelta(days=180))
end_date = st.date_input("End Date", today)

# ----------------------------
# --- PAGE DASHBOARD ---
# ----------------------------
if page == "Dashboard":

    tickers_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

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
        st.markdown(
            metrics_df.to_html(classes="table", index=True), 
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <style>
                .table { color:#E0E0E0; background-color:#1F1F1F; border:1px solid #333333; }
                .table th, .table td { border: 1px solid #333333; padding:5px; }
            </style>
            """,
            unsafe_allow_html=True
        )

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
