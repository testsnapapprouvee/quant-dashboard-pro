import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ===============================
# 1. SIDEBAR - Sélection ETF
# ===============================
st.sidebar.header("Sélection des actifs")
tickers = st.sidebar.text_input("Tickers (séparés par virgule)", "LQQ.PA, PUST.PA")
tickers = [t.strip().upper() for t in tickers.split(",")]

period_options = ["1Y", "6M", "3M", "Custom"]
sel_period = st.sidebar.selectbox("Période", period_options, index=1)

today = datetime.today()
if sel_period == "1Y":
    start_d = today - timedelta(days=365)
elif sel_period == "6M":
    start_d = today - timedelta(days=182)
elif sel_period == "3M":
    start_d = today - timedelta(days=90)
else:
    start_d = st.sidebar.date_input("Start date", today - timedelta(days=365))
end_d = today

# ===============================
# 2. DATA
# ===============================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    price_map = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                price_map[t] = df['Close']
        except:
            continue
    if price_map:
        df_final = pd.concat(price_map.values(), axis=1)
        df_final.columns = price_map.keys()
        return df_final.ffill().dropna()
    return pd.DataFrame()

data = get_data(tickers, start_d, end_d)

if data.empty:
    st.error("Pas de données pour ces tickers/date.")
else:
    st.subheader("📈 Performance des actifs")
    # Normaliser à 100 pour comparer
    df_norm = (data / data.iloc[0]) * 100
    st.line_chart(df_norm)

    st.subheader("📊 Données brutes")
    st.dataframe(df_norm.tail(10))
