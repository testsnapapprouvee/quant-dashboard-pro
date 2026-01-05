import streamlit as st
import pandas as pd
from modules.arbitrage_signals import ArbitrageSignals
from modules.data_engine import get_data  # si tu as un moteur de data

def render():
    st.title("📈 Strategy / Arbitrage")
    
    # Inputs
    tickers_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    start_date = st.date_input("Start Date", pd.Timestamp.now() - pd.Timedelta(days=365))
    end_date = st.date_input("End Date", pd.Timestamp.now())
    
    data = get_data(tickers, start_date, end_date)
    
    if data.empty:
        st.error("❌ No data found")
        return
    
    st.write("### Raw Data")
    st.dataframe(data.tail(10))
    
    # Calcul Arbitrage
    df_rs = ArbitrageSignals.calculate_relative_strength(data)
    df_signals = ArbitrageSignals.get_signal_status(df_rs['RS'])
    
    st.write("### Relative Strength")
    st.dataframe(df_rs.tail(10))
    
    st.write("### Arbitrage Signals")
    st.dataframe(df_signals.tail(10))
    
    # Optionnel : affichage simple sans Plotly
    st.line_chart(df_rs['RS'])
