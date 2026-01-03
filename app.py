import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Import des modules quant ---
from modules.risk_metrics import RiskMetrics
from modules.vector_backtester import VectorBacktester
from modules.leveraged_arbitrage import LeveragedArbitrage
from modules.portfolio_optimizer import PortfolioOptimizer

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

# --- CSS / CHARTE GRAPHIQUE ---
st.markdown("""
<style>
/* Ici tu peux mettre ta charte violette/verte/jaune BlackRock */
</style>
""", unsafe_allow_html=True)

# --- DATA ROBUSTE ---
@st.cache_data(ttl=3600)
def get_data(tickers, start_date):
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
    prices = pd.DataFrame()

    # Si un seul ticker
    if len(tickers) == 1:
        t = tickers[0]
        if 'Close' in df.columns:
            prices[t] = df['Close']
        elif (t, 'Close') in df.columns:
            prices[t] = df[(t, 'Close')]
    else:
        for t in tickers:
            try:
                prices[t] = df[t]['Close']
            except KeyError:
                st.warning(f"Ticker {t} introuvable ou données manquantes sur Yahoo")
    return prices.fillna(method='ffill').dropna()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### CONFIGURATION")
    tickers_input = st.text_input("Assets (Yahoo Finance)", "PUST.PA, LQQ.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    years = st.slider("Lookback Period (Years)", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)

    st.markdown("---")
    st.markdown("### STRATEGY")
    mode = st.radio("Allocation Mode", ["Fixed Weight", "Max Sharpe (AI)"], label_visibility="collapsed")

# --- CHARGEMENT DATA ---
if tickers:
    data = get_data(tickers, start_date)
    
    if not data.empty:
        returns = data.pct_change().dropna()
        
        # Poids
        if mode == "Max Sharpe (AI)" and len(data.columns) >= 2:
            weights = PortfolioOptimizer.optimize(returns)
        else:
            w = st.slider(f"Weight: {tickers[0]}", 0, 100, 50)
            weights = [w/100, 1-(w/100)]
        
        # Portefeuille
        port_ret = returns.dot(weights)
        cum_port = (1 + port_ret).cumprod() * 100

        # KPI
        metrics = RiskMetrics.get_metrics(port_ret)
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        col1.metric("CAGR", f"{metrics['CAGR']:.2%}")
        col2.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
        col3.metric("Volatility", f"{metrics['Vol']:.2%}")
        col4.metric("VaR 95%", f"{metrics['VaR']:.2%}")
        
        # --- GRAPHIQUE ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", line=dict(color='#8B5CF6', width=2)))
        for i, col in enumerate(data.columns):
            cum_asset = (1 + returns[col]).cumprod() * 100
            fig.add_trace(go.Scatter(x=cum_asset.index, y=cum_asset, name=col, line=dict(width=1.5, dash='dash')))
        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Aucune donnée disponible pour ces tickers.")
else:
    st.info("Veuillez entrer au moins un ticker.")
