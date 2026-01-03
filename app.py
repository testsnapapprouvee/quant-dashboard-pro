# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Modules quant
from modules.risk_metrics import RiskMetrics
from modules.portfolio_optimizer import PortfolioOptimizer
from modules.leveraged_arbitrage import LeveragedArbitrage
from modules.monte_carlo import MonteCarloSimulator

# --- 1. CONFIGURATION STREAMLIT & STYLE BLACKROCK ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    h1,h2,h3,p,div { color:#E0E0E0; }

    .title-text { font-size:3rem; font-weight:600; color:#FFFFFF; letter-spacing:-0.05em; }
    .accent-dot { color:#8B5CF6; font-size:3rem; }
    .subtitle { font-size:0.9rem; color:#00FFAA; font-weight:400; letter-spacing:0.1em; text-transform:uppercase; }

    section[data-testid="stSidebar"] { background-color:#161B22; border-right:1px solid #30363D; }
    div[data-testid="stMetric"] { background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:15px 20px; box-shadow:0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stMetricLabel"] { color:#8B949E; font-size:0.8rem; font-weight:500; text-transform:uppercase; }
    div[data-testid="stMetricValue"] { color:#FFFFFF; font-size:1.8rem; font-weight:600; }
    .js-plotly-plot .plotly .modebar { display:none !important; }
    header,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="margin-top:-20px;">
    <span class="title-text">Predict</span><span class="accent-dot">.</span>
    <div class="subtitle">Institutional Asset Allocation</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### CONFIGURATION DES ACTIFS")
    tickers_input = st.text_input("Yahoo Tickers", "PUST.PA, LQQ.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    years = st.slider("Lookback Period (Years)", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)
    
    st.markdown("---")
    st.markdown("### STRATEGIE D'ALLOCATION")
    mode = st.radio("Allocation Mode", ["Fixed Weight", "Max Sharpe (AI)"], label_visibility="collapsed")
    
    weights = []
    roles = {}
    if mode == "Fixed Weight" and len(tickers) >= 2:
        w0 = st.slider(f"Weight {tickers[0]}", 0, 100, 50)
        weights = [w0/100, 1-w0/100]
        for i, t in enumerate(tickers):
            roles[t] = st.selectbox(f"Role {t}", ["Normal", "X2", "Short"], index=0)

# --- DATA FETCH ---
@st.cache_data(ttl=3600)
def get_data(tickers, start_date):
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=True)
    if len(tickers)==1:
        return df['Close'].to_frame()
    else:
        return pd.concat([df[t]['Close'] for t in tickers], axis=1).rename(columns={i:tickers[i] for i in range(len(tickers))}).fillna(method='ffill').dropna()

# --- MAIN ---
if len(tickers) > 0:
    data = get_data(tickers, start_date)
    returns = data.pct_change().dropna()

    # Arbitrage + Optimisation
    if mode == "Max Sharpe (AI)" and len(tickers) >= 2:
        weights = PortfolioOptimizer.optimize_max_sharpe(returns)
        roles = {t: "Normal" for t in tickers}  # par défaut
    if not weights or len(weights) != len(tickers):
        weights = [1/len(tickers)]*len(tickers)

    # Arbitrage levier
    if roles:
        arb_weights = LeveragedArbitrage.compute_optimal_weights(data, roles)
    else:
        arb_weights = dict(zip(tickers, weights))

    # Portefeuille
    port_ret = returns.dot(list(arb_weights.values()))
    cum_port = (1 + port_ret).cumprod() * 100

    # KPIs
    metrics = RiskMetrics.get_metrics(port_ret)
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("CAGR", f"{metrics['CAGR']:.2%}")
    col2.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    col3.metric("Vol", f"{metrics['Vol']:.2%}")
    col4.metric("MaxDD", f"{metrics['MaxDD']:.2%}")

    # --- GRAPHIQUES ---
    st.markdown("### Portefeuille vs Actifs")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", line=dict(color="#8B5CF6", width=3)))
    for t in tickers:
        fig.add_trace(go.Scatter(x=(1+returns[t]).cumprod()*100, y=(1+returns[t]).cumprod()*100, name=t, line=dict(width=1.5, dash="dot", color="#00FFAA")))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#E0E0E0', family='Inter'),
                      hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- MATRICE DE CORRELATION ---
    st.markdown("### Matrice de Corrélation")
    corr = returns.corr()
    st.dataframe(corr.style.background_gradient(cmap='viridis'))

    # --- MONTE-CARLO SIMULATION ---
    st.markdown("### Monte-Carlo Simulation (Base 100)")
    mc_sim = MonteCarloSimulator(port_ret)
    sim_df = mc_sim.run_simulation(n_simulations=500, horizon_days=252)
    fig_mc = go.Figure()
    for col in sim_df.columns:
        fig_mc.add_trace(go.Scatter(x=sim_df.index, y=sim_df[col], line=dict(width=0.5, color='yellow'), opacity=0.3, showlegend=False))
    fig_mc.add_trace(go.Scatter(x=sim_df.index, y=sim_df.mean(axis=1), line=dict(width=2, color='#8B5CF6'), name="Mean"))
    fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='#E0E0E0', family='Inter'),
                         hovermode="x unified", height=500)
    st.plotly_chart(fig_mc, use_container_width=True)

else:
    st.info("Awaiting tickers input.")
