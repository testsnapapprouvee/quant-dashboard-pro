import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# Import du moteur quant (le fichier qu'on vient de créer)
from quant_engine import RiskMetrics, VectorBacktester

# --- 1. CONFIGURATION "BLACKROCK ALADDIN" ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

# CSS "SILENT LUXURY" (TON DESIGN EXACT)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, div { color: #E0E0E0; }
    .title-text { font-size: 3rem; font-weight: 600; color: #FFFFFF; letter-spacing: -0.05em; margin-bottom: 0px; }
    .accent-dot { color: #8B5CF6; font-size: 3rem; } 
    .subtitle { font-size: 0.9rem; color: #888888; font-weight: 400; letter-spacing: 0.1em; text-transform: uppercase; margin-top: -10px; margin-bottom: 30px; }
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stTextInput > div > div > input { background-color: #0D1117; color: #FFFFFF; border: 1px solid #30363D; border-radius: 6px; padding: 10px; }
    .stTextInput > div > div > input:focus { border-color: #8B5CF6; }
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #8B5CF6 !important; } 
    div.stSlider > div[data-baseweb="slider"] > div > div > div { background-color: #30363D !important; }
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; }
    div[data-testid="stMetricValue"] { color: #FFFFFF; font-size: 1.8rem; font-weight: 600; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    header {visibility: hidden;} footer {visibility: hidden;}
    
    /* Style spécifique pour les Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161B22; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: #8B949E; }
    .stTabs [aria-selected="true"] { background-color: #0E1117; color: #8B5CF6; border-top: 2px solid #8B5CF6; }
</style>
""", unsafe_allow_html=True)

# --- MOTEUR DATA ---
@st.cache_data(ttl=3600)
def get_data_robust(tickers, start_date):
    try:
        tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
        if not tickers: return pd.DataFrame()
        df = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()
        if len(tickers) == 1:
            t = tickers[0]
            if 'Close' in df.columns: prices[t] = df['Close']
            elif t in df.columns and 'Close' in df[t]: prices[t] = df[t]['Close']
        else:
            for t in tickers:
                if t in df.columns and 'Close' in df[t]:
                    prices[t] = df[t]['Close']
        return prices.fillna(method='ffill').dropna()
    except: return pd.DataFrame()

def optimize(returns):
    n = len(returns.columns)
    if n < 2: return [1.0]
    def neg_sharpe(w):
        r = np.sum(returns.mean()*w)*252
        v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
        return -r/v if v > 0 else 0
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = tuple((0,1) for _ in range(n))
    return minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=cons).x

# --- LAYOUT PREMIUM ---
st.markdown("""
<div style="margin-top: -20px;">
    <span class="title-text">Predict</span><span class="accent-dot">.</span>
    <div class="subtitle">Institutional Quantitative System</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR GLOBAL ---
with st.sidebar:
    st.markdown("### DATA FEED")
    tickers_input = st.text_input("Assets (Yahoo Finance)", "BTC-USD, ETH-USD, SPY")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    years = st.slider("Lookback (Years)", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)

# --- MAIN LOGIC ---
if len(tickers) > 0:
    data = get_data_robust(tickers, start_date)
    
    if not data.empty:
        # ONGLETS POUR SEPARER LES FONCTIONS
        tab1, tab2 = st.tabs(["ASSET ALLOCATION", "STRATEGY LAB"])
        
        # ==========================================
        # TAB 1: TON CODE ORIGINAL (ALLOCATION)
        # ==========================================
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_ctrl1, col_ctrl2 = st.columns([1, 3])
            
            with col_ctrl1:
                mode = st.radio("Method", ["Fixed Weight", "Max Sharpe (AI)"])
                weights = []
                if mode == "Fixed Weight" and len(tickers) >= 2:
                    w = st.slider(f"Weight {tickers[0]}", 0, 100, 50)
                    weights = [w/100, 1-(w/100)]
            
            returns = data.pct_change().dropna()
            
            if mode == "Max Sharpe (AI)" and len(data.columns) >= 2:
                weights = optimize(returns)
            if len(weights) != len(data.columns):
                weights = [1/len(data.columns)] * len(data.columns)

            port_ret = returns.dot(weights)
            cum_port = (1 + port_ret).cumprod() * 100
            
            # KPIs
            total_ret = cum_port.iloc[-1]/100 - 1
            cagr = (total_ret + 1)**(252/len(data)) - 1
            vol = port_ret.std() * np.sqrt(252)
            sharpe = cagr / vol if vol > 0 else 0

            # Affichage KPI
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            c1.metric("CAGR", f"{cagr:.2%}")
            c2.metric("Sharpe", f"{sharpe:.2f}")
            c3.metric("Volatilité", f"{vol:.2%}")
            c4.metric("Total Return", f"{total_ret:.2%}")

            # Graphique
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="PORTFOLIO", mode='lines', line=dict(color='#8B5CF6', width=2.5), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
            
            colors = ['#E6EDF3', '#6E7681', '#484F58']
            for i, col in enumerate(data.columns):
                cum_asset = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(x=cum_asset.index, y=cum_asset, name=col, line=dict(color=colors[i % len(colors)], width=1), opacity=0.7))

            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif", color='#8B949E', size=11), xaxis=dict(showgrid=False, linecolor='#30363D'), yaxis=dict(title="Growth (Base 100)", showgrid=True, gridcolor='#21262D'), legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # TAB 2: LE NOUVEAU MOTEUR QUANT (BACKTEST)
        # ==========================================
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Contrôles spécifiques au Backtest
            c_sel, c_s1, c_s2 = st.columns([1, 1, 1])
            with c_sel:
                selected_asset = st.selectbox("Active Asset", tickers)
            with c_s1:
                short_w = st.number_input("Short MA", 5, 100, 20)
            with c_s2:
                long_w = st.number_input("Long MA", 50, 365, 50)

            if selected_asset:
                # Appel du moteur quant
                bt_engine = VectorBacktester(data[selected_asset])
                res = bt_engine.run_sma_strategy(short_w, long_w)
                
                # Calcul des risques avancés
                metrics = RiskMetrics.get_metrics(res['Strategy_Returns'])
                
                # KPI Cards (Même design)
                k1, k2, k3, k4 = st.columns(4, gap="medium")
                k1.metric("Strategy Return", f"{(res['Strategy_Curve'].iloc[-1]/100 - 1):.2%}")
                k2.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                k3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                k4.metric("VaR (95%)", f"{metrics['VaR (95%)']:.2%}")

                # Graphique Strategy
                st.markdown("<br>", unsafe_allow_html=True)
                fig_bt = go.Figure()
                
                # Courbe Stratégie (Violet)
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Strategy_Curve'], name="QUANT STRATEGY", line=dict(color='#8B5CF6', width=2.5)))
                # Courbe Buy & Hold (Gris)
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['BuyHold_Curve'], name="BUY & HOLD", line=dict(color='#484F58', width=1.5, dash='dot')))
                
                fig_bt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif", color='#8B949E', size=11), xaxis=dict(showgrid=False, linecolor='#30363D'), yaxis=dict(title="Performance (Base 100)", showgrid=True, gridcolor='#21262D'), legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified", height=500)
                st.plotly_chart(fig_bt, use_container_width=True)

    else:
        st.warning("No Data. Please check tickers.")
else:
    st.info("Awaiting input.")
