import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta
from quant_engine import RiskMetrics, VectorBacktester

# --- CONFIGURATION PAGE & DESIGN "PREDICT" ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

# TON CSS ORIGINAL (INTOUCHÉ) + AJUSTEMENTS TABS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, div, span { color: #E0E0E0; }
    
    /* TITRE */
    .title-text { font-size: 3rem; font-weight: 600; color: #FFFFFF; letter-spacing: -0.05em; margin-bottom: 0px; }
    .accent-dot { color: #8B5CF6; font-size: 3rem; }
    .subtitle { font-size: 0.9rem; color: #888888; font-weight: 400; letter-spacing: 0.1em; text-transform: uppercase; margin-top: -10px; margin-bottom: 30px; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* INPUTS */
    .stTextInput > div > div > input { background-color: #0D1117; color: #FFFFFF; border: 1px solid #30363D; border-radius: 6px; }
    .stTextInput > div > div > input:focus { border-color: #8B5CF6; }
    .stDateInput > div > div > input { background-color: #0D1117; color: white; }
    
    /* SLIDERS */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #8B5CF6 !important; } 
    div.stSlider > div[data-baseweb="slider"] > div > div > div { background-color: #30363D !important; }

    /* METRICS CARDS */
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { color: #FFFFFF; font-size: 1.8rem; font-weight: 600; }

    /* CLEAN PLOTLY */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    header, footer {visibility: hidden;}
    
    /* TABS STYLING (Pour que ça ne soit pas flou) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid #30363D; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #8B949E; font-weight: 400; }
    .stTabs [aria-selected="true"] { color: #8B5CF6; border-bottom: 2px solid #8B5CF6; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    try:
        if not tickers: return pd.DataFrame()
        df = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()
        if len(tickers) == 1:
            t = tickers[0]
            # Gestion cas unique/multi-index yfinance
            if isinstance(df.columns, pd.MultiIndex): prices[t] = df['Close']
            elif 'Close' in df.columns: prices[t] = df['Close']
        else:
            for t in tickers:
                # Gestion robuste des colonnes
                if t in df.columns:
                    try: prices[t] = df[t]['Close']
                    except: pass
                elif isinstance(df.columns, pd.MultiIndex) and t in df.columns.levels[0]:
                     prices[t] = df[t]['Close']
        return prices.fillna(method='ffill').dropna()
    except Exception as e:
        print(e)
        return pd.DataFrame()

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

# --- HEADER ---
st.markdown("""
<div style="margin-top: -20px;">
    <span class="title-text">Predict</span><span class="accent-dot">.</span>
    <div class="subtitle">Institutional Asset Allocation</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR (Contrôles Globaux) ---
with st.sidebar:
    st.markdown("### CONFIGURATION")
    tickers_input = st.text_input("Assets", "BTC-USD, ETH-USD, SPY, NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### CALENDRIER")
    # LE VOILA LE CALENDRIER
    date_range = st.date_input(
        "Période d'analyse",
        value=(datetime.now() - timedelta(days=365*2), datetime.now()),
        format="DD/MM/YYYY"
    )

# --- LOGIQUE PRINCIPALE ---
if len(tickers) > 0 and isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    data = get_data(tickers, start_d, end_d)
    
    if not data.empty:
        returns = data.pct_change().dropna()
        
        # ONGLETS PROPRES
        tab1, tab2, tab3 = st.tabs(["ALLOCATION", "CORRELATIONS", "BACKTEST"])
        
        # === TAB 1: ALLOCATION (Ton code original) ===
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_opt, col_void = st.columns([1, 2])
            with col_opt:
                mode = st.radio("Optimization", ["Fixed Weight", "Max Sharpe (AI)"])
            
            weights = []
            if mode == "Max Sharpe (AI)" and len(data.columns) >= 2:
                weights = optimize(returns)
            else:
                weights = [1/len(data.columns)] * len(data.columns) # Equi-weighted par défaut
            
            port_ret = returns.dot(weights)
            cum_port = (1 + port_ret).cumprod() * 100
            
            # KPI
            total_ret = cum_port.iloc[-1]/100 - 1
            vol = port_ret.std() * np.sqrt(252)
            sharpe = (total_ret) / vol if vol > 0 else 0 # Simplifié
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Return", f"{total_ret:.2%}")
            k2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            k3.metric("Volatilité", f"{vol:.2%}")
            
            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="PORTFOLIO", line=dict(color='#8B5CF6', width=2), fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'))
            for col in data.columns:
                asset_curve = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(x=asset_curve.index, y=asset_curve, name=col, line=dict(color='#484F58', width=1), opacity=0.5))
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#8B949E'), xaxis=dict(showgrid=False, linecolor='#30363D'), yaxis=dict(showgrid=True, gridcolor='#21262D'), height=450, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        # === TAB 2: CORRELATIONS (La feature manquante) ===
        with tab2:
            st.markdown("### Matrix")
            corr = returns.corr()
            
            # Heatmap Plotly Custom "Dark"
            fig_corr = px.imshow(
                corr, 
                text_auto=".2f",
                color_continuous_scale=[[0, '#0E1117'], [1, '#8B5CF6']], # Du noir au violet
                aspect="auto"
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", color='#E0E0E0'),
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # === TAB 3: QUANT BACKTEST (Le moteur rajouté) ===
        with tab3:
            col_b1, col_b2, col_b3 = st.columns(3)
            asset_bt = col_b1.selectbox("Asset to Test", tickers)
            sma_s = col_b2.number_input("Short MA", 10, 100, 20)
            sma_l = col_b3.number_input("Long MA", 50, 365, 50)
            
            if asset_bt:
                bt = VectorBacktester(data[asset_bt])
                res = bt.run_strategy(sma_s, sma_l)
                met = RiskMetrics.get_metrics(res['Strat_Ret'])
                
                # Metrics Backtest
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Strategy Return", f"{(res['Curve'].iloc[-1]/100 - 1):.2%}")
                b2.metric("Max Drawdown", f"{met['MaxDD']:.2%}")
                b3.metric("Sharpe", f"{met['Sharpe']:.2f}")
                b4.metric("VaR 95%", f"{met['VaR']:.2%}")
                
                # Graphique Backtest
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Curve'], name="STRATEGY", line=dict(color='#8B5CF6', width=2)))
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['BH_Curve'], name="BUY & HOLD", line=dict(color='#484F58', width=1, dash='dot')))
                fig_bt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#8B949E'), xaxis=dict(showgrid=False, linecolor='#30363D'), yaxis=dict(showgrid=True, gridcolor='#21262D'), height=400, margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig_bt, use_container_width=True)

    else:
        st.warning("No data found for these tickers/dates.")
else:
    st.info("Please select a date range.")
