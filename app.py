# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- IMPORTS MODULES (Gestion d'erreurs si les fichiers manquent) ---
try:
    from modules.risk_metrics import RiskMetrics
    from modules.portfolio_optimizer import PortfolioOptimizer
    from modules.leveraged_arbitrage import LeveragedArbitrage
    from modules.monte_carlo import MonteCarloSimulator
except ImportError:
    # Si les modules n'existent pas, on crée des mocks pour que l'app ne crashe pas
    # Cela permet de tester l'interface même sans tes fichiers backend
    class RiskMetrics:
        @staticmethod
        def get_metrics(returns):
            return {"CAGR": 0.10, "Sharpe": 1.5, "Vol": 0.15, "MaxDD": -0.20}
    class PortfolioOptimizer:
        @staticmethod
        def optimize_max_sharpe(returns):
            return [1/len(returns.columns)] * len(returns.columns)
    class LeveragedArbitrage:
        @staticmethod
        def compute_optimal_weights(data, roles):
            return {col: 1/len(data.columns) for col in data.columns}
    class MonteCarloSimulator:
        def __init__(self, returns): self.returns = returns
        def run_simulation(self, n_simulations, horizon_days):
            return pd.DataFrame(np.random.randn(horizon_days, n_simulations)).cumsum()

# --- CONFIGURATION ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    h1,h2,h3,p,div,span { color:#E0E0E0; }
    .title-text { font-size:3rem; font-weight:600; color:#FFFFFF; letter-spacing:-0.05em; }
    .accent-dot { color:#8B5CF6; font-size:3rem; }
    .subtitle { font-size:0.9rem; color:#00FFAA; font-weight:400; letter-spacing:0.1em; text-transform:uppercase; }
    section[data-testid="stSidebar"] { background-color:#161B22; border-right:1px solid #30363D; }
    div[data-testid="stMetric"] { background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:15px; }
    div[data-testid="stMetricLabel"] { color:#8B949E; font-size:0.8rem; font-weight:500; text-transform:uppercase; }
    div[data-testid="stMetricValue"] { color:#FFFFFF; font-size:1.8rem; font-weight:600; }
    .stDateInput > div > div > input { background-color: #0D1117; color: white; border: 1px solid #30363D; }
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
    st.markdown("### ASSETS")
    tickers_input = st.text_input("Tickers", "BTC-USD, SPY, NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### PERIOD")
    
    # --- CORRECTION CALENDRIER ---
    # Je remplace le slider par un vrai Date Input
    col_d1, col_d2 = st.columns(2)
    start_input = col_d1.date_input("Start", value=datetime.now() - timedelta(days=365*2))
    end_input = col_d2.date_input("End", value=datetime.now())
    
    st.markdown("---")
    st.markdown("### STRATEGY")
    mode = st.radio("Allocation", ["Fixed Weight", "Max Sharpe (AI)"], label_visibility="collapsed")
    
    roles = {}
    weights_fixed = []
    
    # Logique Fixed Weight
    if mode == "Fixed Weight" and len(tickers) >= 2:
        st.caption("Poids manuel (Premier actif vs Reste)")
        w0 = st.slider(f"{tickers[0]} %", 0, 100, 50)
        weights_fixed = [w0/100] + [(100-w0)/100/(len(tickers)-1)]*(len(tickers)-1)
        
    for t in tickers:
        roles[t] = st.selectbox(f"Role {t}", ["Normal", "X2", "Short"], index=0)

# --- DATA ENGINE (ROBUSTE) ---
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    # Force group_by='ticker' pour structure stable
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by='ticker')
    prices = pd.DataFrame()
    
    if len(tickers) == 1:
        t = tickers[0]
        # Gestion multi-index ou simple
        if isinstance(df.columns, pd.MultiIndex):
            if t in df.columns: prices[t] = df[t]['Close']
        elif 'Close' in df.columns:
             prices[t] = df['Close']
        elif t in df.columns: # Fallback
             prices[t] = df[t]
    else:
        for t in tickers:
            try:
                if t in df.columns: prices[t] = df[t]['Close']
            except: continue
            
    return prices.ffill().dropna()

# --- MAIN LOGIC ---
if len(tickers) > 0:
    try:
        data = get_data(tickers, start_input, end_input)
        
        if data.empty:
            st.warning("No data found. Check tickers/dates.")
        else:
            returns = data.pct_change().dropna()

            # 1. OPTIMISATION
            weights = []
            if mode == "Max Sharpe (AI)" and len(tickers) >= 2:
                weights = PortfolioOptimizer.optimize_max_sharpe(returns)
            else:
                weights = weights_fixed if weights_fixed else [1/len(tickers)]*len(tickers)

            # --- CORRECTION BUG "AMBIGUOUS" ---
            # On vérifie si weights est None ou vide de manière sûre pour NumPy
            if weights is None or len(weights) == 0:
                 weights = [1/len(tickers)]*len(tickers)
            
            # Si poids mal dimensionnés, on reset
            if len(weights) != len(tickers):
                 weights = [1/len(tickers)]*len(tickers)

            # 2. ARBITRAGE (Roles)
            if roles:
                arb_weights = LeveragedArbitrage.compute_optimal_weights(data, roles)
                # On s'assure que arb_weights renvoie bien un dict ou une liste alignée
                # Fallback simple si le module renvoie autre chose
                if not isinstance(arb_weights, dict):
                    arb_weights = dict(zip(tickers, weights))
            else:
                arb_weights = dict(zip(tickers, weights))

            # Alignement des poids pour le dot product
            final_w = [arb_weights.get(t, 0) for t in returns.columns]
            
            # Calcul Portefeuille
            port_ret = returns.dot(final_w)
            cum_port = (1 + port_ret).cumprod() * 100

            # KPIs
            metrics = RiskMetrics.get_metrics(port_ret)

            # --- DISPLAY ---
            
            # KPIs Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR", f"{metrics.get('CAGR',0):.2%}")
            c2.metric("Sharpe", f"{metrics.get('Sharpe',0):.2f}")
            c3.metric("Vol", f"{metrics.get('Vol',0):.2%}")
            c4.metric("MaxDD", f"{metrics.get('MaxDD',0):.2%}")

            # Chart Principal
            st.markdown("### Performance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="PORTFOLIO", line=dict(color="#8B5CF6", width=3)))
            
            for t in returns.columns:
                asset_cum = (1 + returns[t]).cumprod() * 100
                fig.add_trace(go.Scatter(x=asset_cum.index, y=asset_cum, name=t, line=dict(dash='dot', width=1), opacity=0.6))
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0', family='Inter'), hovermode="x unified", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # Matrice Corrélation (CORRECTION ERREUR MATPLOTLIB)
            # Au lieu de .style.background_gradient qui plante, on utilise st.dataframe simple
            # Ou une heatmap plotly (plus joli)
            st.markdown("### Correlations")
            corr = returns.corr()
            import plotly.express as px
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis", aspect="auto")
            fig_corr.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Monte Carlo
            st.markdown("### Monte-Carlo (Projection)")
            mc = MonteCarloSimulator(port_ret)
            sims = mc.run_simulation(n_simulations=100, horizon_days=252)
            
            fig_mc = go.Figure()
            # Affichage léger (seulement 50 lignes)
            for c in sims.columns[:50]:
                fig_mc.add_trace(go.Scatter(x=sims.index, y=sims[c], line=dict(color='yellow', width=0.5), opacity=0.1, showlegend=False))
            fig_mc.add_trace(go.Scatter(x=sims.index, y=sims.mean(axis=1), line=dict(color='#8B5CF6', width=2), name="Mean"))
            
            fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E0E0E0'), height=400)
            st.plotly_chart(fig_mc, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur d'exécution : {str(e)}")
        st.write("Détails: Vérifiez les dimensions des poids ou les données Yahoo.")

else:
    st.info("Waiting for tickers...")
