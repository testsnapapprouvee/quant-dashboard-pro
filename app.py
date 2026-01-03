# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- IMPORTS DES MODULES ---
# Assure-toi que le dossier "modules" existe bien avec ces fichiers dedans
# Si tu n'as pas ces fichiers, dis-le moi, je te donnerai le code pour les créer.
try:
    from modules.risk_metrics import RiskMetrics
    from modules.portfolio_optimizer import PortfolioOptimizer
    from modules.leveraged_arbitrage import LeveragedArbitrage
    from modules.monte_carlo import MonteCarloSimulator
except ImportError as e:
    st.error(f"Erreur d'importation : {e}. Vérifie que le dossier 'modules' et les fichiers existent.")
    st.stop()

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

# --- DATA FETCH (VERSION CORRIGÉE ET ROBUSTE) ---
@st.cache_data(ttl=3600)
def get_data(tickers, start_date):
    """
    Récupère les données avec une structure garantie :
    Si group_by='ticker', yfinance renvoie [Ticker] -> [Close, Open, ...]
    """
    if not tickers:
        return pd.DataFrame()

    # On force group_by='ticker' pour éviter les erreurs de format aléatoire
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, group_by='ticker')
    
    prices = pd.DataFrame()

    # Cas 1: Un seul ticker
    if len(tickers) == 1:
        t = tickers[0]
        # Parfois yfinance aplatit le DataFrame s'il n'y a qu'un ticker
        if isinstance(df.columns, pd.MultiIndex):
            if t in df.columns:
                prices[t] = df[t]['Close']
        elif 'Close' in df.columns:
             prices[t] = df['Close']
        # Fallback
        elif t in df.columns:
             prices[t] = df[t]

    # Cas 2: Plusieurs tickers
    else:
        for t in tickers:
            try:
                if t in df.columns:
                    # On extrait proprement la colonne Close du ticker
                    prices[t] = df[t]['Close']
            except KeyError:
                continue # Si un ticker plante, on continue sans lui

    # Nettoyage (.ffill() remplace fillna(method='ffill') qui est obsolète)
    return prices.ffill().dropna()

# --- MAIN ---
if len(tickers) > 0:
    try:
        data = get_data(tickers, start_date)
        
        if data.empty:
            st.warning("Aucune donnée récupérée. Vérifiez les tickers.")
        else:
            returns = data.pct_change().dropna()

            # Arbitrage + Optimisation
            if mode == "Max Sharpe (AI)" and len(tickers) >= 2:
                # Appel à ton module externe
                weights = PortfolioOptimizer.optimize_max_sharpe(returns)
                roles = {t: "Normal" for t in tickers}  # par défaut
            
            # Fallback si l'optimisation échoue ou pas de poids définis
            if not weights or len(weights) != len(tickers):
                weights = [1/len(tickers)]*len(tickers)

            # Arbitrage levier (Module externe)
            if roles:
                arb_weights = LeveragedArbitrage.compute_optimal_weights(data, roles)
            else:
                arb_weights = dict(zip(tickers, weights))

            # Portefeuille
            # On s'assure que l'ordre des poids correspond aux colonnes
            safe_weights = [arb_weights.get(col, 0) for col in returns.columns]
            port_ret = returns.dot(safe_weights)
            cum_port = (1 + port_ret).cumprod() * 100

            # KPIs (Module externe)
            metrics = RiskMetrics.get_metrics(port_ret)
            
            col1, col2, col3, col4 = st.columns(4, gap="medium")
            col1.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
            col2.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
            col3.metric("Vol", f"{metrics.get('Vol', 0):.2%}")
            col4.metric("MaxDD", f"{metrics.get('MaxDD', 0):.2%}")

            # --- GRAPHIQUES ---
            st.markdown("### Portefeuille vs Actifs")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfolio", line=dict(color="#8B5CF6", width=3)))
            
            for t in returns.columns:
                asset_curve = (1+returns[t]).cumprod()*100
                fig.add_trace(go.Scatter(x=asset_curve.index, y=asset_curve, name=t, line=dict(width=1.5, dash="dot", color="#00FFAA"), opacity=0.7))
            
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
            # Appel à ton module externe
            mc_sim = MonteCarloSimulator(port_ret)
            sim_df = mc_sim.run_simulation(n_simulations=200, horizon_days=252)
            
            fig_mc = go.Figure()
            # On affiche seulement un échantillon pour ne pas alourdir le graph
            for col in sim_df.columns[:50]: 
                fig_mc.add_trace(go.Scatter(x=sim_df.index, y=sim_df[col], line=dict(width=0.5, color='yellow'), opacity=0.1, showlegend=False))
            
            fig_mc.add_trace(go.Scatter(x=sim_df.index, y=sim_df.mean(axis=1), line=dict(width=2, color='#8B5CF6'), name="Mean"))
            
            fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#E0E0E0', family='Inter'),
                                hovermode="x unified", height=500)
            st.plotly_chart(fig_mc, use_container_width=True)

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'exécution : {e}")
        # Affiche le détail technique seulement si besoin pour debugger
        st.caption("Vérifiez que vos modules (risk_metrics, etc.) sont bien compatibles avec les données.")

else:
    st.info("Awaiting tickers input.")
