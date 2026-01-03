import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. CONFIGURATION "SILENT LUXURY" ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

# Injection CSS : Noir Profond, Typo Suisse, Accent Violet subtil
st.markdown("""
<style>
    /* IMPORT FONT Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* RESET GLOBAL */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        background-color: #000000;
        color: #e5e5e5;
    }

    /* FOND NOIR ABSOLU */
    .stApp { background-color: #000000; }

    /* TITRE LOGO */
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 3.5rem;
        letter-spacing: -1.5px;
        color: #ffffff;
        margin-bottom: 0;
        line-height: 1;
    }
    .dot { color: #8b5cf6; } /* LE POINT VIOLET */

    /* SIDEBAR INVISIBLE */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1a1a1a;
    }

    /* INPUTS & WIDGETS */
    .stTextInput > div > div > input {
        background-color: #0a0a0a;
        color: #fff;
        border: 1px solid #333;
        border-radius: 0;
    }
    .stTextInput > div > div > input:focus { border-color: #8b5cf6; }

    /* METRICS "RAZOR" (Minimaliste avec ligne verticale) */
    div[data-testid="stMetric"] {
        background-color: transparent;
        border-left: 1px solid #333;
        padding-left: 15px;
        box-shadow: none;
    }
    div[data-testid="stMetricLabel"] {
        color: #666666;
        font-size: 0.7rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 400;
    }

    /* CHARTS */
    .js-plotly-plot .plotly .modebar { display: none !important; } /* Cache la barre d'outils plotly */
    
    /* ELEMENTS CACHÉS */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    hr { border-color: #1a1a1a; }

</style>
""", unsafe_allow_html=True)

# --- 2. MOTEUR QUANTITATIF (ROBUSTE) ---

@st.cache_data(ttl=3600)
def get_data_robust(tickers, start_date):
    """Téléchargement sécurisé des données"""
    try:
        # Nettoyage des doublons et des espaces
        tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
        if not tickers: return pd.DataFrame()

        df = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()

        # Gestion des formats Yahoo (Parfois complexe)
        if len(tickers) == 1:
            t = tickers[0]
            # Vérifie si 'Close' existe directement ou sous le ticker
            if 'Close' in df.columns: prices[t] = df['Close']
            elif t in df.columns and 'Close' in df[t]: prices[t] = df[t]['Close']
        else:
            for t in tickers:
                if t in df.columns and 'Close' in df[t]:
                    prices[t] = df[t]['Close']
        
        # Nettoyage final : Forward Fill (jours fériés) puis Drop NA
        return prices.fillna(method='ffill').dropna()
    except Exception:
        return pd.DataFrame()

def optimize(returns):
    """Optimisation de portefeuille (Markowitz)"""
    n = len(returns.columns)
    if n < 2: return [1.0] # Pas d'optimisation si 1 seul actif
    
    def neg_sharpe(w):
        r = np.sum(returns.mean()*w)*252
        v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
        return -r/v if v > 0 else 0
    
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = tuple((0,1) for _ in range(n))
    res = minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=cons)
    return res.x

# --- 3. INTERFACE PREDICT. ---

# LOGO / TITRE
st.markdown('<div class="logo-text">Predict<span class="dot">.</span></div>', unsafe_allow_html=True)
st.markdown("<div style='color:#444; font-size: 0.8rem; letter-spacing: 2px; margin-bottom: 40px;'>ALGORITHMIC ASSET ALLOCATION</div>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color:#fff; font-size:12px; margin-bottom:5px;'>TICKERS</h3>", unsafe_allow_html=True)
    tickers_input = st.text_input("Symbole", "PUST.PA, LQQ.PA", help="Séparez par une virgule")
    
    # Traitement propre des tickers
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#fff; font-size:12px; margin-bottom:5px;'>TIMEFRAME</h3>", unsafe_allow_html=True)
    years = st.slider("Années", 1, 10, 3, label_visibility="collapsed")
    start_date = datetime.now() - timedelta(days=years*365)
    
    st.markdown("---")
    
    mode = st.radio("STRATEGY", ["Manual", "Auto-Optimize"], label_visibility="collapsed")
    
    weights = []
    if mode == "Manual" and len(tickers) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        w = st.slider(f"{tickers[0]} Weight", 0, 100, 50)
        weights = [w/100, 1-(w/100)]
        # Affichage texte sobre
        st.markdown(f"<div style='color:#666; font-size:12px; margin-top:10px;'>{tickers[0]}: <b style='color:#fff'>{weights[0]:.0%}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#666; font-size:12px;'>{tickers[1]}: <b style='color:#fff'>{weights[1]:.0%}</b></div>", unsafe_allow_html=True)

# --- DASHBOARD ---

if len(tickers) > 0:
    data = get_data_robust(tickers, start_date)
    
    if not data.empty:
        returns = data.pct_change().dropna()
        
        # --- LOGIQUE D'ALLOCATION ---
        # 1. Si Optimisation demandée et assez d'actifs
        if mode == "Auto-Optimize" and len(data.columns) >= 2:
            weights = optimize(returns)
        
        # 2. Si Poids manuel non défini ou incorrect (ex: changement de tickers)
        if len(weights) != len(data.columns):
            weights = [1/len(data.columns)] * len(data.columns)

        # Calculs
        port_ret = returns.dot(weights)
        cum_port = (1 + port_ret).cumprod() * 100
        
        # Metrics
        total_ret = cum_port.iloc[-1]/100 - 1
        cagr = (total_ret + 1)**(252/len(data)) - 1
        vol = port_ret.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0

        # --- AFFICHAGE KPI (LIGNE ÉPURÉE) ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{cagr:.2%}")
        c2.metric("VOLATILITY", f"{vol:.2%}")
        c3.metric("SHARPE", f"{sharpe:.2f}")
        c4.metric("TOTAL RETURN", f"{total_ret:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- GRAPHIQUE ---
        col_toggle, _ = st.columns([2, 5])
        with col_toggle:
            view_mode = st.radio("VIEW MODE", ["Base 100 (Comparison)", "Price (Raw)"], horizontal=True, label_visibility="collapsed")

        fig = go.Figure()

        if view_mode == "Base 100 (Comparison)":
            # 1. Portefeuille (Blanc Pur)
            fig.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port, 
                name="PORTFOLIO", mode='lines',
                line=dict(color='#FFFFFF', width=2)
            ))
            # 2. Benchmark/Actifs (Gris foncé)
            colors = ['#444444', '#666666', '#888888']
            for i, col in enumerate(data.columns):
                cum_asset = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=cum_asset.index, y=cum_asset, name=col,
                    line=dict(color=colors[i % len(colors)], width=1, dash='solid')
                ))
            y_title = "BASE 100"
            
        else: # Mode Prix Réels
            for i, col in enumerate(data.columns):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name=col,
                    line=dict(width=1.5)
                ))
            y_title = "PRICE"

        # Layout "Chart Graphique" Noir
        fig.update_layout(
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font=dict(family="Inter, sans-serif", color='#666666', size=11),
            xaxis=dict(showgrid=False, linecolor='#333'),
            yaxis=dict(title=y_title, showgrid=True, gridcolor='#1a1a1a', zeroline=False),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            height=500,
            showlegend=False # Légende cachée pour pureté (visible au survol)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- MATRICE CORRELATION (Subtile) ---
        if len(data.columns) > 1:
            st.markdown("<br><p style='color:#444; font-size:0.8rem; text-transform:uppercase;'>Correlation Matrix</p>", unsafe_allow_html=True)
            corr = returns.corr()
            import plotly.express as px
            # Echelle de gris
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=['#000000', '#222222', '#eeeeee'], zmin=-1, zmax=1)
            fig_corr.update_layout(
                paper_bgcolor='black', plot_bgcolor='black', font=dict(color='#888'),
                height=250, margin=dict(l=0,r=0,t=0,b=0), coloraxis_showscale=False
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.warning("No data found. Please check tickers.")
else:
    st.info("Enter tickers to initialize analytics.")
