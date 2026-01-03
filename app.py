import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. CONFIGURATION "BLACKROCK ALADDIN" ---
st.set_page_config(page_title="Predict.", layout="wide", page_icon="▪️")

# CSS "SILENT LUXURY"
st.markdown("""
<style>
    /* IMPORT FONT : Inter (Standard moderne UI) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* --- BASE --- */
    .stApp {
        background-color: #0E1117; /* Anthracite très profond (Pas noir pur) */
        font-family: 'Inter', sans-serif;
    }
    
    /* --- TYPOGRAPHIE --- */
    h1, h2, h3, p, div { color: #E0E0E0; }
    
    /* Le Titre "Predict." */
    .title-text {
        font-size: 3rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.05em;
        margin-bottom: 0px;
    }
    .accent-dot { color: #8B5CF6; font-size: 3rem; } /* VIOLET CHIRURGICAL */
    .subtitle {
        font-size: 0.9rem;
        color: #888888;
        font-weight: 400;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* --- SIDEBAR (PANNEAU DE GAUCHE) --- */
    section[data-testid="stSidebar"] {
        background-color: #161B22; /* Un ton plus clair que le fond pour détacher */
        border-right: 1px solid #30363D;
    }
    
    /* --- INPUTS & SLIDERS (STYLE HAUT DE GAMME) --- */
    .stTextInput > div > div > input {
        background-color: #0D1117;
        color: #FFFFFF;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #8B5CF6; /* Focus Violet */
    }
    
    /* Sliders : La barre est grise, le curseur est blanc/violet */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #8B5CF6 !important; } 
    div.stSlider > div[data-baseweb="slider"] > div > div > div { background-color: #30363D !important; }

    /* --- CARDS KPI (LES BLOCS EN HAUT) --- */
    /* On transforme les métriques Streamlit en "Cartes" élégantes */
    div[data-testid="stMetric"] {
        background-color: #161B22; /* Fond carte */
        border: 1px solid #30363D; /* Bordure subtile */
        border-radius: 8px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); /* Ombre légère pour la profondeur */
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E; /* Gris moyen lisible */
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF; /* Blanc Pur */
        font-size: 1.8rem;
        font-weight: 600;
    }

    /* --- GRAPHIQUE CLEAN --- */
    /* Retirer les marges blanches et les outils */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    
    /* Cacher Header/Footer Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- MOTEUR (Le même, fiable) ---

@st.cache_data(ttl=3600)
def get_data_robust(tickers, start_date):
    try:
        tickers = list(set([t.strip().upper() for t in tickers if t.strip() != ""]))
        if not tickers: return pd.DataFrame()
        df = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()
        
        # Logique extraction prix
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

# 1. HEADER (Titre avec le point violet injecté en HTML)
st.markdown("""
<div style="margin-top: -20px;">
    <span class="title-text">Predict</span><span class="accent-dot">.</span>
    <div class="subtitle">Institutional Asset Allocation</div>
</div>
""", unsafe_allow_html=True)

# 2. SIDEBAR (Configuration)
with st.sidebar:
    st.markdown("### CONFIGURATION")
    tickers_input = st.text_input("Assets (Yahoo Finance)", "PUST.PA, LQQ.PA")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    years = st.slider("Lookback Period (Years)", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)
    
    st.markdown("---")
    st.markdown("### STRATEGY")
    
    # Custom Radio styling is hard, using standard but clean
    mode = st.radio("Allocation Mode", ["Fixed Weight", "Max Sharpe (AI)"], label_visibility="collapsed")
    
    weights = []
    if mode == "Fixed Weight" and len(tickers) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        w = st.slider(f"Weight: {tickers[0]}", 0, 100, 50)
        weights = [w/100, 1-(w/100)]
        
        # Affichage propre des poids
        st.markdown(f"""
        <div style="background-color: #21262d; padding: 10px; border-radius: 4px; margin-top: 10px; border: 1px solid #30363d;">
            <div style="display:flex; justify-content:space-between; color:#C9D1D9; font-size:0.9rem;">
                <span>{tickers[0]}</span> <span><b>{weights[0]:.0%}</b></span>
            </div>
            <div style="display:flex; justify-content:space-between; color:#8B949E; font-size:0.9rem; margin-top:5px;">
                <span>{tickers[1]}</span> <span>{weights[1]:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 3. DASHBOARD PRINCIPAL
if len(tickers) > 0:
    data = get_data_robust(tickers, start_date)
    
    if not data.empty:
        returns = data.pct_change().dropna()
        
        # Logique Poids
        if mode == "Max Sharpe (AI)" and len(data.columns) >= 2:
            weights = optimize(returns)
        if len(weights) != len(data.columns):
            weights = [1/len(data.columns)] * len(data.columns)

        # Calculs Core
        port_ret = returns.dot(weights)
        cum_port = (1 + port_ret).cumprod() * 100
        
        # KPIs
        total_ret = cum_port.iloc[-1]/100 - 1
        cagr = (total_ret + 1)**(252/len(data)) - 1
        vol = port_ret.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0

        # --- SECTION KPI (CARDS) ---
        # Utilisation de colonnes avec un gap pour aérer
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        col1.metric("CAGR (Annual)", f"{cagr:.2%}")
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col3.metric("Volatility", f"{vol:.2%}")
        col4.metric("Total Return", f"{total_ret:.2%}")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # --- SECTION GRAPHIQUE ---
        
        # Toggle Switch (Checkbox stylisée)
        show_base100 = st.toggle("Compare Base 100", value=True)

        fig = go.Figure()

        # COULEURS :
        # Portefeuille = Violet (#8B5CF6) pour le lier à la marque
        # Benchmark/Actifs = Gris / Blanc cassé
        
        if show_base100:
            # 1. Portefeuille (HERO)
            fig.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port, 
                name="PREDICT PORTFOLIO", 
                mode='lines',
                line=dict(color='#8B5CF6', width=2.5), # VIOLET PREDICT
                # Petit dégradé sous la courbe pour effet premium
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)' 
            ))
            
            # 2. Actifs (CONTEXTE)
            colors = ['#E6EDF3', '#6E7681', '#484F58'] # Blanc sale, Gris moyen, Gris foncé
            for i, col in enumerate(data.columns):
                cum_asset = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=cum_asset.index, y=cum_asset, 
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=1, dash='solid'),
                    opacity=0.7 # Légèrement transparent pour ne pas voler la vedette
                ))
            y_title = "Growth (Base 100)"
        else:
            # Mode Prix brut
            for i, col in enumerate(data.columns):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], name=col,
                    line=dict(width=1.5)
                ))
            y_title = "Price"

        # DESIGN GRAPHIQUE "MCKINSEY"
        fig.update_layout(
            # Fond Transparent (se fond dans l'app)
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            
            # Police
            font=dict(family="Inter, sans-serif", color='#8B949E', size=11),
            
            # Grille très fine
            xaxis=dict(showgrid=False, linecolor='#30363D'),
            yaxis=dict(
                title=y_title, 
                showgrid=True, 
                gridcolor='#21262D', # Grille à peine visible
                zeroline=False
            ),
            
            # Légende
            legend=dict(
                orientation="h", y=1.05, x=0,
                font=dict(color="#C9D1D9")
            ),
            
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No Data. Please check tickers.")
else:
    st.info("Awaiting input.")
