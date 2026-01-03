import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. CONFIGURATION "INSTITUTIONAL LUXURY" ---
st.set_page_config(page_title="PREDICT | INSTITUTIONAL", layout="wide", page_icon="▪️")

# Injection CSS : Le style "BlackRock / McKinsey"
st.markdown("""
<style>
    /* IMPORT FONT (Inter pour un look Suisse/Helvetica moderne) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    /* RESET GLOBAL */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 300; /* Light par défaut */
    }

    /* FOND NOIR PROFOND */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }

    /* TYPOGRAPHIE HIÉRARCHISÉE */
    h1 {
        font-weight: 600;
        letter-spacing: -1px;
        font-size: 2.5rem !important;
        color: #ffffff !important;
        margin-bottom: 0rem !important;
    }
    h2, h3 {
        font-weight: 400;
        color: #cccccc !important;
        letter-spacing: -0.5px;
    }
    .caption {
        font-size: 0.8rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* SIDEBAR "INVISIBLE" */
    section[data-testid="stSidebar"] {
        background-color: #050505; /* Noir très légèrement cassé */
        border-right: 1px solid #1a1a1a;
    }

    /* INPUTS MINIMALISTES (Bordures fines, pas de fond) */
    .stTextInput > div > div > input {
        background-color: #000000;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 0px; /* Angles droits */
        font-family: 'Inter';
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffffff; /* Focus blanc pur */
        box-shadow: none;
    }

    /* METRICS "RAZOR" (Pas de carte, juste de l'info) */
    div[data-testid="stMetric"] {
        background-color: transparent;
        border-left: 1px solid #333; /* Ligne fine verticale */
        padding-left: 20px;
        border-radius: 0px;
        box-shadow: none;
    }
    div[data-testid="stMetricLabel"] {
        color: #666666; /* Gris sourd */
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 400;
        font-size: 1.8rem;
    }

    /* BOUTONS & SLIDERS */
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: #ffffff !important; /* Curseurs blancs */
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div {
        background-color: #333333 !important; /* Rails gris foncé */
    }

    /* SEPARATEURS */
    hr {
        border-color: #1a1a1a;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* SUPPRESSION DU HEADER STREAMLIT */
    header {visibility: hidden;}
    footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# --- 2. LOGIQUE QUANTITATIVE (IDENTIQUE MAIS ÉPURÉE) ---

@st.cache_data(ttl=3600)
def get_data_robust(tickers, start_date):
    try:
        tickers = list(set(tickers))
        df = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()
        if len(tickers) == 1:
            t = tickers[0]
            col = 'Close' if 'Close' in df.columns else t
            if t in df.columns: col = df[t]['Close'] 
            prices[t] = df['Close'] if 'Close' in df.columns else df[t]['Close']
        else:
            for t in tickers:
                if t in df.columns: prices[t] = df[t]['Close']
        return prices.fillna(method='ffill').dropna()
    except: return pd.DataFrame()

def optimize(returns):
    n = len(returns.columns)
    def neg_sharpe(w):
        r = np.sum(returns.mean()*w)*252
        v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
        return -r/v if v > 0 else 0
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    return minimize(neg_sharpe, [1/n]*n, bounds=tuple((0,1) for _ in range(n)), constraints=cons).x

# --- 3. LAYOUT INSTITUTIONNEL ---

# En-tête minimaliste
st.markdown("<h1>PREDICT</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption'>QUANTITATIVE ASSET ALLOCATION SYSTEM // V.3.0</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ÉPURÉE ---
with st.sidebar:
    st.markdown("<h3 style='font-size:14px; margin-bottom:10px; color:#fff;'>ASSET SELECTION</h3>", unsafe_allow_html=True)
    
    # Inputs discrets
    tickers_input = st.text_input("", "PUST.PA, LQQ.PA", placeholder="ENTER TICKERS")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Spacer
    
    st.markdown("<h3 style='font-size:14px; margin-bottom:10px; color:#fff;'>TIMEFRAME</h3>", unsafe_allow_html=True)
    years = st.slider("", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)
    
    st.markdown("---")
    
    # Stratégie Switch (Radio horizontal hack via CSS ou juste standard propre)
    st.markdown("<h3 style='font-size:14px; margin-bottom:10px; color:#fff;'>STRATEGY MODE</h3>", unsafe_allow_html=True)
    mode = st.radio("", ["MANUAL ALLOCATION", "MARKOWITZ OPTIMIZATION"], label_visibility="collapsed")
    
    weights = []
    if mode == "MANUAL ALLOCATION" and len(tickers) >= 2:
        st.markdown("<br>", unsafe_allow_html=True)
        w = st.slider(f"WEIGHT {tickers[0]}", 0, 100, 50)
        weights = [w/100, 1-(w/100)]
        # Affichage texte sobre
        st.markdown(f"""
        <div style='font-family:monospace; font-size:12px; color:#888; margin-top:10px;'>
        {tickers[0]}: <span style='color:#fff'>{weights[0]:.0%}</span><br>
        {tickers[1]}: <span style='color:#fff'>{weights[1]:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

# --- DASHBOARD CENTRAL ---

if len(tickers) > 0:
    data = get_data_robust(tickers, start_date)
    
    if not data.empty and len(data.columns) > 0:
        returns = data.pct_change().dropna()
        
        # Moteur d'Optimisation
        if mode == "MARKOWITZ OPTIMIZATION" and len(data.columns) >= 2:
            weights = optimize(returns)
        
        # Fallback poids
        if len(weights) != len(data.columns):
            weights = [1/len(data.columns)] * len(data.columns)

        # Calculs
        port_ret = returns.dot(weights)
        cum_port = (1 + port_ret).cumprod() * 100
        
        # Stats Fin
        total_ret = cum_port.iloc[-1]/100 - 1
        cagr = (total_ret + 1)**(252/len(data)) - 1
        vol = port_ret.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0

        # --- BLOC KPI (GRILLE STRICTE) ---
        # Pas de colonnes serrées, de l'air.
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ANNUAL RETURN (CAGR)", f"{cagr:.2%}")
        c2.metric("VOLATILITY (RISK)", f"{vol:.2%}")
        c3.metric("SHARPE RATIO", f"{sharpe:.2f}")
        c4.metric("TOTAL RETURN", f"{total_ret:.2%}")
        
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # White space

        # --- GRAPHIQUE INSTITUTIONNEL ---
        # Toggle discret
        col_toggle, _ = st.columns([1, 5])
        with col_toggle:
            show_base100 = st.toggle("REBASE 100", value=True)

        fig = go.Figure()

        # Styles de lignes : Blanc pur pour le portefeuille, Gris pour les actifs
        if show_base100:
            # 1. Portefeuille (Blanc pur, ligne franche)
            fig.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port, 
                name="PORTFOLIO", 
                mode='lines',
                line=dict(color='#FFFFFF', width=1.5)
            ))
            
            # 2. Benchmark (Gris, ligne fine)
            colors = ['#444444', '#666666']
            for i, col in enumerate(data.columns):
                cum_asset = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=cum_asset.index, y=cum_asset, 
                    name=col,
                    line=dict(color=colors[i % 2], width=1, dash='solid')
                ))
            title_g = "EQUITY CURVE (BASE 100)"
        else:
            for i, col in enumerate(data.columns):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], 
                    name=col,
                    mode='lines',
                    line=dict(color='#FFFFFF' if i==0 else '#666666', width=1)
                ))
            title_g = "HISTORICAL PRICES"

        # Layout Graphique "Bloomberg Terminal"
        fig.update_layout(
            title=dict(text=title_g.upper(), font=dict(size=12, color='#666666')),
            paper_bgcolor='#000000', # Fond noir
            plot_bgcolor='#000000',
            font=dict(family="Inter, sans-serif", color='#888888', size=10),
            
            # Grille très subtile
            xaxis=dict(showgrid=False, linecolor='#333333', tickfont=dict(color='#666666')),
            yaxis=dict(showgrid=True, gridcolor='#1a1a1a', gridwidth=1, zeroline=False),
            
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            showlegend=False # Légende minimaliste au survol uniquement pour garder pur
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Petit tableau de corrélation sobre en bas
        if len(data.columns) > 1:
            st.markdown("<br><p class='caption'>CORRELATION MATRIX</p>", unsafe_allow_html=True)
            corr = returns.corr()
            # Heatmap en niveaux de gris
            import plotly.figure_factory as ff
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=['#000000', '#333333', '#FFFFFF'])
            fig_corr.update_layout(
                paper_bgcolor='black', plot_bgcolor='black', 
                font=dict(color='#666666', size=10),
                height=250, margin=dict(l=0,r=0,t=0,b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.markdown("<br><br><h3 style='text-align:center; color:#333;'>NO DATA AVAILABLE. CHECK TICKERS.</h3>", unsafe_allow_html=True)
else:
    st.markdown("<br><br><h3 style='text-align:center; color:#333;'>AWAITING INPUT...</h3>", unsafe_allow_html=True)
