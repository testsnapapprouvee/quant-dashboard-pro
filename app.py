import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta
from quant_engine import RiskMetrics, VectorBacktester

# --- 1. CONFIGURATION "BLACKROCK" ---
st.set_page_config(page_title="Predict. | Aladdin", layout="wide", page_icon="▪️")

# --- 2. CSS "INSTITUTIONAL" (CORRIGÉ: Police Helvetica/Arial) ---
st.markdown("""
<style>
    /* Pas d'import Google Fonts. On utilise la police système Finance (Froid/Sec) */
    
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif !important;
    }

    /* FOND GRIS FONCÉ (Pas noir pur) */
    .stApp { background-color: #111111; }
    
    /* TEXTES */
    h1, h2, h3, p, div, span, label { color: #E6E6E6 !important; }
    
    /* TITRE LOGO */
    .title-text { 
        font-family: 'Arial', sans-serif; /* Très carré */
        font-size: 2.8rem; 
        font-weight: 700; 
        color: #FFFFFF !important; 
        letter-spacing: -1px; 
    }
    .accent-dot { color: #00FFAA; font-size: 2.8rem; } /* Vert Terminal Bloomberg */
    .subtitle { 
        font-size: 0.85rem; 
        color: #999999 !important; 
        font-weight: 400; 
        letter-spacing: 1px; 
        text-transform: uppercase; 
        margin-top: -5px; 
        margin-bottom: 25px; 
        border-bottom: 1px solid #333;
        padding-bottom: 15px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] { 
        background-color: #000000; /* Sidebar Noir pur */
        border-right: 1px solid #333333; 
    }
    
    /* INPUTS (Look Terminal) */
    .stTextInput > div > div > input { 
        background-color: #1A1A1A; 
        color: #E6E6E6; 
        border: 1px solid #333; 
        border-radius: 0px; /* Bords carrés pro */
        font-family: 'Courier New', monospace; /* Police code pour les inputs */
    }
    .stTextInput > div > div > input:focus { border-color: #00FFAA; }
    
    /* DATE PICKER */
    .stDateInput > div > div > input { 
        background-color: #1A1A1A; 
        color: white; 
        border-radius: 0px; 
    }
    
    /* BUTTONS */
    div.stButton > button {
        background-color: #333;
        color: white;
        border-radius: 0px;
        border: none;
    }

    /* SLIDERS */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #00FFAA !important; } 
    div.stSlider > div[data-baseweb="slider"] > div > div > div { background-color: #FFFFFF !important; }

    /* METRICS CARDS (Carrées et sobres) */
    div[data-testid="stMetric"] { 
        background-color: #1A1A1A; 
        border: 1px solid #333; 
        border-radius: 0px; /* Pas d'arrondi */
        padding: 15px; 
    }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px;}
    div[data-testid="stMetricValue"] { color: #FFF; font-size: 1.6rem; font-weight: 700; font-family: 'Arial', sans-serif;}

    /* TABS (Style Bloomberg) */
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent; 
        border: none; 
        color: #666; 
        font-weight: 600; 
        border-radius: 0px;
        padding-right: 20px;
        padding-left: 20px;
    }
    .stTabs [aria-selected="true"] { 
        color: #00FFAA; 
        border-bottom: 2px solid #00FFAA; 
    }

    /* NETTOYAGE */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- FONCTIONS DATA ---
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    try:
        if not tickers: return pd.DataFrame()
        # Téléchargement
        df = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()
        
        # Logique d'extraction (Single vs Multi ticker)
        if len(tickers) == 1:
            t = tickers[0]
            # Vérif si MultiIndex ou pas (Yahoo change souvent)
            if isinstance(df.columns, pd.MultiIndex): 
                prices[t] = df['Close']
            elif 'Close' in df.columns: 
                prices[t] = df['Close']
        else:
            for t in tickers:
                if t in df.columns:
                    try: prices[t] = df[t]['Close']
                    except: pass
                elif isinstance(df.columns, pd.MultiIndex) and t in df.columns.levels[0]:
                     prices[t] = df[t]['Close']
        
        return prices.fillna(method='ffill').dropna()
    except Exception as e:
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
    <span class="title-text">BLK</span><span class="accent-dot">.</span><span class="title-text">ALADDIN</span>
    <div class="subtitle">Institutional Risk & Allocation System</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### DATA FEED")
    tickers_input = st.text_input("Tickers (Yahoo)", "BTC-USD, SPY, NVDA, TLT")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### PERIOD")
    # LE VRAI CALENDRIER
    date_range = st.date_input(
        "Select Range",
        value=(datetime.now() - timedelta(days=365*2), datetime.now()),
        format="DD/MM/YYYY"
    )

# --- CORE LOGIC ---
if len(tickers) > 0 and isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    data = get_data(tickers, start_d, end_d)
    
    if not data.empty:
        returns = data.pct_change().dropna()
        
        # --- TABS NAVIGATION ---
        tab1, tab2, tab3 = st.tabs(["ALLOCATION", "CORRELATION", "BACKTEST"])
        
        # === 1. PORTFOLIO ALLOCATION ===
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 3])
            with c1:
                mode = st.radio("Mode", ["Equal Weight", "Max Sharpe"], label_visibility="collapsed")
            
            # Calcul Poids
            if mode == "Max Sharpe" and len(data.columns) >= 2:
                weights = optimize(returns)
            else:
                weights = [1/len(data.columns)] * len(data.columns)
            
            # Performance Portefeuille
            port_ret = returns.dot(weights)
            cum_port = (1 + port_ret).cumprod() * 100
            
            # Metrics
            total_ret = cum_port.iloc[-1]/100 - 1
            vol = port_ret.std() * np.sqrt(252)
            sharpe = total_ret / vol if vol > 0 else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Return", f"{total_ret:.2%}")
            k2.metric("Sharpe Ratio", f"{sharpe:.2f}")
            k3.metric("Annual Vol", f"{vol:.2%}")
            
            # Graphique Ligne (Style Terminal)
            fig = go.Figure()
            # Portefeuille (Blanc/Vert)
            fig.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port, name="PORTFOLIO", 
                line=dict(color='#00FFAA', width=2)
            ))
            # Benchmark Assets (Gris sombre)
            for col in data.columns:
                asset_curve = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=asset_curve.index, y=asset_curve, name=col, 
                    line=dict(color='#444', width=1), opacity=0.7
                ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(family="Arial", color='#888'),
                xaxis=dict(showgrid=False, linecolor='#333'), 
                yaxis=dict(showgrid=True, gridcolor='#222'), 
                margin=dict(l=0,r=0,t=20,b=0),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        # === 2. MATRICE DE CORRELATION ===
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            corr = returns.corr()
            
            # Heatmap propre
            fig_corr = px.imshow(
                corr, 
                text_auto=".2f",
                color_continuous_scale=[[0, 'black'], [1, '#00FFAA']], # Noir vers Vert Terminal
                aspect="auto"
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", color='#EEE'),
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # === 3. BACKTEST QUANT (ENGINE) ===
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns(3)
            asset_bt = col_b1.selectbox("Asset", tickers)
            sma_s = col_b2.number_input("Short MA", 5, 50, 20)
            sma_l = col_b3.number_input("Long MA", 50, 200, 50)
            
            if asset_bt:
                # Appel Moteur
                bt = VectorBacktester(data[asset_bt])
                res = bt.run_strategy(sma_s, sma_l)
                met = RiskMetrics.get_metrics(res['Strat_Ret'])
                
                # KPIs Backtest
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Strategy Ret", f"{(res['Curve'].iloc[-1]/100 - 1):.2%}")
                b2.metric("Max Drawdown", f"{met['MaxDD']:.2%}")
                b3.metric("Sharpe", f"{met['Sharpe']:.2f}")
                b4.metric("VaR 95%", f"{met['VaR']:.2%}")
                
                # Chart Backtest
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['Curve'], name="STRATEGY", line=dict(color='#00FFAA', width=2)))
                fig_bt.add_trace(go.Scatter(x=res.index, y=res['BH_Curve'], name="BUY & HOLD", line=dict(color='#666', width=1, dash='dot')))
                
                fig_bt.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(family="Arial", color='#888'),
                    xaxis=dict(showgrid=False, linecolor='#333'), 
                    yaxis=dict(showgrid=True, gridcolor='#222'),
                    margin=dict(l=0,r=0,t=20,b=0),
                    height=450
                )
                st.plotly_chart(fig_bt, use_container_width=True)

    else:
        st.warning("⚠️ Waiting for data stream...")
else:
    st.info("System Ready. Select date range.")
