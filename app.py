import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. CONFIGURATION VISUELLE (DARK MODE PRO) ---
st.set_page_config(page_title="Quant Dashboard", layout="wide", page_icon="⚡")

# Injection CSS pour le look "Hedge Fund"
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    div[data-testid="stMetric"] {
        background-color: #1f2937; border: 1px solid #374151;
        padding: 15px; border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: #9ca3af; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { color: #f3f4f6; font-size: 1.8rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- 2. MOTEUR QUANTITATIF ---

@st.cache_data(ttl=3600)
def get_data(tickers, start_date):
    try:
        df = yf.download(tickers, start=start_date, progress=False)['Adj Close']
        if isinstance(df, pd.Series): df = df.to_frame()
        return df.dropna()
    except: return pd.DataFrame()

def get_kpis(returns, risk_free=0.0):
    # Performance Cumulée
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    
    # CAGR & Volatilité
    n_years = len(returns) / 252
    cagr = (total_ret + 1)**(1/n_years) - 1 if n_years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    
    # Sharpe & Sortino
    sharpe = (cagr - risk_free) / vol if vol > 0 else 0
    neg_ret = returns[returns < 0]
    down_vol = neg_ret.std() * np.sqrt(252)
    sortino = (cagr - risk_free) / down_vol if down_vol > 0 else 0
    
    # Drawdown
    running_max = cum.cummax()
    dd = (cum / running_max) - 1
    max_dd = dd.min()
    
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": max_dd}

def optimize_portfolio(returns):
    n = len(returns.columns)
    def neg_sharpe(w):
        ret = np.sum(returns.mean() * w) * 252
        vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        return -ret/vol if vol > 0 else 0
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=constraints)
    return res.x

# --- 3. DASHBOARD ---

st.title("⚡ QUANT DASHBOARD : Arbitrage & Levier")
st.markdown("Analyse temps réel : **Spot vs Levier** (Optimisation Markowitz)")

# Sidebar
with st.sidebar:
    st.header("Paramètres")
    tickers = st.text_input("Tickers (Yahoo Finance)", "PUST.PA, LQQ.PA").upper().split(',')
    tickers = [t.strip() for t in tickers]
    years = st.slider("Historique (Années)", 1, 10, 3)
    
    st.divider()
    mode = st.radio("Stratégie", ["Manuel", "Optimisation IA (Max Sharpe)"])
    
    weights = []
    if mode == "Manuel":
        w = st.slider(f"Poids {tickers[0]}", 0, 100, 50)
        weights = [w/100, 1-(w/100)]
        st.write(f"Allocation : {weights[0]:.0%} / {weights[1]:.0%}")

# Main Calculation
start = datetime.now() - timedelta(days=years*365)
data = get_data(tickers, start)

if not data.empty and len(data.columns) > 1:
    returns = data.pct_change().dropna()
    
    if mode == "Optimisation IA (Max Sharpe)":
        with st.spinner("Optimisation en cours..."):
            weights = optimize_portfolio(returns)
        st.sidebar.success(f"Optimisé : {weights[0]:.1%} / {weights[1]:.1%}")
    
    # Calculs finaux
    port_ret = returns.dot(weights)
    bench_ret = returns.iloc[:, 1] # Benchmark = 2ème ticker (LQQ)
    
    stats_p = get_kpis(port_ret)
    stats_b = get_kpis(bench_ret)
    
    # --- VISUALISATION ---
    
    # 1. KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{stats_p['CAGR']:.2%}", f"{stats_p['CAGR']-stats_b['CAGR']:.2%} vs Bench")
    c2.metric("Sharpe", f"{stats_p['Sharpe']:.2f}", f"{stats_p['Sharpe']-stats_b['Sharpe']:.2f}")
    c3.metric("Volatilité", f"{stats_p['Vol']:.2%}", f"{stats_p['Vol']-stats_b['Vol']:.2%}", delta_color="inverse")
    c4.metric("Max Drawdown", f"{stats_p['MaxDD']:.2%}", f"{stats_p['MaxDD']-stats_b['MaxDD']:.2%}", delta_color="inverse")

    # 2. Graphiques
    tab1, tab2 = st.tabs(["📈 Performance & Drawdown", "🔗 Corrélations"])
    
    with tab1:
        # Equity Curve
        cum_p = (1 + port_ret).cumprod() * 100
        cum_b = (1 + bench_ret).cumprod() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_b.index, y=cum_b, name="Benchmark", line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=cum_p.index, y=cum_p, name="Mon Portefeuille", fill='tozeroy', line=dict(color='#00d2ff', width=2)))
        fig.update_layout(title="Performance Base 100", template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown
        rmax = cum_p.cummax()
        dd = (cum_p / rmax) - 1
        fig_dd = px.area(dd, title="Drawdown (Profondeur des pertes)", color_discrete_sequence=['#ef4444'])
        fig_dd.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        # Corrélation Roulante
        roll_corr = returns.iloc[:,0].rolling(30).corr(returns.iloc[:,1])
        fig_corr = px.line(roll_corr, title="Corrélation Glissante (30 jours)")
        fig_corr.update_traces(line_color='#f59e0b')
        fig_corr.update_layout(template="plotly_dark", yaxis_range=[-1, 1])
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("Si la corrélation baisse, c'est le moment idéal pour l'arbitrage.")

else:
    st.error("Erreur de données. Vérifiez les tickers ou attendez quelques secondes.")
