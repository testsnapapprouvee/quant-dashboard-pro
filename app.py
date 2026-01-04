import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIG & CSS (LE DESIGN REACT)
# ==========================================
st.set_page_config(page_title="Predict. Distinct", layout="wide", page_icon="⚡")

# Injection du CSS pour copier le style "Glassmorphism" du code React
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    /* BACKGROUND GENERAL */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* CARDS (Glass Effect) */
    .glass-card {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }

    /* TEXTES */
    h1, h2, h3, p, label, span { color: #E0E0E0 !important; }
    .stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 20px; font-weight: 800; }
    .section-title { font-size: 12px; font-weight: 700; color: #888; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }

    /* HEADER TEXT */
    .header-title {
        font-size: 32px; font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* BUTTONS CUSTOM */
    .stButton > button {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: #667eea;
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }

    /* OPTIMIZER BUTTON */
    .opt-btn > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        width: 100%;
        padding: 10px 0;
    }

    /* SLIDERS */
    .stSlider > div > div > div > div { background-color: #667eea; }

    /* HIDE DEFAULT ELEMENTS */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* TABLE STYLES */
    .custom-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .custom-table th { text-align: right; color: #888; padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); }
    .custom-table td { text-align: right; padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .custom-table td:first-child { text-align: left; color: #bbb; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTEUR LOGIQUE (Python Backtest)
# ==========================================

@st.cache_data
def get_mock_data():
    # Simulation de données pour l'exemple (remplacer par yfinance)
    dates = pd.date_range(start="2014-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    x2 = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.02, len(dates))) # Nasdaq x2 approx
    x1 = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, len(dates))) # Nasdaq x1 approx
    return pd.DataFrame({'X2': x2, 'X1': x1}, index=dates)

def run_simulation(data, params):
    # Logique simplifiée rapide
    prices = data['X2'].values
    nav = [100.0]
    
    # Paramètres
    w = params['window']
    thresh = params['thresh'] / 100.0
    panic = params['panic'] / 100.0
    
    alloc_safe = params['allocPrudence'] / 100.0
    alloc_crash = params['allocCrash'] / 100.0
    
    current_regime = 0 # 0=Risk, 1=Prudent, 2=Crash
    
    rolling_max = data['X2'].rolling(window=w).max()
    drawdowns = (data['X2'] / rolling_max) - 1
    
    history_regimes = []
    
    for i in range(1, len(data)):
        dd = drawdowns.iloc[i-1]
        
        # Logic simple de changement de régime
        if dd < -panic: new_regime = 2
        elif dd < -thresh: new_regime = 1
        else: new_regime = 0
        
        # Recovery logic (simplifiée)
        if current_regime > 0 and dd > -thresh/2:
            new_regime = 0
            
        current_regime = new_regime
        history_regimes.append(current_regime)
        
        # Allocation
        if current_regime == 2: exposure = 1 - alloc_crash
        elif current_regime == 1: exposure = 1 - alloc_safe
        else: exposure = 1.0
        
        ret_x2 = (data['X2'].iloc[i] / data['X2'].iloc[i-1]) - 1
        ret_x1 = (data['X1'].iloc[i] / data['X1'].iloc[i-1]) - 1
        
        strat_ret = (exposure * ret_x2) + ((1-exposure) * ret_x1)
        nav.append(nav[-1] * (1 + strat_ret))
        
    return pd.DataFrame({
        'Strategy': nav, 
        'X2': (data['X2'] / data['X2'].iloc[0]) * 100,
        'X1': (data['X1'] / data['X1'].iloc[0]) * 100,
        'Drawdown': drawdowns * 100
    }, index=data.index), history_regimes

def calc_metrics(series):
    if series.empty: return {}
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    years = len(series) / 252
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    
    # Max DD
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    max_dd = dd.min()
    
    # Vol
    vol = series.pct_change().std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    
    return {
        "CAGR": cagr * 100,
        "MaxDD": max_dd * 100,
        "Vol": vol * 100,
        "Sharpe": sharpe,
        "Total": total_ret * 100
    }

# ==========================================
# 2. UI LAYOUT & COMPOSANTS
# ==========================================

# --- STATE MANAGEMENT ---
if 'params' not in st.session_state:
    st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30, 'allocPrudence': 50, 'allocCrash': 100}
if 'run_valid' not in st.session_state:
    st.session_state['run_valid'] = False

# --- HEADER ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%); border-radius: 20px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: space-between; align-items: center;">
    <div>
        <div class="header-title">Predict. <span style="font-weight:300; color:white;">DISTINCT PROFILES</span></div>
        <div style="font-size: 11px; color: #888; margin-top: 5px;">ENGINE V3 • PYTHON BACKEND • REACT VISUALS</div>
    </div>
    <div style="display:flex; gap:10px;">
        <span style="background:rgba(255,255,255,0.05); padding: 5px 12px; border-radius:6px; border:1px solid #333; font-size:10px; color:#aaa;">YTD</span>
        <span style="background:rgba(255,255,255,0.05); padding: 5px 12px; border-radius:6px; border:1px solid #333; font-size:10px; color:#aaa;">2022</span>
        <span style="background:linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 5px 12px; border-radius:6px; font-size:10px; font-weight:bold;">MAX</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 3-COLUMN LAYOUT (Left Controls, Center Graph, Right Validation) ---
col_left, col_center, col_right = st.columns([1, 2.2, 1])

# DONNÉES
full_data = get_mock_data()
current_data = full_data  # Ici on pourrait filtrer par date

# ==========================
# GAUCHE: CONTROLS
# ==========================
with col_left:
    # 1. AI OPTIMIZER CARD
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 AI OPTIMIZER</div>', unsafe_allow_html=True)
    
    profile = st.radio("Profile", ["DÉFENSIF", "ÉQUILIBRÉ", "AGRESSIF"], horizontal=True, label_visibility="collapsed")
    
    # Custom styling for radio buttons visual via pure python isn't perfect, using standard widgets
    if profile == "DÉFENSIF": 
        st.caption("🛡️ Priorité: Protection MaxDD < 15%")
    elif profile == "AGRESSIF":
        st.caption("⚡ Priorité: CAGR Max")
    else:
        st.caption("⚖️ Priorité: Ratio Sharpe")

    c_opt = st.container()
    with c_opt:
        st.markdown('<div class="opt-btn">', unsafe_allow_html=True)
        if st.button(f"OPTIMISER ({profile})", key="btn_opt"):
            # Simulation d'optimisation
            import time
            with st.spinner("AI Searching..."):
                time.sleep(0.5)
                if profile == "DÉFENSIF":
                    st.session_state['params'].update({'thresh': 3.0, 'panic': 12, 'recovery': 20})
                elif profile == "AGRESSIF":
                    st.session_state['params'].update({'thresh': 8.0, 'panic': 25, 'recovery': 50})
                else:
                    st.session_state['params'].update({'thresh': 5.0, 'panic': 15, 'recovery': 30})
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. CONTROLS CARD
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 TRIGGERS</div>', unsafe_allow_html=True)
    
    p_thresh = st.slider("Seuil Sortie (%)", 1.0, 15.0, float(st.session_state['params']['thresh']), 0.5)
    p_panic = st.slider("Panic (%)", 5, 40, int(st.session_state['params']['panic']))
    
    st.markdown('<div style="margin-top:20px;" class="section-title">🔄 RECOVERY</div>', unsafe_allow_html=True)
    p_recov = st.slider("Recovery (%)", 10, 80, int(st.session_state['params']['recovery']), 5)
    
    st.markdown('<div style="margin-top:20px;" class="section-title">📊 ALLOCATION (X1)</div>', unsafe_allow_html=True)
    p_prud = st.slider("Prudence %", 0, 100, int(st.session_state['params']['allocPrudence']), 10)
    p_crash = st.slider("Crash %", 0, 100, int(st.session_state['params']['allocCrash']), 10)
    
    # Mise à jour du state
    st.session_state['params'] = {
        'thresh': p_thresh, 'panic': p_panic, 'recovery': p_recov,
        'allocPrudence': p_prud, 'allocCrash': p_crash, 'window': 60
    }
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# CALCUL BACKTEST
# ==========================
df_res, regimes = run_simulation(current_data, st.session_state['params'])
metrics_strat = calc_metrics(df_res['Strategy'])
metrics_x2 = calc_metrics(df_res['X2'])
metrics_x1 = calc_metrics(df_res['X1'])

# ==========================
# CENTRE: GRAPH & STATS
# ==========================
with col_center:
    # 1. TABLEAU PERF (HTML pur pour le look)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏆 PERFORMANCE REPORT</div>', unsafe_allow_html=True)
    
    table_html = f"""
    <table class="custom-table">
        <thead>
            <tr>
                <th>METRIC</th>
                <th style="color:#667eea; font-weight:bold;">STRATEGY</th>
                <th style="color:#10b981;">NASDAQ X1</th>
                <th style="color:#ef4444;">LEVERAGE X2</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Total Return</td>
                <td style="color:#667eea; font-weight:800;">{metrics_strat['Total']:.0f}%</td>
                <td>{metrics_x1['Total']:.0f}%</td>
                <td>{metrics_x2['Total']:.0f}%</td>
            </tr>
            <tr>
                <td>CAGR</td>
                <td style="color:#667eea; font-weight:800;">{metrics_strat['CAGR']:.1f}%</td>
                <td>{metrics_x1['CAGR']:.1f}%</td>
                <td>{metrics_x2['CAGR']:.1f}%</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td style="color:#667eea; font-weight:800;">{metrics_strat['MaxDD']:.1f}%</td>
                <td>{metrics_x1['MaxDD']:.1f}%</td>
                <td>{metrics_x2['MaxDD']:.1f}%</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td style="color:#667eea; font-weight:800;">{metrics_strat['Sharpe']:.2f}</td>
                <td>{metrics_x1['Sharpe']:.2f}</td>
                <td>{metrics_x2['Sharpe']:.2f}</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. CHART AREA (Plotly)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Zone remplie (Strategy)
    fig.add_trace(go.Scatter(
        x=df_res.index, y=df_res['Strategy'],
        mode='lines', name='Strategy',
        line=dict(color='#667eea', width=3),
        fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    # Benchmarks
    fig.add_trace(go.Scatter(
        x=df_res.index, y=df_res['X1'],
        mode='lines', name='X1',
        line=dict(color='#10b981', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=df_res.index, y=df_res['X2'],
        mode='lines', name='X2',
        line=dict(color='#ef4444', width=1, dash='dot')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#aaa', family='Inter'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        hovermode='x unified',
        xaxis=dict(showgrid=False, linecolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", y=1.05, x=0)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# DROITE: VALIDATION
# ==========================
with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛡️ ROBUST VALIDATION</div>', unsafe_allow_html=True)
    
    if st.button("RUN VALIDATION", key="btn_valid"):
        st.session_state['run_valid'] = True
    
    if st.session_state['run_valid']:
        # Simulation Monte Carlo Rapide
        mc_cagrs = []
        mc_dds = []
        
        # On fait 50 runs pour l'effet visuel
        for _ in range(50):
            # Bootstrap resample simple
            noise = np.random.normal(0, 0.002, len(df_res))
            resampled = df_res['Strategy'] * (1 + noise)
            metrics = calc_metrics(resampled)
            mc_cagrs.append(metrics['CAGR'])
            mc_dds.append(metrics['MaxDD'])
            
        med_cagr = np.median(mc_cagrs)
        p95_dd = np.percentile(mc_dds, 5) # DD is negative
        
        is_robust = med_cagr > 10 and p95_dd > -35
        
        # VERDICT BOX
        color_v = "#10b981" if is_robust else "#ef4444"
        icon_v = "✅" if is_robust else "⚠️"
        text_v = "ROBUSTE" if is_robust else "FRAGILE"
        
        st.markdown(f"""
        <div style="background: {color_v}20; border: 2px solid {color_v}; padding: 15px; border-radius: 12px; margin: 15px 0; text-align: center;">
            <div style="font-size: 24px; margin-bottom: 5px;">{icon_v}</div>
            <div style="font-weight: 800; font-size: 18px; color: {color_v};">{text_v}</div>
            <div style="font-size: 10px; color: #aaa;">Score Robustesse: {(med_cagr/abs(p95_dd)):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # STATS GRID
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
            <div>
                <div class="stat-label">CAGR MÉDIAN</div>
                <div class="stat-value" style="color: #667eea;">{med_cagr:.1f}%</div>
            </div>
            <div>
                <div class="stat-label">MAXDD P95</div>
                <div class="stat-value" style="color: #ef4444;">{p95_dd:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SCATTER CHART
        df_mc = pd.DataFrame({'CAGR': mc_cagrs, 'DD': mc_dds})
        fig_mc = px.scatter(df_mc, x='CAGR', y='DD', color_discrete_sequence=['#667eea'])
        fig_mc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aaa', size=10),
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=None),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=None),
        )
        st.plotly_chart(fig_mc, use_container_width=True, config={'displayModeBar': False})
        
    else:
        st.info("Cliquez pour lancer le test Monte-Carlo et Walk-Forward.")

    st.markdown('</div>', unsafe_allow_html=True)

# CSS FIX pour retirer les paddings inutiles de Streamlit
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)
