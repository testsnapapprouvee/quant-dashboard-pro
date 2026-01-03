import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Modules
from modules.backtest_engine import VectorizedBacktester
from modules.optimization import SmartOptimizer
from modules.analytics import AnalyticsEngine

# --- CONFIGURATION ---
st.set_page_config(page_title="Quant.Architect | Pro", layout="wide", page_icon="⚡")

# --- CSS LUXURY ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; color: #E0E0E0; }
    
    /* Metrics */
    div[data-testid="stMetric"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px; }
    div[data-testid="stMetricLabel"] { color: #8B949E; font-size: 0.8rem; }
    div[data-testid="stMetricValue"] { color: #F0F6FC; font-weight: 700; }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar { display: none !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #30363D; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; border: none; background: transparent; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; font-weight: 600; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #30363D; }
    
    .header-title { font-size: 32px; font-weight: 800; background: -webkit-linear-gradient(45deg, #A855F7, #6366F1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER ROBUSTE ---
@st.cache_data(ttl=3600)
def get_market_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    
    clean_tickers = [t.strip().upper() for t in tickers]
    data_map = {}
    
    for t in clean_tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty:
                df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if not df.empty:
                # Gestion MultiIndex vs Flat
                col = 'Close' if 'Close' in df.columns else df.columns[0]
                if isinstance(df.columns, pd.MultiIndex):
                    # Si yfinance renvoie un multiindex, on essaie d'aplatir
                    try: s = df['Close'] # Parfois ça marche directement
                    except: s = df.iloc[:, 0]
                else:
                    s = df[col]
                
                s.name = t
                data_map[t] = s
        except Exception as e:
            continue
            
    if len(data_map) >= 2:
        df_final = pd.concat(data_map.values(), axis=1)
        # On suppose Ordre : Risk (X2) puis Safe (X1)
        cols = df_final.columns
        df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
        return df_final.ffill().dropna()
        
    return pd.DataFrame()

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("### 🏛️ ASSET UNIVERSE")
    presets = {
        "Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"],
        "S&P 500 (US)": ["SSO", "SPY"],
        "Custom": []
    }
    preset = st.selectbox("Sélection", list(presets.keys()))
    
    if preset == "Custom":
        t_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = t_input.split(',')
    else:
        tickers = presets[preset]
        st.caption(f"Risk: {tickers[0]} | Safe: {tickers[1]}")
        
    st.markdown("---")
    st.markdown("### ⚙️ PARAMÈTRES (Live)")
    
    if 'params' not in st.session_state:
        st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30, 
                                      'allocPrudence': 50, 'allocCrash': 100, 
                                      'rollingWindow': 60, 'confirm': 2}

    # Sliders connectés au session_state pour update via Optimizer
    p = st.session_state['params']
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(p['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 40, int(p['panic']), 1)
    recov = st.slider("Recovery (%)", 10, 60, int(p['recovery']), 5)
    
    with st.expander("Avancé (Allocation & Coûts)"):
        a_prud = st.slider("Alloc Prudence (X1%)", 0, 100, int(p['allocPrudence']), 10)
        a_crash = st.slider("Alloc Crash (X1%)", 0, 100, int(p['allocCrash']), 10)
        confirm = st.slider("Confirmation (Jours)", 1, 5, int(p['confirm']))
        cost = st.number_input("Frais Transaction (%)", 0.0, 1.0, 0.1) / 100

# --- MAIN PAGE ---
st.markdown('<div class="header-title">Quant.Architect</div>', unsafe_allow_html=True)
st.caption("Institutional Backtesting Engine • Vectorized • Risk-Managed")

# 1. LOAD DATA
start_date = datetime(2015, 1, 1)
data = get_market_data(tickers, start_date, datetime.now())

if data.empty:
    st.error("❌ Données insuffisantes. Vérifiez les tickers.")
else:
    # Paramètres courants
    current_params = {
        'thresh': thresh, 'panic': panic, 'recovery': recov,
        'allocPrudence': a_prud, 'allocCrash': a_crash,
        'rollingWindow': 60, 'confirm': confirm, 'cost': cost
    }

    # 2. RUN BACKTEST
    engine = VectorizedBacktester(data, current_params)
    res_df = engine.run()
    trades = engine.get_trades()
    
    # 3. METRICS
    met_strat = AnalyticsEngine.calculate_metrics(res_df['Portfolio'])
    met_bench = AnalyticsEngine.calculate_metrics(res_df['Bench_X2']) # X2 comme benchmark agressif

    # 4. TABS UI
    tab_dash, tab_risk, tab_opti, tab_mc = st.tabs(["📊 Dashboard", "⚠️ Risk Lab", "🚀 Optimizer", "🎲 Monte Carlo"])
    
    with tab_dash:
        # KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR", f"{met_strat['CAGR']:.1f}%", delta=f"{met_strat['CAGR'] - met_bench['CAGR']:.1f}% vs Bench")
        k2.metric("Sharpe", f"{met_strat['Sharpe']:.2f}", delta=f"{met_strat['Sharpe'] - met_bench['Sharpe']:.2f}")
        k3.metric("Sortino", f"{met_strat['Sortino']:.2f}", help="Ratio rendement / volatilité baissière")
        k4.metric("Calmar", f"{met_strat['Calmar']:.2f}", help="CAGR / MaxDD")
        
        # CHART
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Portfolio'], name='Strategy', line=dict(color='#A855F7', width=2)))
        fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Bench_X2'], name='Benchmark (X2)', line=dict(color='#30363D', width=1, dash='dot')))
        
        # Color bands for regimes
        # Pour une visu propre, on peut ajouter une heatmap en dessous, mais gardons simple pour l'instant
        
        fig.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=500, margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # DRAWDOWN CHART
        st.subheader("Underwater Plot")
        dd_df = pd.DataFrame({'Strategy': res_df['Drawdown'], 'Benchmark': (res_df['Bench_X2']/res_df['Bench_X2'].cummax()-1)})
        fig_dd = px.area(dd_df, color_discrete_map={'Strategy': '#ef4444', 'Benchmark': '#30363D'})
        fig_dd.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab_risk:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Tail Risk Metrics")
            st.write(f"**Max Drawdown:** {met_strat['MaxDD']:.2f}%")
            st.write(f"**CVaR (95%):** {met_strat['CVaR_95']:.2f}% (Perte moyenne les pires jours)")
            st.write(f"**Annual Volatility:** {met_strat['Vol']:.2f}%")
        with c2:
            st.markdown("### Trade Analysis")
            st.write(f"**Total Trades:** {len(trades)}")
            if len(trades) > 0:
                last_t = trades[-1]
                st.info(f"Dernier Trade: {last_t['date'].date()} -> {last_t['regime']}")
            else:
                st.warning("Aucun trade effectué.")

    with tab_opti:
        st.markdown("### 🧠 Bayesian/Random Search Optimizer")
        obj_func = st.selectbox("Objectif", ["Calmar", "Sharpe", "CAGR"])
        
        if st.button("Lancer l'Optimisation (50 itérations)"):
            with st.spinner("Recherche de l'Alpha en cours..."):
                # On utilise les params actuels comme base
                base_p = current_params.copy()
                best_p, best_s = SmartOptimizer.run_random_search(data, n_iter=50, objective=obj_func)
                
                st.success(f"Optimisation terminée ! Meilleur Score ({obj_func}): {best_s:.2f}")
                st.json(best_p)
                
                # Bouton pour appliquer (Simulé ici par un message, en prod on update session_state)
                st.session_state['params'].update(best_p)
                st.warning("Paramètres mis à jour. Rechargez la simulation.")

    with tab_mc:
        st.markdown("### 🎲 Monte Carlo Simulation (Block Bootstrap)")
        n_sims = st.slider("Nombre de simulations", 50, 500, 100)
        
        if st.button("Lancer Monte Carlo"):
            with st.spinner("Calcul des scénarios alternatifs..."):
                sim_paths = AnalyticsEngine.monte_carlo_simulation(res_df['Portfolio'], n_sims=n_sims)
                
                # Visu
                fig_mc = go.Figure()
                # On affiche un échantillon pour ne pas surcharger
                for i in range(min(n_sims, 50)):
                    fig_mc.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(color='#A855F7', width=1, opacity=0.1), showlegend=False))
                
                # Moyenne
                mean_path = np.mean(sim_paths, axis=1)
                fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Average Path', line=dict(color='#FFFFFF', width=2)))
                
                fig_mc.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title=f"Projection sur 252 jours ({n_sims} runs)")
                st.plotly_chart(fig_mc, use_container_width=True)
