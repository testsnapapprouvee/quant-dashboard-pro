import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIGURATION & IMPORTS DES MODULES
# ==========================================
st.set_page_config(page_title="Predict. Distinct", layout="wide", page_icon="⚡")

# On garde tes modules existants
MODULES_STATUS = {"Risk": False, "Leverage": False, "Arbitrage": False}

try:
    from modules.risk_metrics import RiskMetrics
    MODULES_STATUS["Risk"] = True
except ImportError:
    class RiskMetrics:
        @staticmethod
        def get_full_risk_profile(series): return {}

try:
    from modules.leverage_diagnostics import LeverageDiagnostics
    MODULES_STATUS["Leverage"] = True
except ImportError:
    class LeverageDiagnostics:
        @staticmethod
        def calculate_realized_beta(data, window=21): return pd.DataFrame()
        @staticmethod
        def calculate_leverage_health(data): return {}
        @staticmethod
        def detect_decay_regime(data, window=60): return pd.DataFrame()

try:
    from modules.arbitrage_signals import ArbitrageSignals
    MODULES_STATUS["Arbitrage"] = True
except ImportError:
    class ArbitrageSignals:
        @staticmethod
        def calculate_relative_strength(data, window=20): return pd.DataFrame()
        @staticmethod
        def get_signal_status(series): return {}

# ==========================================
# 1. CSS SILENT LUXURY (TON DESIGN EXACT)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0A0A0F; font-family: 'Inter', sans-serif; color: #E0E0E0; }
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    
    /* HEADER */
    .header-container {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A3E 100%);
        border-radius: 12px; padding: 25px; 
        border: 1px solid rgba(255,255,255,0.08); 
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .title-text { font-weight: 800; font-size: 32px; letter-spacing: -1px; color: #FFFFFF; }
    .title-dot { color: #A855F7; font-size: 32px; font-weight: 800; }
    
    /* TABLEAUX */
    table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Inter'; }
    th { text-align: left; color: #aaa; background-color: #1E1E2E; padding: 10px; border-bottom: 1px solid #333; }
    td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #E0E0E0; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; font-weight: 600; }
    
    /* CARDS */
    .glass-card { 
        background: rgba(30, 30, 46, 0.6); 
        border-radius: 12px; padding: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        margin-bottom: 20px; backdrop-filter: blur(10px);
    }
    
    /* BUTTONS */
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1E1E2E; color: #A855F7; border: 1px solid #A855F7; transition: all 0.3s; }
    .stButton > button:hover { background-color: #A855F7; color: white; border: 1px solid #A855F7; }
    
    header, footer { visibility: hidden; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTEUR VECTORISÉ (OPTIMISÉ PRODUCTION)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        # Préparation rapide (Numpy est plus rapide que Pandas dans les boucles si nécessaire)
        dates = data.index
        # On s'assure d'avoir des Arrays Numpy pour la vitesse
        px_x2 = data['X2'].values
        px_x1 = data['X1'].values
        
        # Paramètres
        w = int(params['rollingWindow'])
        thresh = -params['thresh'] / 100.0
        panic = -params['panic'] / 100.0
        recov_factor = params['recovery'] / 100.0
        confirm = int(params['confirm'])
        
        alloc_crash = params['allocCrash'] / 100.0
        alloc_prud = params['allocPrudence'] / 100.0
        cost_rate = params.get('cost', 0.001)

        # 1. Calculs Vectorisés (Indicateurs)
        # Rolling Max pour Drawdown
        # Astuce : On utilise pandas pour le rolling max car c'est optimisé en C
        roll_max = data['X2'].rolling(w, min_periods=1).max().values
        dd = (px_x2 / roll_max) - 1.0
        
        # 2. Boucle de Régime (Nécessaire pour la logique de Recovery/Hystérésis)
        # Mais on l'optimise pour qu'elle soit très rapide
        n = len(data)
        regimes = np.zeros(n, dtype=int) # 0: Offensif, 1: Prudence, 2: Crash
        
        curr_reg = 0
        peak = px_x2[0]
        trough = px_x2[0]
        pending = 0
        conf_count = 0
        
        # Variables pour éviter les lookups couteux
        for i in range(1, n):
            price = px_x2[i]
            cur_dd = dd[i]
            
            # Détermination cible
            target = curr_reg
            
            if curr_reg != 2: # Si pas en crash
                if cur_dd <= panic: target = 2
                elif cur_dd <= thresh: target = 1
                else: target = 0
            
            # Logique Recovery (Si en Prudence ou Crash)
            if curr_reg in [1, 2]:
                if price < trough: trough = price
                # Seuil de recovery dynamique
                recov_price = trough + (peak - trough) * recov_factor
                
                if price >= recov_price:
                    target = 0 # Retour offensif
                else:
                    # On peut s'aggraver, mais pas s'améliorer tant qu'on a pas touché le recov price
                    if cur_dd <= panic: target = 2
                    elif cur_dd <= thresh and curr_reg != 2: target = 1
            else:
                peak = roll_max[i] # Update du peak seulement si on est R0
                trough = price

            # Confirmation
            if target == pending:
                conf_count += 1
            else:
                pending = target
                conf_count = 0
            
            if conf_count >= confirm and pending != curr_reg:
                curr_reg = pending
                conf_count = 0
                # Reset trough/peak si on sort de R0
                if curr_reg != 0:
                    peak = roll_max[i]
                    trough = price
            
            regimes[i] = curr_reg

        # 3. Allocation & Performance (Vectorisé)
        # On crée un masque d'allocation
        # R0 -> 0% X1, R1 -> alloc_prud, R2 -> alloc_crash
        alloc_x1 = np.where(regimes == 2, alloc_crash, np.where(regimes == 1, alloc_prud, 0.0))
        
        # Shift des allocations (On trade à la cloture, appliqué le lendemain)
        # alloc_x1[i] est décidé à la fin de i, appliqué pour le rendement de i+1
        alloc_x1 = np.roll(alloc_x1, 1)
        alloc_x1[0] = 0.0 # Pas d'alloc jour 0
        alloc_x2 = 1.0 - alloc_x1
        
        # Calcul rendements
        ret_x2 = np.zeros_like(px_x2)
        ret_x1 = np.zeros_like(px_x1)
        # Eviter division par zero
        ret_x2[1:] = (px_x2[1:] / px_x2[:-1]) - 1
        ret_x1[1:] = (px_x1[1:] / px_x1[:-1]) - 1
        
        # Coûts de transaction (Changement d'alloc * coût)
        delta_alloc = np.abs(np.diff(alloc_x1, prepend=0))
        costs = delta_alloc * cost_rate
        
        # Rendement Stratégie
        strat_ret = (alloc_x2 * ret_x2) + (alloc_x1 * ret_x1) - costs
        
        # Courbes (Base 100)
        curve_strat = 100 * np.cumprod(1 + strat_ret)
        curve_x2 = 100 * np.cumprod(1 + ret_x2)
        curve_x1 = 100 * np.cumprod(1 + ret_x1)
        
        # DataFrame Final
        df_res = pd.DataFrame({
            'portfolio': curve_strat,
            'benchX2': curve_x2,
            'benchX1': curve_x1,
            'regime': regimes,
            'drawdown': dd
        }, index=dates)
        
        # Extraction des trades pour l'affichage (non vectorisé mais court)
        trades = []
        regime_changes = np.where(np.diff(regimes) != 0)[0] + 1
        labels = {0: "OFFENSIF", 1: "PRUDENCE", 2: "CRASH"}
        
        for idx in regime_changes:
            if idx < len(dates):
                trades.append({
                    'date': dates[idx],
                    'to': regimes[idx], # Int
                    'label': labels[regimes[idx]]
                })
                
        return df_res, trades

# ==========================================
# 3. OPTIMISEUR RÉEL (RANDOM SEARCH)
# ==========================================
class SmartOptimizer:
    @staticmethod
    def run(data, profile, current_params):
        # Espace de recherche (Bounds)
        # On teste 50 combinaisons aléatoires intelligentes
        n_iter = 50
        results = []
        
        best_score = -np.inf
        best_p = current_params.copy()
        
        for _ in range(n_iter):
            # Génération aléatoire autour des valeurs actuelles ou large
            t = np.random.uniform(2.0, 10.0)
            p = np.random.uniform(t + 5.0, 35.0) # Panic doit être > Thresh
            r = np.random.choice([20, 30, 40, 50, 60])
            
            test_p = current_params.copy()
            test_p.update({'thresh': t, 'panic': p, 'recovery': r})
            
            # Simulation rapide
            res, _ = BacktestEngine.run_simulation(data, test_p)
            met = calculate_metrics(res['portfolio'])
            
            # Fonction Objectif
            score = 0
            if profile == "DEFENSIVE": score = met['Calmar']
            elif profile == "BALANCED": score = met['Sharpe']
            elif profile == "AGGRESSIVE": score = met['CAGR'] if met['MaxDD'] > -40 else -1000
            
            if score > best_score:
                best_score = score
                best_p = test_p
        
        return best_p, best_score

# ==========================================
# 4. DATA ENGINE (BULLDOZER) & UTILS
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    price_map = {}
    clean = [t.strip().upper() for t in tickers]
    
    for t in clean:
        try:
            d = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if d.empty: d = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            
            if not d.empty:
                if 'Close' in d.columns: s = d['Close']
                elif 'Adj Close' in d.columns: s = d['Adj Close']
                else: s = d.iloc[:, 0]
                price_map[t] = s
        except: continue

    if len(price_map) >= 2:
        df = pd.concat(price_map.values(), axis=1)
        cols = df.columns
        # On renomme X2 (Risk) et X1 (Safe)
        if len(cols) >= 2:
            df.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
            return df.ffill().dropna()
    return pd.DataFrame()

def calculate_metrics(series):
    if series.empty: return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}
    tot = (series.iloc[-1] / series.iloc[0]) - 1
    days = len(series)
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 2 else 0
    dd = (series / series.cummax() - 1).min()
    vol = series.pct_change().std() * np.sqrt(252)
    return {
        "Cumul": tot*100, "CAGR": cagr*100, "MaxDD": dd*100, "Vol": vol*100,
        "Sharpe": cagr/vol if vol>0 else 0, "Calmar": cagr/abs(dd) if dd!=0 else 0
    }

# ==========================================
# 5. INTERFACE UTILISATEUR (TON LAYOUT)
# ==========================================

# HEADER
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span class="title-text">Predict</span><span class="title-dot">.</span>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.0 • SILENT LUXURY • INSTITUTIONAL</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(168, 85, 247, 0.1); color:#A855F7; padding:5px 10px; border-radius:4px; font-size:11px; border:1px solid rgba(168, 85, 247, 0.3);">LIVE SYSTEM</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sidebar, col_main = st.columns([1, 3])

# --- SIDEBAR ---
with col_sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏛️ ASSETS")
    
    presets = {
        "Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"],
        "S&P 500 (US)": ["SSO", "SPY"],
        "MSCI World": ["CL2.PA", "CW8.PA"],
        "Custom": []
    }
    preset = st.selectbox("Universe", list(presets.keys()))
    
    if preset == "Custom":
        t_in = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [x.strip() for x in t_in.split(',')]
    else:
        tickers = presets[preset]
        st.caption(f"Risk: **{tickers[0]}** | Safe: **{tickers[1]}**")
    
    # Période
    p_opt = ["YTD", "1Y", "3YR", "5YR", "2022", "Custom"]
    per = st.selectbox("Period", p_opt, index=2)
    
    today = datetime.now()
    if per == "YTD": start_d = datetime(today.year, 1, 1)
    elif per == "1Y": start_d = today - timedelta(days=365)
    elif per == "3YR": start_d = today - timedelta(days=365*3)
    elif per == "5YR": start_d = today - timedelta(days=365*5)
    elif per == "2022": start_d = datetime(2022,1,1); end_d = datetime(2022,12,31)
    
    if per == "Custom":
        start_d = st.date_input("Start", datetime(2020,1,1))
        end_d = st.date_input("End", today)
    elif per != "2022":
        end_d = today
        
    st.markdown("---")
    st.markdown("### ⚡ PARAMS")
    
    # Session State pour Params (pour l'optimiseur)
    if 'p' not in st.session_state:
        st.session_state['p'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30, 'allocPrudence': 50, 'allocCrash': 100, 'rollingWindow': 60, 'confirm': 2}
    
    pp = st.session_state['p']
    
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(pp['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 40, int(pp['panic']), 1)
    recov = st.slider("Recovery (%)", 10, 60, int(pp['recovery']), 5)
    
    with st.expander("Avancé (Alloc & Coûts)"):
        a_prud = st.slider("Alloc Prudence (%)", 0, 100, int(pp['allocPrudence']))
        a_crash = st.slider("Alloc Crash (%)", 0, 100, int(pp['allocCrash']))
        confirm = st.slider("Confirm (Jours)", 1, 5, int(pp['confirm']))
        cost = st.number_input("Frais (%)", 0.0, 1.0, 0.1, step=0.05) / 100

    st.markdown("---")
    st.markdown("### 🧠 OPTIMIZER")
    obj = st.selectbox("Objectif", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    if st.button(f"RUN AUTO-TUNE ({obj})"):
        d_opt = get_data(tickers, start_d, end_d)
        if not d_opt.empty:
            with st.spinner("Searching Alpha..."):
                current_p = {'thresh': thresh, 'panic': panic, 'recovery': recov, 'allocPrudence': a_prud, 'allocCrash': a_crash, 'rollingWindow': 60, 'confirm': confirm, 'cost': cost}
                best, score = SmartOptimizer.run(d_opt, obj, current_p)
                st.session_state['p'] = best
                st.success(f"Optimized! Score: {score:.2f}")
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN COLUMN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty or len(data) < 10:
        st.error(f"❌ **NO DATA** for {tickers}. Check tickers or dates.")
    else:
        # Simulation
        sim_p = {'thresh': thresh, 'panic': panic, 'recovery': recov, 'allocPrudence': a_prud, 'allocCrash': a_crash, 'rollingWindow': 60, 'confirm': confirm, 'cost': cost}
        df_res, trades = BacktestEngine.run_simulation(data, sim_p)
        
        # Metrics Calcul
        m_strat = calculate_metrics(df_res['portfolio'])
        m_x2 = calculate_metrics(df_res['benchX2'])
        m_x1 = calculate_metrics(df_res['benchX1'])
        
        # Appels Modules Externes (Si présents)
        r_strat = RiskMetrics.get_full_risk_profile(df_res['portfolio']) if MODULES_STATUS["Risk"] else {}
        l_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
        a_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()
        
        # TABS
        t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "📈 Performance", "⚙️ Risk & Leverage", "🎯 Signals", "🛡️ Validation"])
        
        # --- TAB 1: DASHBOARD ---
        with t1:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Strat", f"{m_strat['CAGR']:.1f}%", delta=f"{m_strat['CAGR']-m_x2['CAGR']:.1f}% vs X2")
            k2.metric("Max Drawdown", f"{m_strat['MaxDD']:.1f}%", delta=f"{m_strat['MaxDD']-m_x2['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe", f"{m_strat['Sharpe']:.2f}", delta=f"{m_strat['Sharpe']-m_x2['Sharpe']:.2f}")
            k4.metric("Trades", len(trades))
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['portfolio'], name='STRATEGY', line=dict(color='#A855F7', width=3), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX2'], name='Risk (X2)', line=dict(color='#ef4444', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX1'], name='Safe (X1)', line=dict(color='#10b981', width=1.5, dash='dot')))
            
            # Trades markers
            for t in trades:
                c = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['portfolio'], text="▼" if t['to']!=0 else "▲", showarrow=False, font=dict(color=c, size=16))
            
            fig.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=450, margin=dict(l=40, r=40, t=20, b=40), xaxis=dict(showgrid=False, linecolor='#333'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), hovermode="x unified", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: PERFORMANCE TABLE ---
        with t2:
            st.markdown("### 🏆 Attribution")
            p_data = {
                "Metric": ["CAGR", "Vol (Ann)", "Sharpe", "MaxDD", "Calmar", "Cumul"],
                "Strategy": [f"{m_strat['CAGR']:.1f}%", f"{m_strat['Vol']:.1f}%", f"{m_strat['Sharpe']:.2f}", f"{m_strat['MaxDD']:.1f}%", f"{m_strat['Calmar']:.2f}", f"{m_strat['Cumul']:.1f}%"],
                "Risk (X2)": [f"{m_x2['CAGR']:.1f}%", f"{m_x2['Vol']:.1f}%", f"{m_x2['Sharpe']:.2f}", f"{m_x2['MaxDD']:.1f}%", f"{m_x2['Calmar']:.2f}", f"{m_x2['Cumul']:.1f}%"],
                "Safe (X1)": [f"{m_x1['CAGR']:.1f}%", f"{m_x1['Vol']:.1f}%", f"{m_x1['Sharpe']:.2f}", f"{m_x1['MaxDD']:.1f}%", f"{m_x1['Calmar']:.2f}", f"{m_x1['Cumul']:.1f}%"]
            }
            st.markdown(pd.DataFrame(p_data).style.hide(axis="index").set_properties(**{'background-color': '#0A0A0F', 'color': '#eee', 'border-color': '#333'}).to_html(), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌊 Underwater")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            dd_s = (df_res['portfolio']/df_res['portfolio'].cummax()-1)*100
            dd_x2 = (df_res['benchX2']/df_res['benchX2'].cummax()-1)*100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy', name='Strategy', line=dict(color='#A855F7', width=1), fillcolor='rgba(168, 85, 247, 0.15)'))
            fig_dd.add_trace(go.Scatter(x=dd_x2.index, y=dd_x2, name='Risk', line=dict(color='#ef4444', width=1, dash='dot')))
            fig_dd.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=250, margin=dict(t=10,b=10), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_dd, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 3: RISK & LEVERAGE ---
        with t3:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ⚠️ Risk Profile")
                if risk_s:
                    st.metric("Ulcer Index", f"{risk_s.get('Ulcer_Index', 0):.2f}")
                    st.metric("VaR 95%", f"{risk_s.get('VaR_95', 0)*100:.2f}%")
                    st.metric("CVaR 95%", f"{risk_s.get('CVaR_95', 0)*100:.2f}%")
                else: st.info("Risk Module missing")
            with c2:
                st.markdown("### ⚙️ Leverage")
                if not l_beta.empty:
                    st.metric("Realized Beta", f"{l_beta['Realized_Beta'].iloc[-1]:.2f}x")
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=l_beta.index, y=l_beta['Realized_Beta'], line=dict(color='#A855F7')))
                    fig_l.add_hline(y=2.0, line_dash="dot", line_color="white")
                    fig_l.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'), height=200, margin=dict(t=10,b=10))
                    st.plotly_chart(fig_l, use_container_width=True)
                else: st.info("Leverage Module missing")

        # --- TAB 4: SIGNALS ---
        with t4:
            if not a_sig.empty:
                curr_z = a_sig['Z_Score'].iloc[-1]
                st.metric("Z-Score", f"{curr_z:.2f}", delta="Rich" if curr_z>0 else "Cheap", delta_color="inverse")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=a_sig.index, y=a_sig['Z_Score'], line=dict(color='#3b82f6')))
                fig_z.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0)
                fig_z.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                fig_z.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'), height=300, margin=dict(t=10,b=10))
                st.plotly_chart(fig_z, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else: st.info("Arbitrage Module missing")

        # --- TAB 5: VALIDATION (MONTE CARLO) ---
        with t5:
            st.markdown("### 🎲 Monte Carlo (50 Runs)")
            if st.button("Lancer Simulation"):
                with st.spinner("Calculs..."):
                    rets = df_res['portfolio'].pct_change().dropna()
                    paths = []
                    for _ in range(50):
                        sim_r = np.random.choice(rets, size=252, replace=True)
                        paths.append(100 * np.cumprod(1 + sim_r)[-1])
                    
                    mc_res = pd.DataFrame(paths, columns=['Final'])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median Final", f"{mc_res['Final'].median():.0f}")
                    c2.metric("Worst Case", f"{mc_res['Final'].quantile(0.05):.0f}")
                    c3.metric("Prob Loss", f"{(mc_res['Final'] < 100).mean()*100:.0f}%")
                    
                    fig_mc = px.histogram(mc_res, x="Final", nbins=15, color_discrete_sequence=['#A855F7'])
                    fig_mc.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'))
                    st.plotly_chart(fig_mc, use_container_width=True)
