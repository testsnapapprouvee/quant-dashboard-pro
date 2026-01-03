import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 0. IMPORT DES MODULES (AUTO-DETECTION)
# ==========================================
MODULES_STATUS = {"Risk": False, "Leverage": False, "Arbitrage": False}

# Module 1: Risk Metrics
try:
    from modules.risk_metrics import RiskMetrics
    MODULES_STATUS["Risk"] = True
except ImportError:
    class RiskMetrics:
        @staticmethod
        def get_full_risk_profile(series): return {}

# Module 2: Leverage Diagnostics
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

# Module 3: Arbitrage Signals
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
# 1. CONFIGURATION & CSS (DESIGN SILENT LUXURY)
# ==========================================
st.set_page_config(page_title="Predict. Distinct", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

    /* BASE */
    .stApp { background-color: #0a0a0f; font-family: 'Inter', sans-serif; }
    
    /* TEXTES & TITRES */
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    
    /* HEADER */
    .header-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .title-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 32px;
    }

    /* CARDS */
    .glass-card {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    
    /* INPUTS & WIDGETS */
    .stTextInput > div > div > input { background-color: #0D1117 !important; color: white !important; border: 1px solid #333 !important; border-radius: 6px; }
    .stDateInput > div > div > input { background-color: #0D1117 !important; color: white !important; }
    div[data-baseweb="select"] > div { background-color: #0D1117 !important; color: white !important; border: 1px solid #333 !important; }
    
    /* METRICS */
    div[data-testid="stMetric"] { background-color: rgba(30,30,46,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 10px; }
    div[data-testid="stMetricLabel"] { font-size: 10px; color: #888; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #fff; font-weight: 700; }

    /* BOUTONS CUSTOM */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    
    /* REMOVE STREAMLIT UI */
    header, footer {visibility: hidden;}
    .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MOTEUR DE SIMULATION (CORE ENGINE)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        # Initialisation
        cash_x2 = 100.0
        cash_x1 = 0.0
        portfolio = 100.0
        
        current_regime = 'R0' # R0: Offensif, R1: Prudence, R2: Crash
        pending_regime = 'R0'
        confirm_count = 0
        
        price_history_x2 = []
        peak_at_crash = 0.0
        trough_x2 = 0.0
        rolling_peak = 0.0
        
        results = []
        trades = []
        
        # Paramètres
        rolling_w = int(params['rollingWindow'])
        thresh = params['thresh']
        panic = params['panic']
        recov = params['recovery']
        confirm = params['confirm']
        alloc_crash = params['allocCrash'] / 100.0
        alloc_prudence = params['allocPrudence'] / 100.0

        dates = data.index
        prices_x2 = data['X2'].values
        prices_x1 = data['X1'].values
        
        for i in range(len(data)):
            # 1. Mise à jour Portfolio
            if i > 0:
                if prices_x2[i-1] != 0: r_x2 = (prices_x2[i] - prices_x2[i-1]) / prices_x2[i-1]
                else: r_x2 = 0
                
                if prices_x1[i-1] != 0: r_x1 = (prices_x1[i] - prices_x1[i-1]) / prices_x1[i-1]
                else: r_x1 = 0

                cash_x2 *= (1 + r_x2)
                cash_x1 *= (1 + r_x1)
                portfolio = cash_x2 + cash_x1
            
            # 2. Indicateurs Techniques
            curr_price = prices_x2[i]
            price_history_x2.append(curr_price)
            if len(price_history_x2) > rolling_w:
                price_history_x2.pop(0)
            
            rolling_peak = max(price_history_x2)
            if rolling_peak == 0: rolling_peak = 1
            current_dd = ((curr_price - rolling_peak) / rolling_peak) * 100
            
            # 3. Logique de Régime
            target = current_regime
            
            if current_regime != 'R2':
                if current_dd <= -panic: target = 'R2'
                elif current_dd <= -thresh: target = 'R1'
                else: target = 'R0'
            
            if current_regime in ['R1', 'R2']:
                if curr_price < trough_x2: trough_x2 = curr_price
                recovery_target = trough_x2 + (peak_at_crash - trough_x2) * (recov / 100.0)
                
                if curr_price >= recovery_target:
                    target = 'R0'
                else:
                    if current_dd <= -panic: target = 'R2'
                    elif current_dd <= -thresh and current_regime != 'R2': target = 'R1'
            else:
                peak_at_crash = rolling_peak
                trough_x2 = curr_price

            # 4. Confirmation & Trade
            if target == pending_regime:
                confirm_count += 1
            else:
                pending_regime = target
                confirm_count = 0
                
            if confirm_count >= confirm and pending_regime != current_regime:
                old_regime = current_regime
                current_regime = pending_regime
                
                target_pct_x1 = 0.0
                label = ""
                if current_regime == 'R2': 
                    target_pct_x1 = alloc_crash
                    label = "CRASH"
                elif current_regime == 'R1': 
                    target_pct_x1 = alloc_prudence
                    label = "PRUDENCE"
                else: 
                    target_pct_x1 = 0.0
                    label = "OFFENSIF"
                
                total_val = cash_x1 + cash_x2
                cash_x1 = total_val * target_pct_x1
                cash_x2 = total_val * (1 - target_pct_x1)
                
                if current_regime != 'R0':
                    peak_at_crash = rolling_peak
                    trough_x2 = curr_price
                
                trades.append({
                    'date': dates[i],
                    'from': old_regime,
                    'to': current_regime,
                    'label': label,
                    'val': total_val
                })
                confirm_count = 0

            # 5. Enregistrement
            results.append({
                'date': dates[i],
                'portfolio': portfolio,
                'X1': prices_x1[i],
                'X2': prices_x2[i],
                'regime': current_regime,
                'drawdown': current_dd
            })
            
        df_res = pd.DataFrame(results).set_index('date')
        
        if not df_res.empty:
            start_p = df_res['portfolio'].iloc[0]
            start_x1 = df_res['X1'].iloc[0]
            start_x2 = df_res['X2'].iloc[0]
            
            df_res['portfolio'] = (df_res['portfolio'] / start_p) * 100
            df_res['benchX1'] = (df_res['X1'] / start_x1) * 100
            df_res['benchX2'] = (df_res['X2'] / start_x2) * 100
            
        return df_res, trades

# ==========================================
# 3. ANALYSES & UTILS
# ==========================================
def calculate_metrics(series):
    if series.empty: return {"CAGR":0, "MaxDD":0, "Vol":0, "Final":0}
    days = len(series)
    if days < 2: return {"CAGR":0, "MaxDD":0, "Vol":0, "Final":0}
    
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 0 else 0
    
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    pct_change = series.pct_change().dropna()
    vol = pct_change.std() * np.sqrt(252)
    
    return {
        "CAGR": cagr * 100,
        "MaxDD": max_dd * 100,
        "Vol": vol * 100,
        "Final": total_ret * 100
    }

def run_walk_forward(data, params, train_years=2, test_years=1):
    years = sorted(list(set(data.index.year)))
    windows = []
    
    for start_year in range(years[0], years[-1] - train_years - test_years + 1):
        train_start = f"{start_year}-01-01"
        train_end = f"{start_year + train_years}-12-31"
        test_start = f"{start_year + train_years}-01-01"
        test_end = f"{start_year + train_years + test_years}-12-31"
        
        train_data = data.loc[train_start:train_end]
        test_data = data.loc[test_start:test_end]
        
        if len(train_data) < 100 or len(test_data) < 50: continue

        res_train, _ = BacktestEngine.run_simulation(train_data, params)
        res_test, _ = BacktestEngine.run_simulation(test_data, params)
        
        met_train = calculate_metrics(res_train['portfolio'])
        met_test = calculate_metrics(res_test['portfolio'])
        
        overfit = met_train['CAGR'] / met_test['CAGR'] if met_test['CAGR'] != 0 else 0
        
        windows.append({
            "period": f"{start_year}-{start_year+train_years}",
            "train_cagr": met_train['CAGR'],
            "test_cagr": met_test['CAGR'],
            "test_dd": met_test['MaxDD'],
            "overfit": overfit
        })
    return windows

def run_monte_carlo(data, params, runs=100):
    returns = data.pct_change().dropna()
    results = []
    
    for _ in range(runs):
        random_idx = np.random.choice(returns.index, size=len(returns), replace=True)
        boot_rets = returns.loc[random_idx]
        boot_rets.index = returns.index
        
        price_x2 = (1 + boot_rets['X2']).cumprod() * 100
        price_x1 = (1 + boot_rets['X1']).cumprod() * 100
        
        fake_data = pd.DataFrame({'X2': price_x2, 'X1': price_x1}, index=data.index[1:])
        
        sim, _ = BacktestEngine.run_simulation(fake_data, params)
        met = calculate_metrics(sim['portfolio'])
        results.append(met)
        
    return pd.DataFrame(results)

# ==========================================
# 4. DATA ENGINE (VERSION BULLDOZER CORRIGÉE)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """
    Récupère les données de manière agressive.
    Si le téléchargement groupé échoue, on passe ticker par ticker.
    """
    if not tickers: return pd.DataFrame()
    
    price_map = {}
    clean_tickers = [t.strip().upper() for t in tickers]
    
    # TELECHARGEMENT INDIVIDUEL (ROBUSTE)
    for t in clean_tickers:
        try:
            # On force auto_adjust=True
            df_temp = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df_temp.empty:
                # Retry sans auto_adjust
                df_temp = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            
            if not df_temp.empty:
                if 'Close' in df_temp.columns:
                    price_map[t] = df_temp['Close']
                elif 'Adj Close' in df_temp.columns:
                    price_map[t] = df_temp['Adj Close']
                    
        except Exception as e:
            continue

    if len(price_map) >= 2:
        # On suppose que l'ordre d'entrée est respecté : X2 (Risk) puis X1 (Safe)
        df_final = pd.DataFrame(price_map)
        
        # On renomme pour le moteur interne : Colonne 1 -> X2, Colonne 2 -> X1
        cols = df_final.columns
        df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
        
        return df_final.ffill().dropna()
        
    return pd.DataFrame()

# ==========================================
# 5. UI LAYOUT PRINCIPAL
# ==========================================

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="margin:0;" class="title-gradient">Predict. DISTINCT PROFILES</h1>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.0 • RISK INTELLIGENCE • INSTITUTIONAL GRADE</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(16, 185, 129, 0.1); color:#10b981; padding:5px 10px; border-radius:4px; font-size:11px; border:1px solid rgba(16, 185, 129, 0.2);">LIVE SYSTEM</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sidebar, col_main, col_valid = st.columns([1, 2.5, 1.2])

# --- SIDEBAR (CONTROLS) ---
with col_sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ CONFIGURATION")
    
    t_input = st.text_input("Tickers (X2 Risk, X1 Safe)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in t_input.split(',')]
    
    start_d = st.date_input("Début", datetime(2022, 1, 1))
    end_d = st.date_input("Fin", datetime.now())
    
    st.markdown("---")
    thresh = st.slider("Seuil Sortie (%)", 2.0, 15.0, 5.0, 0.5)
    panic = st.slider("Seuil Panic (%)", 10, 30, 15)
    recov = st.slider("Recovery (%)", 10, 80, 30, 5)
    
    st.markdown("---")
    alloc_prud = st.slider("Prudence (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash (X1%)", 0, 100, 100, 10)
    
    st.markdown("---")
    profile = st.selectbox("PROFIL AI", ["DÉFENSIF", "ÉQUILIBRÉ", "AGRESSIF"])
    if st.button(f"🚀 OPTIMISER ({profile})"):
        with st.spinner("Recherche..."):
            if profile == "DÉFENSIF": st.session_state['opt_params'] = {'thresh': 3.0, 'panic': 12, 'recov': 50}
            elif profile == "AGRESSIF": st.session_state['opt_params'] = {'thresh': 8.0, 'panic': 20, 'recov': 20}
            else: st.session_state['opt_params'] = {'thresh': 5.0, 'panic': 15, 'recov': 30}
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- DATA FETCH ---
data = get_data(tickers, start_d, end_d)
params = {
    'thresh': thresh, 'panic': panic, 'recovery': recov,
    'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
    'rollingWindow': 60, 'confirm': 2
}
if 'opt_params' in st.session_state:
    p = st.session_state['opt_params']
    st.sidebar.success(f"AI: Seuil {p['thresh']}% | Panic {p['panic']}% | Recov {p['recov']}%")

# --- MAIN DISPLAY ---
with col_main:
    if data.empty or len(data) < 10:
        st.error(f"""
        ❌ **DONNÉES NON DISPONIBLES**
        
        Impossible de récupérer : **{', '.join(tickers)}**.
        
        **Conseils :**
        1. Vérifiez que les tickers existent sur [Yahoo Finance](https://finance.yahoo.com).
        2. Pour Paris, ajoutez **.PA** (ex: `LQQ.PA`). Pour les USA, rien (ex: `QLD`).
        3. Vérifiez que la date de début n'est pas fériée.
        """)
    else:
        # 1. Backtest
        df_res, trades = BacktestEngine.run_simulation(data, params)
        metrics = calculate_metrics(df_res['portfolio'])
        bench_met = calculate_metrics(df_res['benchX2'])
        
        # 2. Risk & Leverage Calculation
        risk_strat = RiskMetrics.get_full_risk_profile(df_res['portfolio'])
        risk_bench = RiskMetrics.get_full_risk_profile(df_res['benchX2'])
        lev_health = LeverageDiagnostics.calculate_leverage_health(data)
        lev_beta = LeverageDiagnostics.calculate_realized_beta(data)
        lev_decay = LeverageDiagnostics.detect_decay_regime(data)
        arb_signals = ArbitrageSignals.calculate_relative_strength(data)
        arb_status = ArbitrageSignals.get_signal_status(arb_signals['Z_Score']) if not arb_signals.empty else {}

        # 3. KPI ROW 1: Performance
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR Strat", f"{metrics['CAGR']:.1f}%", delta=f"{metrics['CAGR']-bench_met['CAGR']:.1f}%")
        k2.metric("Max Drawdown", f"{metrics['MaxDD']:.1f}%", delta=f"{metrics['MaxDD']-bench_met['MaxDD']:.1f}%", delta_color="inverse")
        k3.metric("Volatilité", f"{metrics['Vol']:.1f}%")
        k4.metric("Trades", len(trades))

        # 4. KPI ROW 2: Advanced Risk (Module 1)
        if MODULES_STATUS["Risk"]:
            st.markdown("### ⚠️ Profil de Risque (Institutionnel)")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Ulcer Index", f"{risk_strat.get('Ulcer_Index', 0):.2f}", delta=f"{risk_strat.get('Ulcer_Index', 0)-risk_bench.get('Ulcer_Index', 0):.2f}", delta_color="inverse")
            r2.metric("VaR 95%", f"{risk_strat.get('VaR_95', 0)*100:.2f}%", delta_color="inverse")
            r3.metric("CVaR 95%", f"{risk_strat.get('CVaR_95', 0)*100:.2f}%", delta_color="inverse")
            r4.metric("Vol Annuelle", f"{risk_strat.get('Vol_Ann', 0)*100:.1f}%", delta_color="inverse")

        # 5. KPI ROW 3: Leverage (Module 2)
        if MODULES_STATUS["Leverage"]:
            st.markdown("### ⚙️ Efficacité du Levier")
            l1, l2, l3, l4 = st.columns(4)
            real_beta = lev_health.get('Realized_Leverage', 0)
            decay_val = lev_decay['Decay_Spread'].iloc[-1]*100 if not lev_decay.empty else 0
            
            l1.metric("Beta Réalisé", f"{real_beta:.2f}x", delta=f"{real_beta-2.0:.2f}", help="Cible: 2.0x")
            l2.metric("Vol Ratio", f"{lev_health.get('Vol_Ratio', 0):.2f}x")
            l3.metric("Decay (60J)", f"{decay_val:.2f}%", delta_color="normal" if decay_val > 0 else "inverse")
            l4.metric("Tracking Err", f"{lev_health.get('Tracking_Error', 0):.4f}")

        # 6. KPI ROW 4: Arbitrage (Module 3)
        if MODULES_STATUS["Arbitrage"]:
            st.markdown("### 🎯 Signal d'Arbitrage (X2/X1)")
            a1, a2, a3 = st.columns([1, 2, 1])
            curr_z = arb_status.get('Current_Z', 0)
            a1.metric("Z-Score (20D)", f"{curr_z:.2f}", delta="Rich" if curr_z>0 else "Cheap", delta_color="inverse")
            col_z = arb_status.get('Color', 'white')
            a2.markdown(f"<div style='border:1px solid {col_z}; border-radius:8px; padding:10px; text-align:center; background:{col_z}20;'><span style='color:{col_z}; font-weight:bold;'>{arb_status.get('Status', 'N/A')}</span></div>", unsafe_allow_html=True)
            a3.markdown(f"<div style='text-align:center; padding-top:10px; font-size:12px; color:#888;'>Action:<br><b style='color:#fff'>{arb_status.get('Action','Wait')}</b></div>", unsafe_allow_html=True)

        # 7. CHART PRINCIPAL (Performance)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['portfolio'], mode='lines', name='STRATÉGIE', line=dict(color='#667eea', width=3), fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX2'], mode='lines', name=f'{tickers[0]} (RISK)', line=dict(color='#ef4444', width=1.5, dash='dot'), opacity=0.8))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX1'], mode='lines', name=f'{tickers[1]} (SAFE)', line=dict(color='#10b981', width=1.5, dash='dot'), opacity=0.6))
        
        for t in trades:
            col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
            fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['portfolio'], text="▼" if t['to'] != 'R0' else "▲", showarrow=False, font=dict(color=col, size=14))
            
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=450, margin=dict(l=0, r=0, t=20, b=0), xaxis=dict(showgrid=False, linecolor='#333'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), hovermode="x unified", legend=dict(orientation="h", y=1.05, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 8. OSCILLATEUR Z-SCORE (Module 3)
        if MODULES_STATUS["Arbitrage"] and not arb_signals.empty:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_arb = go.Figure()
            fig_arb.add_trace(go.Scatter(x=arb_signals.index, y=arb_signals['Z_Score'], mode='lines', name='Z-Score', line=dict(color='#3b82f6', width=2)))
            fig_arb.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0)
            fig_arb.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
            fig_arb.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=200, margin=dict(t=10,b=10), yaxis=dict(title="Z-Score", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-3.5, 3.5]), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_arb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 9. ROLLING BETA (Module 2)
        if MODULES_STATUS["Leverage"] and not lev_beta.empty:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(x=lev_beta.index, y=lev_beta['Realized_Beta'], mode='lines', name='Rolling Beta', line=dict(color='#A855F7', width=2)))
            fig_lev.add_hline(y=2.0, line_dash="dot", line_color="white")
            fig_lev.add_hrect(y0=0.0, y1=1.5, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0)
            fig_lev.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=200, margin=dict(t=10,b=10), yaxis=dict(title="Beta", showgrid=True, gridcolor='rgba(255,255,255,0.05)'), xaxis=dict(showgrid=False), showlegend=False)
            st.plotly_chart(fig_lev, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 10. UNDERWATER (Standard)
        st.markdown("### 🌊 Underwater Plot")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        dd_series = (df_res['portfolio'] / df_res['portfolio'].cummax() - 1) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='#ef4444', width=1), fillcolor='rgba(239, 68, 68, 0.2)'))
        fig_dd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=200, margin=dict(t=0,b=0,l=0,r=0), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig_dd, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- VALIDATION COLUMN ---
with col_valid:
    if not data.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛡️ ROBUSTESSE")
        if st.button("LANCER VALIDATION"):
            with st.spinner("Calculs..."):
                wf_res = run_walk_forward(data, params)
                mc_res = run_monte_carlo(data, params)
                avg_overfit = np.mean([w['overfit'] for w in wf_res]) if wf_res else 0
                prob_loss = len(mc_res[mc_res['CAGR'] < 0]) / len(mc_res) * 100
                
                verdict = "ROBUSTE" if avg_overfit < 1.5 and prob_loss < 20 else "FRAGILE"
                col_v = "#10b981" if verdict == "ROBUSTE" else "#ef4444"
                
                st.markdown(f"<div style='background:{col_v}20; border:1px solid {col_v}; padding:15px; border-radius:10px; text-align:center;'><h2 style='color:{col_v}; margin:0;'>{verdict}</h2><p style='font-size:11px; margin:5px 0 0 0; color:#aaa;'>Overfit: {avg_overfit:.2f}x • Prob. Perte: {prob_loss:.0f}%</p></div>", unsafe_allow_html=True)
                
                st.markdown("#### Walk-Forward")
                for w in wf_res:
                    c_o = "#ef4444" if w['overfit'] > 1.5 else "#10b981"
                    st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:8px; border-radius:6px; margin-bottom:5px; font-size:11px; display:flex; justify-content:space-between;'><span>{w['period']}</span><span>Tr:{w['train_cagr']:.0f}% Te:{w['test_cagr']:.0f}%</span><span style='color:{c_o}'>{w['overfit']:.1f}x</span></div>", unsafe_allow_html=True)
                    
                st.markdown("#### Monte-Carlo Dist.")
                fig_mc = px.histogram(mc_res, x="CAGR", nbins=20, color_discrete_sequence=['#667eea'])
                fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#888', size=10), height=150, margin=dict(l=0,r=0,t=0,b=0), xaxis_title=None, yaxis_title=None, showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
