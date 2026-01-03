import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIGURATION & IMPORTS
# ==========================================
st.set_page_config(page_title="Predict. Distinct | Institutional", layout="wide", page_icon="⚡")

# --- MODULE IMPORT WITH FALLBACK ---
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

# --- CSS: SILENT LUXURY ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0a0a0f; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    
    /* HEADER */
    .header-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;
    }
    .title-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 28px;
    }
    
    /* METRICS TABLE */
    .metric-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .metric-table th { text-align: left; color: #888; border-bottom: 1px solid #333; padding: 8px; }
    .metric-table td { padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .metric-val { font-family: 'Courier New', monospace; font-weight: 600; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 20px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 2px solid #667eea !important; }
    
    /* CARDS */
    .glass-card { background: rgba(30, 30, 46, 0.6); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px; }
    
    /* UTILS */
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; }
    header, footer { visibility: hidden; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE ENGINE (With Transaction Costs)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        # Initialisation
        cash_x2 = 100.0
        cash_x1 = 0.0
        portfolio = 100.0
        
        current_regime = 'R0' 
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
        
        # New: Transaction Cost (Institutional Reality)
        tx_cost = params.get('cost', 0.001) # Default 0.10%

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
                
                # Allocation Logic
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
                
                # --- APPLY TRANSACTION COST ---
                # Institutional rule: Switching regime costs money (spread + comms)
                cost_impact = total_val * tx_cost
                total_val -= cost_impact
                # ------------------------------

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
                    'val': total_val,
                    'cost': cost_impact
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
# 2. REAL OPTIMIZER (GRID SEARCH)
# ==========================================
class Optimizer:
    @staticmethod
    def run_grid_search(data, profile, fixed_params):
        """
        Runs a real Grid Search over the parameter space.
        Replaces the mock logic with quantitative selection.
        """
        # Parameter Space (Institutional Range)
        thresholds = [2, 4, 6, 8, 10]
        panics = [10, 15, 20, 25, 30]
        recoveries = [20, 30, 40, 50]
        
        best_score = -np.inf
        best_params = {}
        
        # Base parameters
        test_params = fixed_params.copy()
        
        iterations = len(thresholds) * len(panics) * len(recoveries)
        # In production, we might use a progress bar here
        
        for t in thresholds:
            for p in panics:
                if p <= t: continue # Panic must be > Threshold
                for r in recoveries:
                    test_params.update({'thresh': t, 'panic': p, 'recovery': r})
                    
                    # Run Simulation (Fast)
                    res, _ = BacktestEngine.run_simulation(data, test_params)
                    metrics = calculate_metrics(res['portfolio'])
                    
                    # Objective Functions
                    score = -np.inf
                    
                    if profile == "DEFENSIVE":
                        # Maximize Calmar Ratio (Return per unit of Drawdown)
                        score = metrics['Calmar']
                        
                    elif profile == "BALANCED":
                        # Maximize Sharpe Ratio
                        score = metrics['Sharpe']
                        
                    elif profile == "AGGRESSIVE":
                        # Maximize CAGR, subject to MaxDD constraint (e.g. not worse than -35%)
                        if metrics['MaxDD'] > -35.0: # MaxDD is usually negative, so > -35 means -20, -10 etc.
                            score = metrics['CAGR']
                        else:
                            score = -1000 # Penalize heavy drawdowns
                            
                    if score > best_score:
                        best_score = score
                        best_params = {'thresh': t, 'panic': p, 'recovery': r}
                        
        return best_params, best_score

# ==========================================
# 3. METRICS & HELPERS
# ==========================================
def calculate_metrics(series):
    if series.empty: return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}
    
    # Cumulative
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    
    # CAGR
    days = len(series)
    if days < 2: return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1)
    
    # MaxDD
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Volatility & Sharpe
    pct_change = series.pct_change().dropna()
    vol = pct_change.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "Cumul": total_ret * 100,
        "CAGR": cagr * 100,
        "MaxDD": max_dd * 100,
        "Vol": vol * 100,
        "Sharpe": sharpe,
        "Calmar": calmar
    }

# ==========================================
# 4. DATA FETCHING (BULLDOZER)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    
    price_map = {}
    clean_tickers = [t.strip().upper() for t in tickers]
    
    for t in clean_tickers:
        try:
            df_temp = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df_temp.empty:
                df_temp = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            
            if not df_temp.empty:
                if 'Close' in df_temp.columns: s = df_temp['Close']
                elif 'Adj Close' in df_temp.columns: s = df_temp['Adj Close']
                else: s = df_temp.iloc[:, 0]
                
                s.name = t
                price_map[t] = s
        except: continue

    if len(price_map) >= 2:
        df_final = pd.concat(price_map.values(), axis=1)
        # Rename columns to X2 (Risk) and X1 (Safe) based on input order
        # Assuming user inputs Risk first
        cols = df_final.columns
        if len(cols) >= 2:
            df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
            return df_final.ffill().dropna()
            
    return pd.DataFrame()

# ==========================================
# 5. UI LAYOUT
# ==========================================

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="margin:0;" class="title-gradient">Predict. DISTINCT PROFILES</h1>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">INSTITUTIONAL GRADE • REAL OPTIMIZER • TRANSACTION COSTS</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(16, 185, 129, 0.1); color:#10b981; padding:5px 10px; border-radius:4px; font-size:11px; border:1px solid rgba(16, 185, 129, 0.2);">LIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sidebar, col_main = st.columns([1, 3])

# --- SIDEBAR ---
with col_sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏛️ ASSET SELECTION")
    
    # PRESETS DROPDOWN
    presets = {
        "Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"],
        "S&P 500 (US)": ["SSO", "SPY"],
        "MSCI World (Amundi)": ["CL2.PA", "CW8.PA"],
        "Custom (Manual)": []
    }
    
    selected_preset = st.selectbox("Universe", list(presets.keys()))
    
    if selected_preset == "Custom (Manual)":
        t_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [t.strip().upper() for t in t_input.split(',')]
    else:
        tickers = presets[selected_preset]
        st.caption(f"Risk (X2): **{tickers[0]}**")
        st.caption(f"Safe (X1): **{tickers[1]}**")
    
    start_d = st.date_input("Start Date", datetime(2022, 1, 1))
    end_d = st.date_input("End Date", datetime.now())
    
    st.markdown("---")
    st.markdown("### ⚡ PARAMETERS")
    
    # Initialize Session State for Params
    if 'params' not in st.session_state:
        st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30}
    
    # Sliders linked to Session State
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(st.session_state['params']['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 30, int(st.session_state['params']['panic']), 1)
    recov = st.slider("Recovery (%)", 20, 60, int(st.session_state['params']['recovery']), 5)
    
    st.markdown("---")
    alloc_prud = st.slider("Prudence Alloc (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash Alloc (X1%)", 0, 100, 100, 10)
    tx_cost = st.number_input("Transaction Cost (%)", 0.0, 1.0, 0.10, 0.01) / 100
    
    st.markdown("---")
    st.markdown("### 🧠 AI OPTIMIZER")
    profile = st.selectbox("Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    if st.button(f"RUN GRID SEARCH ({profile})"):
        # We need data to optimize
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner(f"Testing parameter combinations for {profile}..."):
                base_params = {
                    'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
                    'rollingWindow': 60, 'confirm': 2, 'cost': tx_cost
                }
                best_p, best_s = Optimizer.run_grid_search(opt_data, profile, base_params)
                
                # Update Session State
                st.session_state['params'] = best_p
                st.success(f"Optimized! Score: {best_s:.2f}")
                st.rerun() # Refresh app with new params
        else:
            st.error("No data to optimize.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN ENGINE ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty or len(data) < 10:
        st.error(f"❌ **NO DATA FOUND** for {tickers}. Please check tickers or date range.")
    else:
        # Build Parameter Dict
        sim_params = {
            'thresh': thresh, 'panic': panic, 'recovery': recov,
            'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
            'rollingWindow': 60, 'confirm': 2, 'cost': tx_cost
        }
        
        # 1. RUN SIMULATION ONCE
        df_res, trades = BacktestEngine.run_simulation(data, sim_params)
        
        # 2. CALCULATE ALL METRICS ONCE
        met_strat = calculate_metrics(df_res['portfolio'])
        met_bench_x2 = calculate_metrics(df_res['benchX2'])
        met_bench_x1 = calculate_metrics(df_res['benchX1'])
        
        if MODULES_STATUS["Risk"]:
            risk_strat = RiskMetrics.get_full_risk_profile(df_res['portfolio'])
        else: risk_strat = {}

        # 3. TABS LAYOUT (UI RESTRUCTURING)
        tab_dash, tab_perf, tab_risk, tab_signals, tab_valid = st.tabs([
            "📊 Dashboard", "📈 Performance", "⚙️ Risk & Leverage", "🎯 Signals", "🛡️ Validation"
        ])
        
        # --- TAB 1: DASHBOARD ---
        with tab_dash:
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Strategy", f"{met_strat['CAGR']:.1f}%", delta=f"{met_strat['CAGR'] - met_bench_x2['CAGR']:.1f}% vs X2")
            k2.metric("Max Drawdown", f"{met_strat['MaxDD']:.1f}%", delta=f"{met_strat['MaxDD'] - met_bench_x2['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe Ratio", f"{met_strat['Sharpe']:.2f}", delta=f"{met_strat['Sharpe'] - met_bench_x2['Sharpe']:.2f}")
            k4.metric("Trades Executed", len(trades))
            
            # Main Chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['portfolio'], name='STRATEGY', line=dict(color='#667eea', width=3), fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX2'], name='Risk Asset (X2)', line=dict(color='#ef4444', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX1'], name='Safe Asset (X1)', line=dict(color='#10b981', width=1.5, dash='dot')))
            
            # Trade Markers
            for t in trades:
                col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['portfolio'], text="▼" if t['to'] != 'R0' else "▲", showarrow=False, font=dict(color=col, size=14))
                
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=450, margin=dict(l=0, r=0, t=20, b=0), xaxis=dict(showgrid=False, linecolor='#333'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), hovermode="x unified", legend=dict(orientation="h", y=1.05, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: PERFORMANCE TABLE ---
        with tab_perf:
            st.markdown("### 🏆 Institutional Performance Attribution")
            
            perf_data = {
                "Metric": ["Cumul. Return", "CAGR (Annualized)", "Volatility (Ann.)", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"],
                "Strategy": [f"{met_strat['Cumul']:.1f}%", f"{met_strat['CAGR']:.1f}%", f"{met_strat['Vol']:.1f}%", f"{met_strat['Sharpe']:.2f}", f"{met_strat['MaxDD']:.1f}%", f"{met_strat['Calmar']:.2f}"],
                "Benchmark X2 (Risk)": [f"{met_bench_x2['Cumul']:.1f}%", f"{met_bench_x2['CAGR']:.1f}%", f"{met_bench_x2['Vol']:.1f}%", f"{met_bench_x2['Sharpe']:.2f}", f"{met_bench_x2['MaxDD']:.1f}%", f"{met_bench_x2['Calmar']:.2f}"],
                "Benchmark X1 (Safe)": [f"{met_bench_x1['Cumul']:.1f}%", f"{met_bench_x1['CAGR']:.1f}%", f"{met_bench_x1['Vol']:.1f}%", f"{met_bench_x1['Sharpe']:.2f}", f"{met_bench_x1['MaxDD']:.1f}%", f"{met_bench_x1['Calmar']:.2f}"]
            }
            
            df_perf = pd.DataFrame(perf_data)
            
            # HTML Table Render
            st.markdown(df_perf.style.hide(axis="index")
                        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#1e1e2e'), ('color', '#aaa')]}])
                        .set_properties(**{'background-color': '#0a0a0f', 'color': '#eee', 'border-color': '#333'})
                        .to_html(), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌊 Underwater Drawdown")
            
            dd_strat = (df_res['portfolio'] / df_res['portfolio'].cummax() - 1) * 100
            dd_x2 = (df_res['benchX2'] / df_res['benchX2'].cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat, fill='tozeroy', name='Strategy DD', line=dict(color='#667eea', width=1)))
            fig_dd.add_trace(go.Scatter(x=dd_x2.index, y=dd_x2, name='Benchmark X2 DD', line=dict(color='#ef4444', width=1, dash='dot')))
            fig_dd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='#888'), height=250, margin=dict(t=10,b=10), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_dd, use_container_width=True)

        # --- TAB 3: RISK & LEVERAGE ---
        with tab_risk:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ⚠️ Risk Profile")
                if MODULES_STATUS["Risk"]:
                    st.metric("Ulcer Index", f"{risk_strat.get('Ulcer_Index', 0):.2f}", help="Pain Index (Depth x Duration)")
                    st.metric("VaR 95% (Daily)", f"{risk_strat.get('VaR_95', 0)*100:.2f}%")
                    st.metric("CVaR 95% (Tail)", f"{risk_strat.get('CVaR_95', 0)*100:.2f}%")
                else:
                    st.info("Risk Module not loaded.")
            
            with c2:
                st.markdown("### ⚙️ Leverage Health")
                if MODULES_STATUS["Leverage"]:
                    lev_health = LeverageDiagnostics.calculate_leverage_health(data)
                    lev_beta = LeverageDiagnostics.calculate_realized_beta(data)
                    
                    st.metric("Realized Beta", f"{lev_health.get('Realized_Leverage', 0):.2f}x")
                    st.metric("Volatility Ratio", f"{lev_health.get('Vol_Ratio', 0):.2f}x")
                    
                    if not lev_beta.empty:
                        fig_lev = go.Figure()
                        fig_lev.add_trace(go.Scatter(x=lev_beta.index, y=lev_beta['Realized_Beta'], line=dict(color='#A855F7')))
                        fig_lev.add_hline(y=2.0, line_dash="dot", line_color="white")
                        fig_lev.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#888'), height=200, margin=dict(t=10,b=10), yaxis_title="Beta")
                        st.plotly_chart(fig_lev, use_container_width=True)
                else:
                    st.info("Leverage Module not loaded.")

        # --- TAB 4: SIGNALS ---
        with tab_signals:
            if MODULES_STATUS["Arbitrage"]:
                arb_signals = ArbitrageSignals.calculate_relative_strength(data)
                if not arb_signals.empty:
                    last_z = arb_signals['Z_Score'].iloc[-1]
                    
                    s1, s2 = st.columns([1, 3])
                    s1.metric("Z-Score (X2/X1)", f"{last_z:.2f}", delta="Expensive" if last_z > 0 else "Cheap", delta_color="inverse")
                    
                    fig_arb = go.Figure()
                    fig_arb.add_trace(go.Scatter(x=arb_signals.index, y=arb_signals['Z_Score'], line=dict(color='#3b82f6')))
                    fig_arb.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0)
                    fig_arb.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                    fig_arb.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#888'), height=300, margin=dict(t=10,b=10), yaxis_title="Z-Score")
                    s2.plotly_chart(fig_arb, use_container_width=True)
            else:
                st.info("Arbitrage Module not loaded.")

        # --- TAB 5: VALIDATION ---
        with tab_valid:
            st.markdown("### 🛡️ Robustness Testing")
            
            # Simple Monte Carlo Logic defined directly here to ensure it works
            def run_monte_carlo(data, params):
                rets = data.pct_change().dropna()
                res_mc = []
                for _ in range(50): # 50 runs for speed
                    idx = np.random.choice(rets.index, size=len(rets), replace=True)
                    boot_rets = rets.loc[idx]
                    boot_rets.index = rets.index
                    
                    # Reconstruct prices
                    p_x2 = (1 + boot_rets['X2']).cumprod() * 100
                    p_x1 = (1 + boot_rets['X1']).cumprod() * 100
                    fake_data = pd.DataFrame({'X2': p_x2, 'X1': p_x1}, index=data.index[1:])
                    
                    sim, _ = BacktestEngine.run_simulation(fake_data, params)
                    met = calculate_metrics(sim['portfolio'])
                    res_mc.append(met)
                return pd.DataFrame(res_mc)

            if st.button("RUN MONTE CARLO (50 Runs)"):
                with st.spinner("Simulating alternate realities..."):
                    mc_df = run_monte_carlo(data, sim_params)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median CAGR", f"{mc_df['CAGR'].median():.1f}%")
                    c2.metric("Worst Case CAGR (5%)", f"{mc_df['CAGR'].quantile(0.05):.1f}%")
                    c3.metric("Prob of Loss", f"{(mc_df['CAGR'] < 0).mean() * 100:.0f}%")
                    
                    fig_mc = px.histogram(mc_df, x="CAGR", nbins=15, color_discrete_sequence=['#667eea'], title="CAGR Distribution")
                    fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#888'))
                    st.plotly_chart(fig_mc, use_container_width=True)
