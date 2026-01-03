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

# --- CSS: SILENT LUXURY THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0a0a0f; font-family: 'Inter', sans-serif; color: #E0E0E0; }
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    
    /* HEADER */
    .header-container {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border-radius: 12px; padding: 25px; 
        border: 1px solid rgba(255,255,255,0.08); 
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .title-gradient {
        background: linear-gradient(135deg, #667eea 0%, #A855F7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 32px; letter-spacing: -1px;
    }
    
    /* METRICS TABLE */
    table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Inter'; }
    th { text-align: left; color: #aaa; background-color: #1e1e2e; padding: 10px; border-bottom: 1px solid #333; }
    td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #E0E0E0; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; font-weight: 600; }
    
    /* CARDS */
    .glass-card { 
        background: rgba(30, 30, 46, 0.6); 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* WIDGETS */
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1e1e2e; color: #A855F7; border: 1px solid #A855F7; }
    .stButton > button:hover { background-color: #A855F7; color: white; border: 1px solid #A855F7; }
    
    /* REMOVE UTILS */
    header, footer { visibility: hidden; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CORE ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        cash_x2, cash_x1, portfolio = 100.0, 0.0, 100.0
        current_regime, pending_regime, confirm_count = 'R0', 'R0', 0
        price_history_x2 = []
        peak_at_crash, trough_x2 = 0.0, 0.0
        results, trades = [], []
        
        # Params
        rolling_w = int(params['rollingWindow'])
        thresh, panic, recov = params['thresh'], params['panic'], params['recovery']
        confirm = params['confirm']
        alloc_crash = params['allocCrash'] / 100.0
        alloc_prudence = params['allocPrudence'] / 100.0
        tx_cost = params.get('cost', 0.001)

        dates = data.index
        px_x2, px_x1 = data['X2'].values, data['X1'].values
        
        for i in range(len(data)):
            # 1. Update Portfolio
            if i > 0:
                r_x2 = (px_x2[i] - px_x2[i-1]) / px_x2[i-1] if px_x2[i-1] != 0 else 0
                r_x1 = (px_x1[i] - px_x1[i-1]) / px_x1[i-1] if px_x1[i-1] != 0 else 0
                cash_x2 *= (1 + r_x2)
                cash_x1 *= (1 + r_x1)
                portfolio = cash_x2 + cash_x1
            
            # 2. Indicators
            curr_price = px_x2[i]
            price_history_x2.append(curr_price)
            if len(price_history_x2) > rolling_w: price_history_x2.pop(0)
            
            rolling_peak = max(price_history_x2)
            if rolling_peak == 0: rolling_peak = 1
            current_dd = ((curr_price - rolling_peak) / rolling_peak) * 100
            
            # 3. Regime Logic
            target = current_regime
            if current_regime != 'R2':
                if current_dd <= -panic: target = 'R2'
                elif current_dd <= -thresh: target = 'R1'
                else: target = 'R0'
            
            if current_regime in ['R1', 'R2']:
                if curr_price < trough_x2: trough_x2 = curr_price
                recovery_target = trough_x2 + (peak_at_crash - trough_x2) * (recov / 100.0)
                if curr_price >= recovery_target: target = 'R0'
                else:
                    if current_dd <= -panic: target = 'R2'
                    elif current_dd <= -thresh and current_regime != 'R2': target = 'R1'
            else:
                peak_at_crash, trough_x2 = rolling_peak, curr_price

            # 4. Execution
            if target == pending_regime: confirm_count += 1
            else: pending_regime = target; confirm_count = 0
                
            if confirm_count >= confirm and pending_regime != current_regime:
                old_regime, current_regime = current_regime, pending_regime
                
                target_pct_x1, label = 0.0, ""
                if current_regime == 'R2': target_pct_x1, label = alloc_crash, "CRASH"
                elif current_regime == 'R1': target_pct_x1, label = alloc_prudence, "PRUDENCE"
                else: target_pct_x1, label = 0.0, "OFFENSIF"
                
                total_val = cash_x1 + cash_x2
                cost_impact = total_val * tx_cost
                total_val -= cost_impact
                
                cash_x1 = total_val * target_pct_x1
                cash_x2 = total_val * (1 - target_pct_x1)
                
                if current_regime != 'R0': peak_at_crash, trough_x2 = rolling_peak, curr_price
                
                trades.append({'date': dates[i], 'from': old_regime, 'to': current_regime, 'label': label, 'val': total_val, 'cost': cost_impact})
                confirm_count = 0

            results.append({'date': dates[i], 'portfolio': portfolio, 'X1': px_x1[i], 'X2': px_x2[i], 'regime': current_regime})
            
        df_res = pd.DataFrame(results).set_index('date')
        if not df_res.empty:
            df_res['portfolio'] = (df_res['portfolio'] / df_res['portfolio'].iloc[0]) * 100
            df_res['benchX1'] = (df_res['X1'] / df_res['X1'].iloc[0]) * 100
            df_res['benchX2'] = (df_res['X2'] / df_res['X2'].iloc[0]) * 100
            
        return df_res, trades

# ==========================================
# 2. REAL OPTIMIZER
# ==========================================
class Optimizer:
    @staticmethod
    def run_grid_search(data, profile, fixed_params):
        thresholds = [2, 4, 6, 8, 10]
        panics = [10, 15, 20, 25, 30]
        recoveries = [20, 30, 40, 50]
        
        best_score, best_params = -np.inf, {}
        test_params = fixed_params.copy()
        
        for t in thresholds:
            for p in panics:
                if p <= t: continue
                for r in recoveries:
                    test_params.update({'thresh': t, 'panic': p, 'recovery': r})
                    res, _ = BacktestEngine.run_simulation(data, test_params)
                    metrics = calculate_metrics(res['portfolio'])
                    
                    score = -np.inf
                    if profile == "DEFENSIVE": score = metrics['Calmar']
                    elif profile == "BALANCED": score = metrics['Sharpe']
                    elif profile == "AGGRESSIVE":
                        score = metrics['CAGR'] if metrics['MaxDD'] > -35.0 else -1000
                            
                    if score > best_score:
                        best_score, best_params = score, {'thresh': t, 'panic': p, 'recovery': r}
        return best_params, best_score

# ==========================================
# 3. METRICS
# ==========================================
def calculate_metrics(series):
    if series.empty: return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}
    
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = len(series)
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 1 else 0
    
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    pct_change = series.pct_change().dropna()
    vol = pct_change.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return { "Cumul": total_ret*100, "CAGR": cagr*100, "MaxDD": max_dd*100, "Vol": vol*100, "Sharpe": sharpe, "Calmar": calmar }

# ==========================================
# 4. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    price_map = {}
    
    for t in [x.strip().upper() for x in tickers]:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty: df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            
            if not df.empty:
                if 'Close' in df.columns: s = df['Close']
                elif 'Adj Close' in df.columns: s = df['Adj Close']
                else: s = df.iloc[:, 0]
                price_map[t] = s
        except: continue

    if len(price_map) >= 2:
        df_final = pd.concat(price_map.values(), axis=1)
        cols = df_final.columns
        # Rename to X2, X1 (assuming input order Risk, Safe)
        if len(cols) >= 2:
            df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
            return df_final.ffill().dropna()
            
    return pd.DataFrame()

# ==========================================
# 5. UI & CHARTS
# ==========================================
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="margin:0;" class="title-gradient">Predict. DISTINCT PROFILES</h1>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.0 • SILENT LUXURY THEME • INSTITUTIONAL</p>
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
        "Custom": []
    }
    
    sel_preset = st.selectbox("Universe", list(presets.keys()))
    if sel_preset == "Custom":
        t_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [t.strip().upper() for t in t_input.split(',')]
    else:
        tickers = presets[sel_preset]
        st.caption(f"Risk: **{tickers[0]}** | Safe: **{tickers[1]}**")
    
    start_d = st.date_input("Start", datetime(2022, 1, 1))
    end_d = st.date_input("End", datetime.now())
    
    st.markdown("---")
    st.markdown("### ⚡ PARAMS")
    
    if 'params' not in st.session_state: st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30}
    
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(st.session_state['params']['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 30, int(st.session_state['params']['panic']), 1)
    recov = st.slider("Recovery (%)", 20, 60, int(st.session_state['params']['recovery']), 5)
    
    st.markdown("---")
    alloc_prud = st.slider("Prudence (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash (X1%)", 0, 100, 100, 10)
    tx_cost = st.number_input("Tx Cost (%)", 0.0, 1.0, 0.10, 0.01) / 100
    
    st.markdown("---")
    profile = st.selectbox("Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    if st.button(f"RUN OPTIMIZER ({profile})"):
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner("Grid Searching..."):
                base_p = {'allocPrudence': alloc_prud, 'allocCrash': alloc_crash, 'rollingWindow': 60, 'confirm': 2, 'cost': tx_cost}
                best_p, _ = Optimizer.run_grid_search(opt_data, profile, base_p)
                st.session_state['params'] = best_p
                st.success("Optimized!")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty or len(data) < 10:
        st.error(f"❌ **NO DATA** for {tickers}. Check tickers or date range.")
    else:
        sim_params = {
            'thresh': thresh, 'panic': panic, 'recovery': recov,
            'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
            'rollingWindow': 60, 'confirm': 2, 'cost': tx_cost
        }
        
        df_res, trades = BacktestEngine.run_simulation(data, sim_params)
        met_s = calculate_metrics(df_res['portfolio'])
        met_x2 = calculate_metrics(df_res['benchX2'])
        met_x1 = calculate_metrics(df_res['benchX1'])
        
        risk_s = RiskMetrics.get_full_risk_profile(df_res['portfolio']) if MODULES_STATUS["Risk"] else {}
        lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
        arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()

        # TABS
        tabs = st.tabs(["📊 Dashboard", "📈 Performance", "⚙️ Risk & Leverage", "🎯 Signals"])
        
        # --- TAB 1: DASHBOARD ---
        with tabs[0]:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Strat", f"{met_s['CAGR']:.1f}%", delta=f"{met_s['CAGR']-met_x2['CAGR']:.1f}%")
            k2.metric("Max Drawdown", f"{met_s['MaxDD']:.1f}%", delta=f"{met_s['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe Ratio", f"{met_s['Sharpe']:.2f}", delta=f"{met_s['Sharpe']-met_x2['Sharpe']:.2f}")
            k4.metric("Trades", len(trades))
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['portfolio'], name='STRATEGY', 
                                     line=dict(color='#A855F7', width=3), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX2'], name='Risk (X2)', 
                                     line=dict(color='#ef4444', width=1.5, dash='dot')))
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX1'], name='Safe (X1)', 
                                     line=dict(color='#10b981', width=1.5, dash='dot')))
            
            for t in trades:
                col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['portfolio'], text="▼" if t['to']!='R0' else "▲", 
                                   showarrow=False, font=dict(color=col, size=14))
                
            fig.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#0a0a0f', font=dict(family="Inter", color='#E0E0E0'), 
                              height=450, margin=dict(l=40, r=40, t=20, b=40), xaxis=dict(showgrid=False, linecolor='#333'), 
                              yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'), hovermode="x unified", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: PERFORMANCE TABLE ---
        with tabs[1]:
            st.markdown("### 🏆 Performance Attribution")
            p_data = {
                "Metric": ["CAGR", "Vol (Ann)", "Sharpe", "MaxDD", "Calmar", "Cumul"],
                "Strategy": [f"{met_s['CAGR']:.1f}%", f"{met_s['Vol']:.1f}%", f"{met_s['Sharpe']:.2f}", f"{met_s['MaxDD']:.1f}%", f"{met_s['Calmar']:.2f}", f"{met_s['Cumul']:.1f}%"],
                "Risk (X2)": [f"{met_x2['CAGR']:.1f}%", f"{met_x2['Vol']:.1f}%", f"{met_x2['Sharpe']:.2f}", f"{met_x2['MaxDD']:.1f}%", f"{met_x2['Calmar']:.2f}", f"{met_x2['Cumul']:.1f}%"],
                "Safe (X1)": [f"{met_x1['CAGR']:.1f}%", f"{met_x1['Vol']:.1f}%", f"{met_x1['Sharpe']:.2f}", f"{met_x1['MaxDD']:.1f}%", f"{met_x1['Calmar']:.2f}", f"{met_x1['Cumul']:.1f}%"]
            }
            st.markdown(pd.DataFrame(p_data).style.hide(axis="index").set_properties(**{'background-color': '#0a0a0f', 'color': '#eee', 'border-color': '#333'}).to_html(), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌊 Underwater Drawdown")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            dd_s = (df_res['portfolio'] / df_res['portfolio'].cummax() - 1) * 100
            dd_x2 = (df_res['benchX2'] / df_res['benchX2'].cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy', name='Strategy', line=dict(color='#A855F7', width=1), fillcolor='rgba(168, 85, 247, 0.15)'))
            fig_dd.add_trace(go.Scatter(x=dd_x2.index, y=dd_x2, name='Risk (X2)', line=dict(color='#ef4444', width=1, dash='dot')))
            fig_dd.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#0a0a0f', font=dict(family="Inter", color='#E0E0E0'), height=250, margin=dict(t=10,b=10), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_dd, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 3: RISK & LEVERAGE ---
        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ⚠️ Risk Profile")
                if risk_s:
                    st.metric("Ulcer Index", f"{risk_s.get('Ulcer_Index', 0):.2f}")
                    st.metric("VaR 95%", f"{risk_s.get('VaR_95', 0)*100:.2f}%")
                    st.metric("CVaR 95%", f"{risk_s.get('CVaR_95', 0)*100:.2f}%")
            with c2:
                st.markdown("### ⚙️ Leverage")
                if not lev_beta.empty:
                    st.metric("Realized Beta", f"{lev_beta['Realized_Beta'].iloc[-1]:.2f}x")
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=lev_beta.index, y=lev_beta['Realized_Beta'], line=dict(color='#A855F7')))
                    fig_l.add_hline(y=2.0, line_dash="dot", line_color="white")
                    fig_l.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#0a0a0f', font=dict(family="Inter", color='#E0E0E0'), height=200, margin=dict(t=10,b=10))
                    st.plotly_chart(fig_l, use_container_width=True)

        # --- TAB 4: SIGNALS ---
        with tabs[3]:
            if not arb_sig.empty:
                st.markdown("### 🎯 Arbitrage Z-Score")
                curr_z = arb_sig['Z_Score'].iloc[-1]
                st.metric("Current Z-Score", f"{curr_z:.2f}", delta="Rich" if curr_z>0 else "Cheap", delta_color="inverse")
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=arb_sig.index, y=arb_sig['Z_Score'], line=dict(color='#3b82f6', width=2)))
                fig_z.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0)
                fig_z.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                fig_z.update_layout(paper_bgcolor='#0a0a0f', plot_bgcolor='#0a0a0f', font=dict(family="Inter", color='#E0E0E0'), height=300, margin=dict(t=10,b=10), yaxis=dict(title="Sigma", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-3.5, 3.5]))
                st.plotly_chart(fig_z, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No Arbitrage Data Available")
