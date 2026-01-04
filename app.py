import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# ==========================================
# 0. CONFIGURATION & IMPORTS
# ==========================================
st.set_page_config(page_title="Predict.", layout="wide", page_icon="⚡")

# --- MODULE IMPORT WITH FALLBACK ---
MODULES_STATUS = {"Risk": False, "Leverage": False, "Arbitrage": False}
MODULES_ERRORS = {"Risk": "", "Leverage": "", "Arbitrage": ""}

try:
    from modules.risk_metrics import RiskMetrics
    MODULES_STATUS["Risk"] = True
except ImportError as e:
    MODULES_ERRORS["Risk"] = str(e)
    class RiskMetrics:
        @staticmethod
        def get_full_risk_profile(series): return {}

try:
    from modules.leverage_diagnostics import LeverageDiagnostics
    MODULES_STATUS["Leverage"] = True
except ImportError as e:
    MODULES_ERRORS["Leverage"] = str(e)
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
except ImportError as e:
    MODULES_ERRORS["Arbitrage"] = str(e)
    class ArbitrageSignals:
        @staticmethod
        def calculate_relative_strength(data, window=20): return pd.DataFrame()
        @staticmethod
        def get_signal_status(series): return {}

# --- CSS: SILENT LUXURY THEME ---
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
    
    /* METRICS TABLE */
    table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'Inter'; }
    th { text-align: left; color: #aaa; background-color: #1E1E2E; padding: 10px; border-bottom: 1px solid #333; }
    tr:nth-child(even) { background-color: #1E1E2E; }
    tr:nth-child(odd) { background-color: #2A2A3E; }
    td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #E0E0E0; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; font-weight: 600; }
    
    /* SIDEBAR / CARDS */
    .glass-card { 
        background: rgba(30, 30, 46, 0.6); 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* LEGEND BOX */
    .legend-box {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin: 10px 0;
        font-size: 12px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
    }
    
    .legend-symbol {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    /* WIDGETS */
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1E1E2E; color: #A855F7; border: 1px solid #A855F7; transition: all 0.3s; }
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

            # Store allocation percentages
            total = cash_x2 + cash_x1
            pct_x2 = (cash_x2 / total * 100) if total > 0 else 0
            pct_x1 = (cash_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i], 
                'portfolio': portfolio, 
                'X1': px_x1[i], 
                'X2': px_x2[i], 
                'regime': current_regime,
                'alloc_X2': pct_x2,
                'alloc_X1': pct_x1
            })
            
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
# 3. MONTE CARLO FORECAST
# ==========================================
class MonteCarloForecaster:
    @staticmethod
    def run_forecast(data, params, n_simulations=500, forecast_days=252):
        """
        Generate Monte Carlo forecast with probability cone
        """
        # Calculate returns
        returns_x2 = data['X2'].pct_change().dropna()
        returns_x1 = data['X1'].pct_change().dropna()
        
        # Get current portfolio value
        df_res, _ = BacktestEngine.run_simulation(data, params)
        current_value = df_res['portfolio'].iloc[-1]
        
        # Statistics
        mu_x2, sigma_x2 = returns_x2.mean(), returns_x2.std()
        mu_x1, sigma_x1 = returns_x1.mean(), returns_x1.std()
        
        # Run simulations
        simulations = []
        for _ in range(n_simulations):
            # Simulate returns
            sim_rets_x2 = np.random.normal(mu_x2, sigma_x2, forecast_days)
            sim_rets_x1 = np.random.normal(mu_x1, sigma_x1, forecast_days)
            
            # Build price series
            sim_px_x2 = [data['X2'].iloc[-1]]
            sim_px_x1 = [data['X1'].iloc[-1]]
            
            for i in range(forecast_days):
                sim_px_x2.append(sim_px_x2[-1] * (1 + sim_rets_x2[i]))
                sim_px_x1.append(sim_px_x1[-1] * (1 + sim_rets_x1[i]))
            
            # Create dataframe
            future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='D')[1:]
            sim_data = pd.DataFrame({
                'X2': sim_px_x2[1:],
                'X1': sim_px_x1[1:]
            }, index=future_dates)
            
            # Run backtest on simulated data
            try:
                sim_result, _ = BacktestEngine.run_simulation(sim_data, params)
                if not sim_result.empty:
                    # Scale to current value
                    sim_result['portfolio'] = (sim_result['portfolio'] / sim_result['portfolio'].iloc[0]) * current_value
                    simulations.append(sim_result['portfolio'])
            except:
                continue
        
        if len(simulations) == 0:
            return pd.DataFrame()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(simulations).T
        
        # Calculate percentiles
        forecast_summary = pd.DataFrame({
            'median': forecast_df.median(axis=1),
            'p5': forecast_df.quantile(0.05, axis=1),
            'p25': forecast_df.quantile(0.25, axis=1),
            'p75': forecast_df.quantile(0.75, axis=1),
            'p95': forecast_df.quantile(0.95, axis=1)
        })
        
        return forecast_summary

# ==========================================
# 4. METRICS
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
# 5. DATA ENGINE
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
        if len(cols) >= 2:
            df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
            return df_final.ffill().dropna()
            
    return pd.DataFrame()

# ==========================================
# 6. UI & CHARTS
# ==========================================
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
        "Custom": []
    }
    
    sel_preset = st.selectbox("Universe", list(presets.keys()))
    if sel_preset == "Custom":
        t_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [t.strip().upper() for t in t_input.split(',')]
    else:
        tickers = presets[sel_preset]
        st.caption(f"Risk: **{tickers[0]}** | Safe: **{tickers[1]}**")
    
    period_options = ["YTD", "1Y", "3YR", "5YR", "2022", "2008", "Custom"]
    sel_period = st.selectbox("Period", period_options, index=3)
    
    today = datetime.now()
    if sel_period == "YTD": start_d = datetime(today.year, 1, 1)
    elif sel_period == "1Y": start_d = today - timedelta(days=365)
    elif sel_period == "3YR": start_d = today - timedelta(days=365*3)
    elif sel_period == "5YR": start_d = today - timedelta(days=365*5)
    elif sel_period == "2022": start_d = datetime(2022,1,1); end_d = datetime(2022,12,31)
    elif sel_period == "2008": start_d = datetime(2008,1,1); end_d = datetime(2008,12,31)
    else: start_d = datetime(2022,1,1)
    
    if sel_period == "Custom":
        start_d = st.date_input("Start", datetime(2022, 1, 1))
        end_d = st.date_input("End", datetime.now())
    elif sel_period not in ["2022", "2008"]:
        end_d = today
    
    st.markdown("---")
    st.markdown("### ⚡ PARAMS")
    
    if 'params' not in st.session_state: st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30}
    
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(st.session_state['params']['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 30, int(st.session_state['params']['panic']), 1)
    recov = st.slider("Recovery (%)", 20, 60, int(st.session_state['params']['recovery']), 5)
    
    st.markdown("---")
    alloc_prud = st.slider("Prudence (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash (X1%)", 0, 100, 100, 10)
    confirm = st.slider("Confirm (Days)", 1, 3, 2, 1)
    
    st.markdown("---")
    profile = st.selectbox("Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    if st.button(f"RUN OPTIMIZER ({profile})"):
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner("Grid Searching..."):
                base_p = {'allocPrudence': alloc_prud, 'allocCrash': alloc_crash, 'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001}
                best_p, _ = Optimizer.run_grid_search(opt_data, profile, base_p)
                st.session_state['params'] = best_p
                st.success("Optimized!")
                st.rerun()
    
    # Module Status
    st.markdown("---")
    with st.expander("📦 Module Status"):
        for module, status in MODULES_STATUS.items():
            if status:
                st.success(f"✅ {module}")
            else:
                st.error(f"❌ {module}")
                if MODULES_ERRORS[module]:
                    st.caption(f"Error: {MODULES_ERRORS[module][:50]}...")
    
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
            'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001
        }
        
        df_res, trades = BacktestEngine.run_simulation(data, sim_params)
        met_s = calculate_metrics(df_res['portfolio'])
        met_x2 = calculate_metrics(df_res['benchX2'])
        met_x1 = calculate_metrics(df_res['benchX1'])
        
        # Modules Externes
        risk_s = RiskMetrics.get_full_risk_profile(df_res['portfolio']) if MODULES_STATUS["Risk"] else {}
        lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
        arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()

        # TABS
        tabs = st.tabs(["Performance", "Risk & Leverage", "Signals", "Validation", "Monte Carlo Forecast"])
        
        # --- TAB 1: DASHBOARD WITH DUAL CHART ---
        with tabs[0]:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Strat", f"{met_s['CAGR']:.1f}%", delta=f"{met_s['CAGR']-met_x2['CAGR']:.1f}% vs X2")
            k2.metric("Max Drawdown", f"{met_s['MaxDD']:.1f}%", delta=f"{met_s['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe Ratio", f"{met_s['Sharpe']:.2f}", delta=f"{met_s['Sharpe']-met_x2['Sharpe']:.2f}")
            k4.metric("Trades", len(trades))
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Create dual subplot: Performance + Allocation
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=("Portfolio Performance", "Asset Allocation (%)"),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # TOP: Performance Chart
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['portfolio'], 
                name='STRATEGY', 
                line=dict(color='#A855F7', width=3), 
                fill='tozeroy', 
                fillcolor='rgba(168, 85, 247, 0.1)'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['benchX2'], 
                name='Risk (X2)', 
                line=dict(color='#ef4444', width=1.5, dash='dot')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['benchX1'], 
                name='Safe (X1)', 
                line=dict(color='#10b981', width=1.5, dash='dot')
            ), row=1, col=1)
            
            # Add trade annotations
            min_val = min(df_res['portfolio'].min(), df_res['benchX2'].min(), df_res['benchX1'].min())
            max_val = max(df_res['portfolio'].max(), df_res['benchX2'].max(), df_res['benchX1'].max())
            
            for t in trades:
                col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                fig.add_annotation(
                    x=t['date'], 
                    y=df_res.loc[t['date']]['portfolio'], 
                    text="▼" if t['to']!='R0' else "▲", 
                    showarrow=False, 
                    font=dict(color=col, size=14),
                    row=1, col=1
                )
            
            # BOTTOM: Allocation Area Chart
            fig.add_trace(go.Scatter(
                x=df_res.index,
                y=df_res['alloc_X2'],
                name='X2 (Risk)',
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(239, 68, 68, 0.5)',
                stackgroup='one'
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_res.index,
                y=df_res['alloc_X1'],
                name='X1 (Safe)',
                fill='tonexty',
                mode='none',
                fillcolor='rgba(16, 185, 129, 0.5)',
                stackgroup='one'
            ), row=2, col=1)
            
            # Layout
            fig.update_xaxes(showgrid=False, linecolor='#333', row=1, col=1)
            fig.update_xaxes(showgrid=False, linecolor='#333', row=2, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[min_val*0.9, max_val*1.1], row=1, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 100], row=2, col=1)
            
            fig.update_layout(
                paper_bgcolor='#0A0A0F', 
                plot_bgcolor='#0A0A0F', 
                font=dict(family="Inter", color='#E0E0E0'), 
                height=650, 
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", y=1.08, x=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legend for trade signals
            st.markdown("""
            <div class="legend-box">
                <strong>📊 Trade Signal Legend:</strong>
                <div class="legend-item">
                    <span class="legend-symbol" style="color:#10b981;">▲</span>
                    <span><strong>Green Triangle (UP):</strong> Return to OFFENSIVE mode (100% Risk Asset X2)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-symbol" style="color:#f59e0b;">▼</span>
                    <span><strong>Orange Triangle (DOWN):</strong> PRUDENCE mode activated (partial shift to Safe X1)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-symbol" style="color:#ef4444;">▼</span>
                    <span><strong>Red Triangle (DOWN):</strong> CRASH mode activated (maximum protection in Safe X1)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 🏆 Performance Attribution")
            p_data = {
                "Metric": ["CAGR", "Vol (Ann)", "Sharpe", "MaxDD", "Calmar", "Cumul"],
                "Strategy": [f"{met_s['CAGR']:.1f}%", f"{met_s['Vol']:.1f}%", f"{met_s['Sharpe']:.2f}", f"{met_s['MaxDD']:.1f}%", f"{met_s['Calmar']:.2f}", f"{met_s['Cumul']:.1f}%"],
                "Risk (X2)": [f"{met_x2['CAGR']:.1f}%", f"{met_x2['Vol']:.1f}%", f"{met_x2['Sharpe']:.2f}", f"{met_x2['MaxDD']:.1f}%", f"{met_x2['Calmar']:.2f}", f"{met_x2['Cumul']:.1f}%"],
                "Safe (X1)": [f"{met_x1['CAGR']:.1f}%", f"{met_x1['Vol']:.1f}%", f"{met_x1['Sharpe']:.2f}", f"{met_x1['MaxDD']:.1f}%", f"{met_x1['Calmar']:.2f}", f"{met_x1['Cumul']:.1f}%"]
            }
            st.markdown(pd.DataFrame(p_data).style.hide(axis="index").set_properties(**{'background-color': '#0A0A0F', 'color': '#eee', 'border-color': '#333'}).to_html(), unsafe_allow_html=True)

        # --- TAB 2: RISK & LEVERAGE ---
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ⚠️ Risk Profile")
                if risk_s:
                    st.metric("Ulcer Index", f"{risk_s.get('Ulcer_Index', 0):.2f}")
                    st.metric("VaR 95%", f"{risk_s.get('VaR_95', 0)*100:.2f}%")
                    st.metric("CVaR 95%", f"{risk_s.get('CVaR_95', 0)*100:.2f}%")
                else:
                    st.info("Risk metrics module not available")
                    
            with c2:
                st.markdown("### ⚙️ Leverage")
                if not lev_beta.empty and 'Realized_Beta' in lev_beta.columns:
                    st.metric("Realized Beta", f"{lev_beta['Realized_Beta'].iloc[-1]:.2f}x")
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=lev_beta.index, y=lev_beta['Realized_Beta'], line=dict(color='#A855F7')))
                    fig_l.add_hline(y=2.0, line_dash="dot", line_color="white")
                    fig_l.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=200, margin=dict(t=10,b=10))
                    st.plotly_chart(fig_l, use_container_width=True)
                else:
                    st.info("Leverage diagnostics module not available")
            
            st.markdown("### 🌊 Underwater Drawdown")
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            dd_s = (df_res['portfolio'] / df_res['portfolio'].cummax() - 1) * 100
            dd_x2 = (df_res['benchX2'] / df_res['benchX2'].cummax() - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy', name='Strategy', line=dict(color='#A855F7', width=1), fillcolor='rgba(168, 85, 247, 0.15)'))
            fig_dd.add_trace(go.Scatter(x=dd_x2.index, y=dd_x2, name='Risk (X2)', line=dict(color='#ef4444', width=1, dash='dot')))
            fig_dd.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=250, margin=dict(t=10,b=10), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig_dd, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 3: SIGNALS ---
        with tabs[2]:
            if not arb_sig.empty and 'Z_Score' in arb_sig.columns:
                st.markdown("### 🎯 Arbitrage Z-Score")
                curr_z = arb_sig['Z_Score'].iloc[-1]
                st.metric("Current Z-Score", f"{curr_z:.2f}", delta="Rich" if curr_z>0 else "Cheap", delta_color="inverse")
                
                # Z-Score explanation
                st.markdown("""
                <div class="legend-box">
                    <strong>📈 Z-Score Interpretation:</strong><br>
                    <div style="margin-top:8px;">
                        <div class="legend-item">
                            <span style="color:#10b981;">● Z < -2:</span> <strong>OVERSOLD</strong> - Risk asset may be undervalued (green zone)
                        </div>
                        <div class="legend-item">
                            <span style="color:#888;">● -2 < Z < 2:</span> <strong>NEUTRAL</strong> - Normal valuation range
                        </div>
                        <div class="legend-item">
                            <span style="color:#ef4444;">● Z > 2:</span> <strong>OVERBOUGHT</strong> - Risk asset may be overvalued (red zone)
                        </div>
                    </div>
                    <p style="margin-top:10px; font-size:11px; color:#888;">
                    The Z-Score measures how many standard deviations the risk asset price is from its historical mean relative to the safe asset. 
                    Extreme values may signal mean reversion opportunities.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=arb_sig.index, y=arb_sig['Z_Score'], line=dict(color='#3b82f6', width=2)))
                fig_z.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0, annotation_text="OVERBOUGHT", annotation_position="top right")
                fig_z.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0, annotation_text="OVERSOLD", annotation_position="bottom right")
                fig_z.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1)
                fig_z.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=300, margin=dict(t=10,b=10), yaxis=dict(title="Sigma", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-3.5, 3.5]))
                st.plotly_chart(fig_z, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 Arbitrage signals module not available. Enable the module to see relative strength analysis between assets.")

        # --- TAB 4: VALIDATION ---
        with tabs[3]:
            st.markdown("### 🛡️ Robustness Testing")
            
            st.markdown("""
            <div class="legend-box">
                <strong>🔬 Bootstrap Monte Carlo Validation:</strong><br>
                <p style="margin-top:8px; font-size:12px;">
                This test randomly resamples historical returns to create 50 alternative scenarios. 
                It helps assess if your strategy's performance is robust or dependent on a specific sequence of events.
                </p>
                <ul style="font-size:11px; color:#aaa; margin-top:5px;">
                    <li><strong>Median CAGR:</strong> Expected return in typical scenarios</li>
                    <li><strong>Worst Case (5%):</strong> Return in pessimistic scenarios</li>
                    <li><strong>Prob of Loss:</strong> Percentage of scenarios with negative returns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            def run_monte_carlo(data, params):
                rets = data.pct_change().dropna()
                res_mc = []
                for _ in range(50):
                    idx = np.random.choice(rets.index, size=len(rets), replace=True)
                    boot_rets = rets.loc[idx]
                    boot_rets.index = rets.index
                    p_x2 = (1 + boot_rets['X2']).cumprod() * 100
                    p_x1 = (1 + boot_rets['X1']).cumprod() * 100
                    fake_data = pd.DataFrame({'X2': p_x2, 'X1': p_x1}, index=data.index[1:])
                    sim, _ = BacktestEngine.run_simulation(fake_data, params)
                    met = calculate_metrics(sim['portfolio'])
                    res_mc.append(met)
                return pd.DataFrame(res_mc)

            if st.button("RUN MONTE CARLO (50 Runs)"):
                with st.spinner("Simulating..."):
                    mc_df = run_monte_carlo(data, sim_params)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Median CAGR", f"{mc_df['CAGR'].median():.1f}%")
                    c2.metric("Worst Case CAGR (5%)", f"{mc_df['CAGR'].quantile(0.05):.1f}%")
                    c3.metric("Prob of Loss", f"{(mc_df['CAGR'] < 0).mean() * 100:.0f}%")
                    fig_mc = px.histogram(mc_df, x="CAGR", nbins=15, color_discrete_sequence=['#A855F7'])
                    fig_mc.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'))
                    st.plotly_chart(fig_mc, use_container_width=True)

        # --- TAB 5: MONTE CARLO FORECAST (FAN CHART) ---
        with tabs[4]:
            st.markdown("### 🎲 Monte Carlo Forecast - Probability Cone")
            
            st.markdown("""
            <div class="legend-box">
                <strong>🔮 Forward-Looking Simulation:</strong><br>
                <p style="margin-top:8px; font-size:12px;">
                This fan chart projects your portfolio 252 trading days (1 year) into the future using 500 simulated scenarios.
                Unlike backward-looking validation, this shows potential future outcomes based on historical volatility and returns.
                </p>
                <ul style="font-size:11px; color:#aaa; margin-top:5px;">
                    <li><strong>Dark Purple (Median):</strong> Most likely trajectory (50th percentile)</li>
                    <li><strong>Light Purple Zone:</strong> 50% probability range (25th-75th percentile)</li>
                    <li><strong>Wider Cone:</strong> 90% probability range (5th-95th percentile)</li>
                </ul>
                <p style="font-size:11px; color:#888; margin-top:5px;">
                ⚠️ Note: Assumes returns follow historical patterns. Black swan events may fall outside these ranges.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            n_sims = st.slider("Number of Simulations", 100, 1000, 500, 100)
            forecast_days = st.slider("Forecast Horizon (days)", 60, 504, 252, 21)
            
            if st.button(f"🚀 RUN FORECAST ({n_sims} scenarios)"):
                with st.spinner(f"Running {n_sims} Monte Carlo simulations..."):
                    try:
                        forecast = MonteCarloForecaster.run_forecast(data, sim_params, n_simulations=n_sims, forecast_days=forecast_days)
                        
                        if not forecast.empty:
                            # Create fan chart
                            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                            fig_fan = go.Figure()
                            
                            # Add confidence intervals
                            fig_fan.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast['p95'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig_fan.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast['p5'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(168, 85, 247, 0.1)',
                                fill='tonexty',
                                name='90% CI',
                                hovertemplate='5th-95th percentile<br>%{y:.1f}<extra></extra>'
                            ))
                            
                            fig_fan.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast['p75'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig_fan.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast['p25'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(168, 85, 247, 0.2)',
                                fill='tonexty',
                                name='50% CI',
                                hovertemplate='25th-75th percentile<br>%{y:.1f}<extra></extra>'
                            ))
                            
                            # Add median line
                            fig_fan.add_trace(go.Scatter(
                                x=forecast.index,
                                y=forecast['median'],
                                mode='lines',
                                line=dict(color='#A855F7', width=3),
                                name='Median Forecast',
                                hovertemplate='Median: %{y:.1f}<extra></extra>'
                            ))
                            
                            # Add current value marker
                            current_val = df_res['portfolio'].iloc[-1]
                            fig_fan.add_trace(go.Scatter(
                                x=[data.index[-1]],
                                y=[current_val],
                                mode='markers',
                                marker=dict(color='#10b981', size=12, symbol='diamond'),
                                name='Current Value',
                                hovertemplate='Today: %{y:.1f}<extra></extra>'
                            ))
                            
                            fig_fan.update_layout(
                                paper_bgcolor='#0A0A0F',
                                plot_bgcolor='#0A0A0F',
                                font=dict(family="Inter", color='#E0E0E0'),
                                height=500,
                                margin=dict(l=40, r=40, t=40, b=40),
                                xaxis=dict(
                                    title="Date",
                                    showgrid=False,
                                    linecolor='#333'
                                ),
                                yaxis=dict(
                                    title="Portfolio Value",
                                    showgrid=True,
                                    gridcolor='rgba(255,255,255,0.05)'
                                ),
                                hovermode='x unified',
                                legend=dict(orientation="h", y=1.05, x=0)
                            )
                            
                            st.plotly_chart(fig_fan, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Statistics
                            st.markdown("### 📊 Forecast Statistics")
                            final_median = forecast['median'].iloc[-1]
                            final_p5 = forecast['p5'].iloc[-1]
                            final_p95 = forecast['p95'].iloc[-1]
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Expected Value", f"{final_median:.1f}", delta=f"{((final_median/current_val)-1)*100:.1f}%")
                            c2.metric("Pessimistic (5%)", f"{final_p5:.1f}", delta=f"{((final_p5/current_val)-1)*100:.1f}%")
                            c3.metric("Optimistic (95%)", f"{final_p95:.1f}", delta=f"{((final_p95/current_val)-1)*100:.1f}%")
                            c4.metric("Range", f"{final_p95 - final_p5:.1f}")
                            
                        else:
                            st.error("Failed to generate forecast. Try adjusting parameters.")
                            
                    except Exception as e:
                        st.error(f"Error during forecast: {str(e)}")
                        st.info("Try reducing the number of simulations or forecast horizon.")
