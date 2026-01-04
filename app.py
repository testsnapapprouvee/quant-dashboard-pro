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

# Enable debug mode
DEBUG_MODE = st.sidebar.checkbox("🔍 Debug Mode", value=False)

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
    
    /* DEBUG BOX */
    .debug-box {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    }
    
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
# 1. CORE ENGINE (VERSION ULTRA-CORRIGÉE)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params, debug=False):
        """
        🔧 VERSION ULTRA-CORRIGÉE
        - Les benchmarks commencent à 100 et évoluent avec les rendements
        - Plus de normalisation après coup
        """
        # Initialisation
        cash_x2, cash_x1, portfolio = 100.0, 0.0, 100.0
        bench_x2, bench_x1 = 100.0, 100.0  # ✅ Benchmarks à base 100
        
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
        
        debug_info = []
        
        for i in range(len(data)):
            # 1. Calculate returns
            if i > 0:
                r_x2 = (px_x2[i] - px_x2[i-1]) / px_x2[i-1] if px_x2[i-1] != 0 else 0
                r_x1 = (px_x1[i] - px_x1[i-1]) / px_x1[i-1] if px_x1[i-1] != 0 else 0
                
                # 2. Update portfolio (avec allocations actuelles)
                cash_x2 *= (1 + r_x2)
                cash_x1 *= (1 + r_x1)
                portfolio = cash_x2 + cash_x1
                
                # 3. ✅ Update benchmarks (toujours 100% sur chaque actif)
                bench_x2 *= (1 + r_x2)
                bench_x1 *= (1 + r_x1)
                
                # Debug pour les 5 premiers jours
                if debug and i <= 5:
                    debug_info.append({
                        'day': i,
                        'r_x2': f"{r_x2*100:.4f}%",
                        'r_x1': f"{r_x1*100:.4f}%",
                        'portfolio': f"{portfolio:.2f}",
                        'bench_x2': f"{bench_x2:.2f}",
                        'bench_x1': f"{bench_x1:.2f}"
                    })
            
            # 4. Indicators pour le regime
            curr_price = px_x2[i]
            price_history_x2.append(curr_price)
            if len(price_history_x2) > rolling_w: price_history_x2.pop(0)
            
            rolling_peak = max(price_history_x2) if price_history_x2 else curr_price
            if rolling_peak == 0: rolling_peak = 1
            current_dd = ((curr_price - rolling_peak) / rolling_peak) * 100
            
            # 5. Regime Logic
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

            # 6. Execution des trades
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
                
                trades.append({
                    'date': dates[i], 
                    'from': old_regime, 
                    'to': current_regime, 
                    'label': label, 
                    'val': total_val, 
                    'cost': cost_impact
                })
                confirm_count = 0

            # 7. Store Results (✅ avec benchmarks normalisés)
            total = cash_x2 + cash_x1
            pct_x2 = (cash_x2 / total * 100) if total > 0 else 0
            pct_x1 = (cash_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i], 
                'portfolio': portfolio,
                'benchX2': bench_x2,  # ✅ Déjà base 100
                'benchX1': bench_x1,  # ✅ Déjà base 100
                'regime': current_regime,
                'alloc_X2': pct_x2,
                'alloc_X1': pct_x1
            })
            
        df_res = pd.DataFrame(results).set_index('date')
        
        # ✅ AUCUNE NORMALISATION ICI - C'EST DÉJÀ FAIT !
        
        return df_res, trades, debug_info if debug else []

# ==========================================
# 2. AUTRES FONCTIONS (identiques)
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
                    try:
                        res, _, _ = BacktestEngine.run_simulation(data, test_params, debug=False)
                        if res.empty: continue
                        metrics = calculate_metrics(res['portfolio'])
                        
                        score = -np.inf
                        if profile == "DEFENSIVE": score = metrics['Calmar']
                        elif profile == "BALANCED": score = metrics['Sharpe']
                        elif profile == "AGGRESSIVE":
                            score = metrics['CAGR'] if metrics['MaxDD'] > -35.0 else -1000
                                
                        if score > best_score:
                            best_score, best_params = score, {'thresh': t, 'panic': p, 'recovery': r}
                    except:
                        continue
                        
        return best_params, best_score

def calculate_metrics(series):
    if series.empty or len(series) < 2: 
        return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}
    
    try:
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        days = len(series)
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 1 else 0
        
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        max_dd = drawdown.min()
        
        pct_change = series.pct_change().dropna()
        if len(pct_change) > 0:
            vol = pct_change.std() * np.sqrt(252)
            sharpe = cagr / vol if vol != 0 else 0
        else:
            vol = 0
            sharpe = 0
            
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return { 
            "Cumul": total_ret*100, 
            "CAGR": cagr*100, 
            "MaxDD": max_dd*100, 
            "Vol": vol*100, 
            "Sharpe": sharpe, 
            "Calmar": calmar 
        }
    except:
        return {"CAGR":0, "MaxDD":0, "Vol":0, "Sharpe":0, "Calmar":0, "Cumul":0}

@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    price_map = {}
    
    for t in [x.strip().upper() for x in tickers]:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty: 
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            
            if not df.empty:
                if 'Close' in df.columns: s = df['Close']
                elif 'Adj Close' in df.columns: s = df['Adj Close']
                else: s = df.iloc[:, 0]
                price_map[t] = s
        except: 
            continue

    if len(price_map) >= 2:
        df_final = pd.concat(price_map.values(), axis=1)
        cols = df_final.columns
        if len(cols) >= 2:
            df_final.rename(columns={cols[0]: 'X2', cols[1]: 'X1'}, inplace=True)
            df_final = df_final.ffill().dropna()
            return df_final
            
    return pd.DataFrame()

# ==========================================
# 6. UI & MAIN
# ==========================================
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span class="title-text">Predict</span><span class="title-dot">.</span>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.2 • ULTRA-FIX • BASE 100</p>
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
    
    if 'params' not in st.session_state: 
        st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30}
    
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
                base_p = {
                    'allocPrudence': alloc_prud, 
                    'allocCrash': alloc_crash, 
                    'rollingWindow': 60, 
                    'confirm': confirm, 
                    'cost': 0.001
                }
                best_p, _ = Optimizer.run_grid_search(opt_data, profile, base_p)
                if best_p:
                    st.session_state['params'] = best_p
                    st.success("Optimized!")
                    st.rerun()
                else:
                    st.error("Optimization failed")
    
    # Module Status
    st.markdown("---")
    with st.expander("📦 Module Status"):
        for module, status in MODULES_STATUS.items():
            if status:
                st.success(f"✅ {module}")
            else:
                st.error(f"❌ {module}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty or len(data) < 10:
        st.error(f"❌ **NO DATA** for {tickers}. Check tickers or date range.")
    else:
        # Show raw data stats if debug mode
        if DEBUG_MODE:
            st.markdown("### 🔍 DEBUG: RAW DATA FROM YAHOO")
            st.markdown(f"""
            <div class="debug-box">
            <strong>X2 ({tickers[0]}):</strong><br>
            - First price: {data['X2'].iloc[0]:.2f}<br>
            - Last price: {data['X2'].iloc[-1]:.2f}<br>
            - Performance: {((data['X2'].iloc[-1]/data['X2'].iloc[0])-1)*100:.2f}%<br>
            - Range: {data['X2'].min():.2f} to {data['X2'].max():.2f}<br>
            <br>
            <strong>X1 ({tickers[1]}):</strong><br>
            - First price: {data['X1'].iloc[0]:.2f}<br>
            - Last price: {data['X1'].iloc[-1]:.2f}<br>
            - Performance: {((data['X1'].iloc[-1]/data['X1'].iloc[0])-1)*100:.2f}%<br>
            - Range: {data['X1'].min():.2f} to {data['X1'].max():.2f}
            </div>
            """, unsafe_allow_html=True)
        
        sim_params = {
            'thresh': thresh, 'panic': panic, 'recovery': recov,
            'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
            'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001
        }
        
        df_res, trades, debug_info = BacktestEngine.run_simulation(data, sim_params, debug=DEBUG_MODE)
        
        if df_res.empty:
            st.error("❌ Simulation failed. Check your data.")
        else:
            # Debug info
            if DEBUG_MODE and debug_info:
                st.markdown("### 🔍 DEBUG: FIRST 5 DAYS OF SIMULATION")
                st.dataframe(pd.DataFrame(debug_info))
                
                st.markdown(f"""
                <div class="debug-box">
                <strong>After simulation:</strong><br>
                - Portfolio: {df_res['portfolio'].iloc[0]:.2f} → {df_res['portfolio'].iloc[-1]:.2f}<br>
                - BenchX2: {df_res['benchX2'].iloc[0]:.2f} → {df_res['benchX2'].iloc[-1]:.2f}<br>
                - BenchX1: {df_res['benchX1'].iloc[0]:.2f} → {df_res['benchX1'].iloc[-1]:.2f}<br>
                <br>
                <strong>Data ranges for plotting:</strong><br>
                - Portfolio range: {df_res['portfolio'].min():.2f} to {df_res['portfolio'].max():.2f}<br>
                - BenchX2 range: {df_res['benchX2'].min():.2f} to {df_res['benchX2'].max():.2f}<br>
                - BenchX1 range: {df_res['benchX1'].min():.2f} to {df_res['benchX1'].max():.2f}
                </div>
                """, unsafe_allow_html=True)
            
            met_s = calculate_metrics(df_res['portfolio'])
            met_x2 = calculate_metrics(df_res['benchX2'])
            met_x1 = calculate_metrics(df_res['benchX1'])
            
            # Main display
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Strat", f"{met_s['CAGR']:.1f}%", delta=f"{met_s['CAGR']-met_x2['CAGR']:.1f}% vs X2")
            k2.metric("Max Drawdown", f"{met_s['MaxDD']:.1f}%", delta=f"{met_s['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe Ratio", f"{met_s['Sharpe']:.2f}", delta=f"{met_s['Sharpe']-met_x2['Sharpe']:.2f}")
            k4.metric("Trades", len(trades))
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Create main chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['portfolio'], 
                name='STRATEGY', 
                line=dict(color='#A855F7', width=3),
                fill='tozeroy',
                fillcolor='rgba(168, 85, 247, 0.1)'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['benchX2'], 
                name=f'Risk ({tickers[0]})', 
                line=dict(color='#ef4444', width=2, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['benchX1'], 
                name=f'Safe ({tickers[1]})', 
                line=dict(color='#10b981', width=2, dash='dot')
            ))
            
            # Add trade markers
            for t in trades:
                if t['date'] in df_res.index:
                    col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                    fig.add_annotation(
                        x=t['date'], 
                        y=df_res.loc[t['date']]['portfolio'], 
                        text="▼" if t['to']!='R0' else "▲", 
                        showarrow=False, 
                        font=dict(color=col, size=16)
                    )
            
            fig.update_layout(
                paper_bgcolor='#0A0A0F', 
                plot_bgcolor='#0A0A0F', 
                font=dict(family="Inter", color='#E0E0E0'), 
                height=500, 
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", y=1.05, x=0),
                xaxis=dict(showgrid=False, linecolor='#333'),
                yaxis=dict(
                    title="Value (Base 100)",
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.05)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance table
            st.markdown("### 🏆 Performance Attribution")
            p_data = {
                "Metric": ["CAGR", "Vol (Ann)", "Sharpe", "MaxDD", "Calmar", "Cumul"],
                "Strategy": [f"{met_s['CAGR']:.1f}%", f"{met_s['Vol']:.1f}%", f"{met_s['Sharpe']:.2f}", f"{met_s['MaxDD']:.1f}%", f"{met_s['Calmar']:.2f}", f"{met_s['Cumul']:.1f}%"],
                f"Risk ({tickers[0]})": [f"{met_x2['CAGR']:.1f}%", f"{met_x2['Vol']:.1f}%", f"{met_x2['Sharpe']:.2f}", f"{met_x2['MaxDD']:.1f}%", f"{met_x2['Calmar']:.2f}", f"{met_x2['Cumul']:.1f}%"],
                f"Safe ({tickers[1]})": [f"{met_x1['CAGR']:.1f}%", f"{met_x1['Vol']:.1f}%", f"{met_x1['Sharpe']:.2f}", f"{met_x1['MaxDD']:.1f}%", f"{met_x1['Calmar']:.2f}", f"{met_x1['Cumul']:.1f}%"]
            }
            st.markdown(pd.DataFrame(p_data).style.hide(axis="index").to_html(), unsafe_allow_html=True)
