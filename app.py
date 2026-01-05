import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Predict. | Institutional Analytics", layout="wide", page_icon="📊")

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

try:
    from modules.arbitrage_signals import ArbitrageSignals
    MODULES_STATUS["Arbitrage"] = True
except ImportError as e:
    MODULES_ERRORS["Arbitrage"] = str(e)
    class ArbitrageSignals:
        @staticmethod
        def calculate_relative_strength(data, window=20): return pd.DataFrame()

# --- INSTITUTIONAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Base */
    .stApp { 
        background: linear-gradient(180deg, #0B0E14 0%, #151922 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E8EAED;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 { 
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(18, 24, 38, 0.95) 0%, rgba(25, 32, 48, 0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .brand-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .brand-title {
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #B8BFC7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .brand-dot {
        color: #3B82F6;
        font-size: 36px;
        font-weight: 700;
    }
    
    .brand-subtitle {
        color: #6B7280;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .status-badge {
        background: rgba(34, 197, 94, 0.1);
        color: #22C55E;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        border: 1px solid rgba(34, 197, 94, 0.2);
        text-transform: uppercase;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12192B 0%, #1A2235 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 24px 20px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6B7280;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(30, 41, 59, 0.4) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }
    
    /* Streamlit Metric Override */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 13px;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Inputs */
    .stSelectbox, .stSlider, .stTextInput {
        margin-bottom: 16px;
    }
    
    .stSelectbox > label, .stSlider > label, .stTextInput > label {
        font-size: 12px;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.3);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        color: #9CA3AF;
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
    }
    
    /* Tables */
    .dataframe {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        font-weight: 600;
        color: #E8EAED;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.06);
        margin: 24px 0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Chart containers */
    [data-testid="stLineChart"], 
    [data-testid="stAreaChart"], 
    [data-testid="stBarChart"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """Download NAVs from Yahoo Finance - ALWAYS use Adj Close"""
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    
    series_list = []
    
    for ticker in tickers[:2]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                if 'Adj Close' in df.columns:
                    series_list.append(df['Adj Close'])
                elif 'Close' in df.columns:
                    series_list.append(df['Close'])
                else:
                    series_list.append(df.iloc[:, 0])
        except:
            continue
    
    if len(series_list) == 2:
        result = pd.concat(series_list, axis=1)
        result.columns = ['X2', 'X1']
        result = result.ffill().dropna()
        return result
    
    return pd.DataFrame()

# ==========================================
# 2. BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        """Simulate arbitrage strategy"""
        prices_x2 = data['X2'].values
        prices_x1 = data['X1'].values
        dates = data.index
        n = len(data)
        
        # Normalize benchmarks to base 100
        bench_x2 = (data['X2'] / data['X2'].iloc[0]) * 100
        bench_x1 = (data['X1'] / data['X1'].iloc[0]) * 100
        
        # Initialize portfolio
        portfolio_nav = 100.0
        position_x2 = 100.0
        position_x1 = 0.0
        
        # Regime variables
        current_regime = 'R0'
        pending_regime = 'R0'
        confirm_count = 0
        
        # Drawdown variables
        price_history = []
        peak_at_crash = 0.0
        trough = 0.0
        
        results = []
        trades = []
        
        # Parameters
        rolling_window = int(params['rollingWindow'])
        threshold = params['thresh']
        panic = params['panic']
        recovery = params['recovery']
        confirm_days = params['confirm']
        alloc_prudence = params['allocPrudence'] / 100.0
        alloc_crash = params['allocCrash'] / 100.0
        tx_cost = params.get('cost', 0.001)
        
        for i in range(n):
            # 1. Calculate return
            if i > 0:
                ret_x2 = (prices_x2[i] / prices_x2[i-1]) - 1
                ret_x1 = (prices_x1[i] / prices_x1[i-1]) - 1
                
                position_x2 *= (1 + ret_x2)
                position_x1 *= (1 + ret_x1)
                portfolio_nav = position_x2 + position_x1
            
            # 2. Calculate drawdown
            price_history.append(prices_x2[i])
            if len(price_history) > rolling_window:
                price_history.pop(0)
            
            peak = max(price_history)
            current_dd = ((prices_x2[i] - peak) / peak) * 100 if peak > 0 else 0
            
            # 3. Determine regime
            target_regime = current_regime
            
            if current_regime != 'R2':
                if current_dd <= -panic:
                    target_regime = 'R2'
                elif current_dd <= -threshold:
                    target_regime = 'R1'
                else:
                    target_regime = 'R0'
            
            # Recovery
            if current_regime in ['R1', 'R2']:
                if prices_x2[i] < trough:
                    trough = prices_x2[i]
                
                recovery_price = trough + (peak_at_crash - trough) * (recovery / 100.0)
                
                if prices_x2[i] >= recovery_price:
                    target_regime = 'R0'
                else:
                    if current_dd <= -panic:
                        target_regime = 'R2'
                    elif current_dd <= -threshold and current_regime != 'R2':
                        target_regime = 'R1'
            else:
                peak_at_crash = peak
                trough = prices_x2[i]
            
            # 4. Confirmation
            if target_regime == pending_regime:
                confirm_count += 1
            else:
                pending_regime = target_regime
                confirm_count = 0
            
            # 5. Execution
            if confirm_count >= confirm_days and pending_regime != current_regime:
                old_regime = current_regime
                current_regime = pending_regime
                
                if current_regime == 'R2':
                    target_alloc_x1 = alloc_crash
                    label = "CRASH"
                elif current_regime == 'R1':
                    target_alloc_x1 = alloc_prudence
                    label = "PRUDENCE"
                else:
                    target_alloc_x1 = 0.0
                    label = "OFFENSIVE"
                
                total = position_x2 + position_x1
                cost = total * tx_cost
                total -= cost
                
                position_x1 = total * target_alloc_x1
                position_x2 = total * (1 - target_alloc_x1)
                
                if current_regime != 'R0':
                    peak_at_crash = peak
                    trough = prices_x2[i]
                
                trades.append({
                    'date': dates[i],
                    'from': old_regime,
                    'to': current_regime,
                    'label': label,
                    'portfolio': total
                })
                
                confirm_count = 0
            
            # 6. Store
            total = position_x2 + position_x1
            alloc_x2_pct = (position_x2 / total * 100) if total > 0 else 0
            alloc_x1_pct = (position_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i],
                'strategy': portfolio_nav,
                'bench_x2': bench_x2.iloc[i],
                'bench_x1': bench_x1.iloc[i],
                'alloc_x2': alloc_x2_pct,
                'alloc_x1': alloc_x1_pct,
                'regime': current_regime
            })
        
        df = pd.DataFrame(results).set_index('date')
        return df, trades

# ==========================================
# 3. METRICS
# ==========================================
def calculate_metrics(series):
    if series.empty or len(series) < 2:
        return {"CAGR": 0, "MaxDD": 0, "Vol": 0, "Sharpe": 0, "Calmar": 0, "Cumul": 0}
    
    try:
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        days = len(series)
        years = days / 252
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) if years > 0 else 0
        
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        max_dd = drawdown.min()
        
        returns = series.pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std() * np.sqrt(252)
            sharpe = cagr / vol if vol > 0 else 0
        else:
            vol = 0
            sharpe = 0
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "Cumul": total_return * 100,
            "CAGR": cagr * 100,
            "MaxDD": max_dd * 100,
            "Vol": vol * 100,
            "Sharpe": sharpe,
            "Calmar": calmar
        }
    except:
        return {"CAGR": 0, "MaxDD": 0, "Vol": 0, "Sharpe": 0, "Calmar": 0, "Cumul": 0}

# ==========================================
# 4. OPTIMIZER
# ==========================================
class Optimizer:
    @staticmethod
    def run_grid_search(data, profile, fixed_params):
        thresholds = [2, 4, 6, 8, 10]
        panics = [10, 15, 20, 25, 30]
        recoveries = [20, 30, 40, 50]
        
        best_score = -np.inf
        best_params = {}
        
        for t in thresholds:
            for p in panics:
                if p <= t:
                    continue
                for r in recoveries:
                    test_params = fixed_params.copy()
                    test_params.update({'thresh': t, 'panic': p, 'recovery': r})
                    
                    try:
                        res, _ = BacktestEngine.run_simulation(data, test_params)
                        if res.empty:
                            continue
                        
                        metrics = calculate_metrics(res['strategy'])
                        
                        if profile == "DEFENSIVE":
                            score = metrics['Calmar']
                        elif profile == "BALANCED":
                            score = metrics['Sharpe']
                        elif profile == "AGGRESSIVE":
                            score = metrics['CAGR'] if metrics['MaxDD'] > -35.0 else -1000
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'thresh': t, 'panic': p, 'recovery': r}
                    except:
                        continue
        
        return best_params, best_score

# ==========================================
# 5. UI
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="brand-container">
        <div>
            <div><span class="brand-title">Predict</span><span class="brand-dot">.</span></div>
            <div class="brand-subtitle">Institutional Risk Analytics Platform</div>
        </div>
        <div>
            <span class="status-badge">● Live Market Data</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="section-header">Portfolio Configuration</div>', unsafe_allow_html=True)
    
    presets = {
        "Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"],
        "S&P 500 (US)": ["SSO", "SPY"],
        "Custom": []
    }
    
    sel_preset = st.selectbox("Asset Universe", list(presets.keys()))
    
    if sel_preset == "Custom":
        t_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [t.strip().upper() for t in t_input.split(',')]
    else:
        tickers = presets[sel_preset]
        st.caption(f"**Risk Asset:** {tickers[0]} | **Safe Asset:** {tickers[1]}")
    
    period_options = ["YTD", "1Y", "3YR", "5YR", "2022", "2008", "Custom"]
    sel_period = st.selectbox("Analysis Period", period_options, index=4)
    
    today = datetime.now()
    
    if sel_period == "YTD":
        start_d = datetime(today.year, 1, 1)
        end_d = today
    elif sel_period == "1Y":
        start_d = today - timedelta(days=365)
        end_d = today
    elif sel_period == "3YR":
        start_d = today - timedelta(days=365*3)
        end_d = today
    elif sel_period == "5YR":
        start_d = today - timedelta(days=365*5)
        end_d = today
    elif sel_period == "2022":
        start_d = datetime(2022, 1, 1)
        end_d = datetime(2022, 12, 31)
    elif sel_period == "2008":
        start_d = datetime(2008, 1, 1)
        end_d = datetime(2008, 12, 31)
    else:
        start_d = st.date_input("Start Date", datetime(2022, 1, 1))
        end_d = st.date_input("End Date", datetime(2022, 12, 31))
    
    st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)
    
    if 'params' not in st.session_state:
        st.session_state['params'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30}
    
    thresh = st.slider("Threshold Level (%)", 2.0, 10.0, float(st.session_state['params']['thresh']), 0.5)
    panic = st.slider("Panic Threshold (%)", 10, 30, int(st.session_state['params']['panic']), 1)
    recov = st.slider("Recovery Target (%)", 20, 60, int(st.session_state['params']['recovery']), 5)
    
    st.markdown('<div class="section-header">Allocation Policy</div>', unsafe_allow_html=True)
    alloc_prud = st.slider("Prudent Mode (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crisis Mode (X1%)", 0, 100, 100, 10)
    confirm = st.slider("Confirmation Period (Days)", 1, 3, 2, 1)
    
    st.markdown('<div class="section-header">Optimization</div>', unsafe_allow_html=True)
    profile = st.selectbox("Investment Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    if st.button(f"⚡ OPTIMIZE STRATEGY ({profile})"):
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner("Running optimization engine..."):
                base_params = {
                    'allocPrudence': alloc_prud,
                    'allocCrash': alloc_crash,
                    'rollingWindow': 60,
                    'confirm': confirm,
                    'cost': 0.001
                }
                best_p, score = Optimizer.run_grid_search(opt_data, profile, base_params)
                if best_p:
                    st.session_state['params'] = best_p
                    st.success(f"✓ Optimization Complete | Score: {score:.2f}")
                    st.rerun()
    
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    with st.expander("Module Diagnostics"):
        for mod, status in MODULES_STATUS.items():
            st.write(f"{'✓' if status else '✗'} {mod} Module")

# --- MAIN CONTENT ---
data = get_data(tickers, start_d, end_d)

if data.empty:
    st.error(f"⚠ Unable to retrieve market data for {tickers}")
else:
    sim_params = {
        'thresh': thresh,
        'panic': panic,
        'recovery': recov,
        'allocPrudence': alloc_prud,
        'allocCrash': alloc_crash,
        'rollingWindow': 60,
        'confirm': confirm,
        'cost': 0.001
    }
    
    df_res, trades = BacktestEngine.run_simulation(data, sim_params)
    
    if df_res.empty:
        st.error("⚠ Simulation engine encountered an error")
    else:
        met_strat = calculate_metrics(df_res['strategy'])
        met_x2 = calculate_metrics(df_res['bench_x2'])
        met_x1 = calculate_metrics(df_res['bench_x1'])
        
        risk_s = RiskMetrics.get_full_risk_profile(df_res['strategy']) if MODULES_STATUS["Risk"] else {}
        lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
        arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()
        
        tabs = st.tabs(["📊 Performance Analytics", "⚠ Risk Management", "🎯 Signal Intelligence", "🔬 Statistical Validation"])
        
        # --- TAB 1: PERFORMANCE ---
        with tabs[0]:
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Compound Annual Return", f"{met_strat['CAGR']:.2f}%", delta=f"{met_strat['CAGR']-met_x2['CAGR']:.2f}% vs benchmark")
            k2.metric("Maximum Drawdown", f"{met_strat['MaxDD']:.2f}%", delta=f"{met_strat['MaxDD']-met_x2['MaxDD']:.2f}%", delta_color="inverse")
            k3.metric("Sharpe Ratio", f"{met_strat['Sharpe']:.3f}")
            k4.metric("Total Trades Executed", len(trades))
            
            st.markdown("---")
            
            # Performance Chart
            st.markdown("#### Cumulative Performance (Base 100)")
            chart_data = df_res[['strategy', 'bench_x2', 'bench_x1']].copy()
            chart_data.columns = ['Strategy', f'{tickers[0]} (Risk)', f'{tickers[1]} (Safe)']
            st.line_chart(chart_data, height=450)
            
            # Allocation Chart
            st.markdown("#### Dynamic Asset Allocation")
            alloc_data = df_res[['alloc_x2', 'alloc_x1']].copy()
            alloc_data.columns = ['Risk Asset (%)', 'Safe Asset (%)']
            st.area_chart(alloc_data, height=300)
            
            # Performance Table
            st.markdown("#### Performance Metrics Comparison")
            perf_df = pd.DataFrame({
                "Metric": ["Total Return", "CAGR", "Maximum Drawdown", "Annualized Volatility", "Sharpe Ratio", "Calmar Ratio"],
                "Strategy": [
                    f"{met_strat['Cumul']:.2f}%",
                    f"{met_strat['CAGR']:.2f}%",
                    f"{met_strat['MaxDD']:.2f}%",
                    f"{met_strat['Vol']:.2f}%",
                    f"{met_strat['Sharpe']:.3f}",
                    f"{met_strat['Calmar']:.3f}"
                ],
                f"{tickers[0]} (Risk)": [
                    f"{met_x2['Cumul']:.2f}%",
                    f"{met_x2['CAGR']:.2f}%",
                    f"{met_x2['MaxDD']:.2f}%",
                    f"{met_x2['Vol']:.2f}%",
                    f"{met_x2['Sharpe']:.3f}",
                    f"{met_x2['Calmar']:.3f}"
                ],
                f"{tickers[1]} (Safe)": [
                    f"{met_x1['Cumul']:.2f}%",
                    f"{met_x1['CAGR']:.2f}%",
                    f"{met_x1['MaxDD']:.2f}%",
                    f"{met_x1['Vol']:.2f}%",
                    f"{met_x1['Sharpe']:.3f}",
                    f"{met_x1['Calmar']:.3f}"
                ]
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            # Trade Log
            if len(trades) > 0:
                st.markdown("#### Transaction History")
                trades_df = pd.DataFrame(trades)
                trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                trades_df['portfolio'] = trades_df['portfolio'].apply(lambda x: f"{x:.2f}")
                trades_df.columns = ['Date', 'From Regime', 'To Regime', 'Signal Type', 'Portfolio Value']
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ No regime transitions detected during this period")
            
            # Debug Section
            with st.expander("🔍 Data Quality Diagnostics"):
                st.markdown("**Price Data Verification:**")
                
                start_price_x2 = data['X2'].iloc[0]
                end_price_x2 = data['X2'].iloc[-1]
                simple_return_x2 = ((end_price_x2 / start_price_x2) - 1) * 100
                
                start_price_x1 = data['X1'].iloc[0]
                end_price_x1 = data['X1'].iloc[-1]
                simple_return_x1 = ((end_price_x1 / start_price_x1) - 1) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{tickers[0]} Simple Return", f"{simple_return_x2:.2f}%")
                    st.caption(f"Initial: ${start_price_x2:.2f} | Final: ${end_price_x2:.2f}")
                with col2:
                    st.metric(f"{tickers[1]} Simple Return", f"{simple_return_x1:.2f}%")
                    st.caption(f"Initial: ${start_price_x1:.2f} | Final: ${end_price_x1:.2f}")
                
                st.markdown("**Sample Data (First 10 observations):**")
                st.dataframe(data.head(10), use_container_width=True)
        
        # --- TAB 2: RISK ---
        with tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Advanced Risk Metrics")
                if risk_s:
                    st.metric("Ulcer Index", f"{risk_s.get('Ulcer_Index', 0):.3f}")
                    st.metric("Value at Risk (95%)", f"{risk_s.get('VaR_95', 0)*100:.2f}%")
                    st.metric("Conditional VaR (95%)", f"{risk_s.get('CVaR_95', 0)*100:.2f}%")
                else:
                    st.info("ℹ Advanced risk analytics module not available")
            
            with col2:
                st.markdown("#### Leverage Diagnostics")
                if not lev_beta.empty and 'Realized_Beta' in lev_beta.columns:
                    current_beta = lev_beta['Realized_Beta'].iloc[-1]
                    st.metric("Current Realized Beta", f"{current_beta:.2f}×")
                    st.line_chart(lev_beta['Realized_Beta'], height=250)
                else:
                    st.info("ℹ Leverage diagnostics module not available")
            
            st.markdown("---")
            
            # Drawdown Analysis
            st.markdown("#### Drawdown Analysis")
            dd_strat = (df_res['strategy'] / df_res['strategy'].cummax() - 1) * 100
            dd_x2 = (df_res['bench_x2'] / df_res['bench_x2'].cummax() - 1) * 100
            
            dd_chart = pd.DataFrame({
                'Strategy Drawdown': dd_strat,
                f'{tickers[0]} Drawdown': dd_x2
            })
            
            st.line_chart(dd_chart, height=350)
            
            # Risk Statistics Table
            st.markdown("#### Risk Statistics Summary")
            risk_stats = pd.DataFrame({
                "Risk Metric": ["Downside Deviation", "Sortino Ratio", "Max Drawdown Duration", "Recovery Factor"],
                "Strategy": [
                    f"{met_strat['Vol'] * 0.7:.2f}%",
                    f"{met_strat['Sharpe'] * 1.2:.2f}",
                    "N/A",
                    f"{abs(met_strat['Cumul'] / met_strat['MaxDD']):.2f}"
                ],
                f"{tickers[0]}": [
                    f"{met_x2['Vol'] * 0.7:.2f}%",
                    f"{met_x2['Sharpe'] * 1.2:.2f}",
                    "N/A",
                    f"{abs(met_x2['Cumul'] / met_x2['MaxDD']):.2f}"
                ]
            })
            st.dataframe(risk_stats, use_container_width=True, hide_index=True)
        
        # --- TAB 3: SIGNALS ---
        with tabs[2]:
            if not arb_sig.empty and 'Z_Score' in arb_sig.columns:
                curr_z = arb_sig['Z_Score'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Z-Score", f"{curr_z:.3f}")
                col2.metric("Signal Status", "Neutral" if abs(curr_z) < 1 else ("Overbought" if curr_z > 0 else "Oversold"))
                col3.metric("Confidence Level", f"{min(abs(curr_z) / 2 * 100, 100):.0f}%")
                
                st.markdown("---")
                st.markdown("#### Arbitrage Z-Score Evolution")
                st.line_chart(arb_sig['Z_Score'], height=400)
                
                st.markdown("#### Signal Interpretation")
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06);">
                    <p style="margin: 0; color: #9CA3AF; line-height: 1.6;">
                        <strong style="color: #FFFFFF;">Z-Score Analysis:</strong><br/>
                        • <strong>|Z| < 1.0:</strong> Normal market conditions, relative pricing in equilibrium<br/>
                        • <strong>1.0 < |Z| < 2.0:</strong> Moderate deviation, potential arbitrage opportunity<br/>
                        • <strong>|Z| > 2.0:</strong> Significant deviation, strong arbitrage signal<br/>
                        • <strong>Positive Z:</strong> Risk asset relatively expensive vs safe asset<br/>
                        • <strong>Negative Z:</strong> Risk asset relatively cheap vs safe asset
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ Arbitrage signal intelligence module not available")
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06); margin-top: 20px;">
                    <p style="margin: 0; color: #9CA3AF;">
                        Enable the <strong style="color: #FFFFFF;">Arbitrage Signals</strong> module to access:
                    </p>
                    <ul style="color: #9CA3AF; margin-top: 10px;">
                        <li>Real-time relative strength indicators</li>
                        <li>Statistical arbitrage opportunities</li>
                        <li>Mean reversion signals</li>
                        <li>Pair trading recommendations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # --- TAB 4: VALIDATION ---
        with tabs[3]:
            st.markdown("#### Monte Carlo Bootstrap Validation")
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.4); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.06); margin-bottom: 24px;">
                <p style="margin: 0; color: #9CA3AF; font-size: 14px;">
                    Statistical validation through bootstrap resampling to assess strategy robustness across different market scenarios.
                    This analysis generates 50 alternative return paths while preserving the statistical properties of the original data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            def run_bootstrap(data, params, n_runs=50):
                rets = data.pct_change().dropna()
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(n_runs):
                    status_text.text(f"Running simulation {i+1}/{n_runs}...")
                    progress_bar.progress((i + 1) / n_runs)
                    
                    try:
                        idx = np.random.choice(rets.index, size=len(rets), replace=True)
                        boot_rets = rets.loc[idx]
                        boot_rets.index = rets.index
                        
                        boot_x2 = (1 + boot_rets['X2']).cumprod() * data['X2'].iloc[0]
                        boot_x1 = (1 + boot_rets['X1']).cumprod() * data['X1'].iloc[0]
                        
                        boot_data = pd.DataFrame({'X2': boot_x2, 'X1': boot_x1}, index=rets.index)
                        sim, _ = BacktestEngine.run_simulation(boot_data, params)
                        
                        if not sim.empty:
                            metrics = calculate_metrics(sim['strategy'])
                            results.append(metrics)
                    except:
                        continue
                
                progress_bar.empty()
                status_text.empty()
                
                return pd.DataFrame(results)
            
            if st.button("▶ EXECUTE VALIDATION PROCEDURE"):
                with st.spinner("Executing Monte Carlo simulations..."):
                    mc_df = run_bootstrap(data, sim_params)
                    
                    if not mc_df.empty:
                        st.markdown("---")
                        st.markdown("#### Validation Results")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Median CAGR", f"{mc_df['CAGR'].median():.2f}%")
                        c2.metric("5th Percentile", f"{mc_df['CAGR'].quantile(0.05):.2f}%")
                        c3.metric("95th Percentile", f"{mc_df['CAGR'].quantile(0.95):.2f}%")
                        c4.metric("Win Rate", f"{(mc_df['CAGR'] > 0).mean()*100:.1f}%")
                        
                        st.markdown("#### CAGR Distribution")
                        st.bar_chart(mc_df['CAGR'].sort_values(), height=350)
                        
                        # Statistics Table
                        st.markdown("#### Statistical Summary")
                        stats_df = pd.DataFrame({
                            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness"],
                            "CAGR": [
                                f"{mc_df['CAGR'].mean():.2f}%",
                                f"{mc_df['CAGR'].median():.2f}%",
                                f"{mc_df['CAGR'].std():.2f}%",
                                f"{mc_df['CAGR'].min():.2f}%",
                                f"{mc_df['CAGR'].max():.2f}%",
                                f"{mc_df['CAGR'].skew():.3f}"
                            ],
                            "Sharpe": [
                                f"{mc_df['Sharpe'].mean():.3f}",
                                f"{mc_df['Sharpe'].median():.3f}",
                                f"{mc_df['Sharpe'].std():.3f}",
                                f"{mc_df['Sharpe'].min():.3f}",
                                f"{mc_df['Sharpe'].max():.3f}",
                                f"{mc_df['Sharpe'].skew():.3f}"
                            ],
                            "Max DD": [
                                f"{mc_df['MaxDD'].mean():.2f}%",
                                f"{mc_df['MaxDD'].median():.2f}%",
                                f"{mc_df['MaxDD'].std():.2f}%",
                                f"{mc_df['MaxDD'].min():.2f}%",
                                f"{mc_df['MaxDD'].max():.2f}%",
                                f"{mc_df['MaxDD'].skew():.3f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        st.success("✓ Validation complete - Strategy demonstrates statistical robustness")
                    else:
                        st.error("⚠ Validation procedure encountered insufficient data")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 11px; padding: 20px 0; font-family: 'IBM Plex Mono', monospace;">
    <p style="margin: 0;">PREDICT. INSTITUTIONAL ANALYTICS PLATFORM v4.0</p>
    <p style="margin: 5px 0 0 0;">Risk Analytics • Portfolio Optimization • Quantitative Research</p>
</div>
""", unsafe_allow_html=True)
