import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 0. CONFIGURATION
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

try:
    from modules.arbitrage_signals import ArbitrageSignals
    MODULES_STATUS["Arbitrage"] = True
except ImportError as e:
    MODULES_ERRORS["Arbitrage"] = str(e)
    class ArbitrageSignals:
        @staticmethod
        def calculate_relative_strength(data, window=20): return pd.DataFrame()

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { background-color: #0A0A0F; font-family: 'Inter', sans-serif; color: #E0E0E0; }
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    .header-container {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A3E 100%);
        border-radius: 12px; padding: 25px; border: 1px solid rgba(255,255,255,0.08); 
        margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .title-text { font-weight: 800; font-size: 32px; letter-spacing: -1px; color: #FFFFFF; }
    .title-dot { color: #A855F7; font-size: 32px; font-weight: 800; }
    header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """Récupère les prix réels (Adj Close) des deux tickers et garde uniquement les jours de trading communs"""
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    
    series_list = []
    
    for ticker in tickers[:2]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Toujours utiliser Adj Close si dispo
                series_list.append(df['Adj Close'])
        except Exception as e:
            print(f"Erreur sur {ticker}: {e}")
            continue
    
    if len(series_list) == 2:
        result = pd.concat(series_list, axis=1)
        result.columns = ['X2', 'X1']  # X2 = risque, X1 = safe
        result = result.dropna()        # IMPORTANT : seulement les jours où les deux tickers ont des données
        return result
    
    return pd.DataFrame()

# ==========================================
# 2. BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        """Simule la stratégie d'arbitrage"""
        prices_x2 = data['X2'].values
        prices_x1 = data['X1'].values
        dates = data.index
        n = len(data)
        
        # Normaliser les benchmarks en base 100
        bench_x2 = (data['X2'] / data['X2'].iloc[0]) * 100
        bench_x1 = (data['X1'] / data['X1'].iloc[0]) * 100
        
        # Initialiser le portfolio
        portfolio_nav = 100.0
        position_x2 = 100.0
        position_x1 = 0.0
        
        # Variables de régime
        current_regime = 'R0'
        pending_regime = 'R0'
        confirm_count = 0
        
        # Variables de drawdown
        price_history = []
        peak_at_crash = 0.0
        trough = 0.0
        
        results = []
        trades = []
        
        # Paramètres
        rolling_window = int(params['rollingWindow'])
        threshold = params['thresh']
        panic = params['panic']
        recovery = params['recovery']
        confirm_days = params['confirm']
        alloc_prudence = params['allocPrudence'] / 100.0
        alloc_crash = params['allocCrash'] / 100.0
        tx_cost = params.get('cost', 0.001)
        
        for i in range(n):
            # 1. Calculer le return
            if i > 0:
                ret_x2 = (prices_x2[i] / prices_x2[i-1]) - 1
                ret_x1 = (prices_x1[i] / prices_x1[i-1]) - 1
                
                position_x2 *= (1 + ret_x2)
                position_x1 *= (1 + ret_x1)
                portfolio_nav = position_x2 + position_x1
            
            # 2. Calculer le drawdown
            price_history.append(prices_x2[i])
            if len(price_history) > rolling_window:
                price_history.pop(0)
            
            peak = max(price_history)
            current_dd = ((prices_x2[i] - peak) / peak) * 100 if peak > 0 else 0
            
            # 3. Déterminer le régime
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
            
            # 5. Exécution
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
                    label = "OFFENSIF"
                
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
            
            # 6. Stocker
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
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span class="title-text">Predict</span><span class="title-dot">.</span>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V4.0 • STREAMLIT CHARTS</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(168, 85, 247, 0.1); color:#A855F7; padding:5px 10px; border-radius:4px; font-size:11px; border:1px solid rgba(168, 85, 247, 0.3);">LIVE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sidebar, col_main = st.columns([1, 3])

# --- SIDEBAR ---
with col_sidebar:
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
    sel_period = st.selectbox("Period", period_options, index=4)
    
    today = datetime.now()

    # Définition des dates selon le preset
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
        start_d = st.date_input("Start", datetime(2022, 1, 1))
        end_d = st.date_input("End", datetime(2022, 12, 31))

    # ----------------------------
    # Ajustement automatique aux jours de trading
    # ----------------------------
    def adjust_to_trading_days(tickers, start, end):
        data = yf.download(tickers[:2], start=start, end=end, progress=False)['Adj Close']
        data = data.dropna()
        if data.empty:
            return start, end
        return data.index[0].to_pydatetime(), data.index[-1].to_pydatetime()

    start_d, end_d = adjust_to_trading_days(tickers, start_d, end_d)

    # ----------------------------
    # STRATEGY PARAMETERS
    # ----------------------------
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
    
    if st.button(f"🎯 OPTIMIZE ({profile})"):
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner("Running optimization..."):
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
                    st.success(f"✅ Optimized! Score: {score:.2f}")
                    st.rerun()
    
    st.markdown("---")
    with st.expander("📦 Modules"):
        for mod, status in MODULES_STATUS.items():
            st.write(f"{'✅' if status else '❌'} {mod}")


# --- MAIN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty:
        st.error(f"❌ No data for {tickers}")
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
            st.error("❌ Simulation failed")
        else:
            met_strat = calculate_metrics(df_res['strategy'])
            met_x2 = calculate_metrics(df_res['bench_x2'])
            met_x1 = calculate_metrics(df_res['bench_x1'])
            
            risk_s = RiskMetrics.get_full_risk_profile(df_res['strategy']) if MODULES_STATUS["Risk"] else {}
            lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
            arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()
            
            tabs = st.tabs(["📈 Performance", "⚠️ Risk", "🎯 Signals", "🛡️ Validation"])
            
            # --- TAB 1: PERFORMANCE ---
            with tabs[0]:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("CAGR", f"{met_strat['CAGR']:.1f}%", delta=f"{met_strat['CAGR']-met_x2['CAGR']:.1f}% vs X2")
                k2.metric("MaxDD", f"{met_strat['MaxDD']:.1f}%", delta=f"{met_strat['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")
                k3.metric("Sharpe", f"{met_strat['Sharpe']:.2f}")
                k4.metric("Trades", len(trades))
                
                if len(trades) == 0:
                    st.warning("⚠️ No trades executed")
                
                st.markdown("### 📊 Performance Chart (Base 100)")
                
                # Préparer les données pour le graphique
                chart_data = df_res[['strategy', 'bench_x2', 'bench_x1']].copy()
                chart_data.columns = ['Strategy', 'X2 (Risk)', 'X1 (Safe)']
                
                # Graphique linéaire natif Streamlit
                st.line_chart(chart_data, height=400)
                
                st.markdown("### 📊 Allocation (%)")
                
                # Graphique d'allocation
                alloc_data = df_res[['alloc_x2', 'alloc_x1']].copy()
                alloc_data.columns = ['X2 (Risk)', 'X1 (Safe)']
                
                st.area_chart(alloc_data, height=250)
                
                st.markdown("### 🏆 Performance Table")
                perf_df = pd.DataFrame({
                    "Metric": ["CAGR", "MaxDD", "Vol", "Sharpe", "Calmar", "Cumul"],
                    "Strategy": [
                        f"{met_strat['CAGR']:.1f}%",
                        f"{met_strat['MaxDD']:.1f}%",
                        f"{met_strat['Vol']:.1f}%",
                        f"{met_strat['Sharpe']:.2f}",
                        f"{met_strat['Calmar']:.2f}",
                        f"{met_strat['Cumul']:.1f}%"
                    ],
                    "X2 (Risk)": [
                        f"{met_x2['CAGR']:.1f}%",
                        f"{met_x2['MaxDD']:.1f}%",
                        f"{met_x2['Vol']:.1f}%",
                        f"{met_x2['Sharpe']:.2f}",
                        f"{met_x2['Calmar']:.2f}",
                        f"{met_x2['Cumul']:.1f}%"
                    ],
                    "X1 (Safe)": [
                        f"{met_x1['CAGR']:.1f}%",
                        f"{met_x1['MaxDD']:.1f}%",
                        f"{met_x1['Vol']:.1f}%",
                        f"{met_x1['Sharpe']:.2f}",
                        f"{met_x1['Calmar']:.2f}",
                        f"{met_x1['Cumul']:.1f}%"
                    ]
                })
                st.dataframe(perf_df, use_container_width=True)
                
                if len(trades) > 0:
                    st.markdown("### 📋 Trade Log")
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df, use_container_width=True)
            
            # --- TAB 2: RISK ---
            with tabs[1]:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("### ⚠️ Risk Metrics")
                    if risk_s:
                        st.metric("Ulcer Index", f"{risk_s.get('Ulcer_Index', 0):.2f}")
                        st.metric("VaR 95%", f"{risk_s.get('VaR_95', 0)*100:.2f}%")
                        st.metric("CVaR 95%", f"{risk_s.get('CVaR_95', 0)*100:.2f}%")
                    else:
                        st.info("Risk module not available")
                
                with c2:
                    st.markdown("### ⚙️ Leverage")
                    if not lev_beta.empty and 'Realized_Beta' in lev_beta.columns:
                        st.metric("Realized Beta", f"{lev_beta['Realized_Beta'].iloc[-1]:.2f}x")
                        st.line_chart(lev_beta['Realized_Beta'], height=200)
                    else:
                        st.info("Leverage module not available")
                
                st.markdown("### 🌊 Drawdown Chart")
                dd_strat = (df_res['strategy'] / df_res['strategy'].cummax() - 1) * 100
                dd_x2 = (df_res['bench_x2'] / df_res['bench_x2'].cummax() - 1) * 100
                
                dd_chart = pd.DataFrame({
                    'Strategy': dd_strat,
                    'X2 (Risk)': dd_x2
                })
                
                st.line_chart(dd_chart, height=300)
            
            # --- TAB 3: SIGNALS ---
            with tabs[2]:
                if not arb_sig.empty and 'Z_Score' in arb_sig.columns:
                    st.markdown("### 🎯 Arbitrage Z-Score")
                    curr_z = arb_sig['Z_Score'].iloc[-1]
                    st.metric("Current Z", f"{curr_z:.2f}")
                    
                    st.line_chart(arb_sig['Z_Score'], height=300)
                else:
                    st.info("Arbitrage module not available")
            
            # --- TAB 4: VALIDATION ---
            with tabs[3]:
                st.markdown("### 🛡️ Bootstrap Monte Carlo")
                
                def run_bootstrap(data, params, n_runs=50):
                    rets = data.pct_change().dropna()
                    results = []
                    
                    for _ in range(n_runs):
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
                    
                    return pd.DataFrame(results)
                
                if st.button("▶️ RUN VALIDATION (50 runs)"):
                    with st.spinner("Running..."):
                        mc_df = run_bootstrap(data, sim_params)
                        
                        if not mc_df.empty:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Median CAGR", f"{mc_df['CAGR'].median():.1f}%")
                            c2.metric("5th Percentile", f"{mc_df['CAGR'].quantile(0.05):.1f}%")
                            c3.metric("Win Rate", f"{(mc_df['CAGR'] > 0).mean()*100:.0f}%")
                            
                            st.bar_chart(mc_df['CAGR'], height=300)
                        else:
                            st.error("Validation failed")
