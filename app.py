import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.express as px

from datetime import datetime, timedelta

from plotly.subplots import make_subplots



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

    .glass-card { 

        background: rgba(30, 30, 46, 0.6); border-radius: 12px; padding: 20px; 

        border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 20px;

        backdrop-filter: blur(10px);

    }

    .legend-box {

        background: rgba(30, 30, 46, 0.8); border-radius: 8px; padding: 15px;

        border: 1px solid rgba(168, 85, 247, 0.3); margin: 10px 0; font-size: 12px;

    }

    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }

    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; }

    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; }

    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1E1E2E; color: #A855F7; border: 1px solid #A855F7; }

    .stButton > button:hover { background-color: #A855F7; color: white; }

    header, footer { visibility: hidden; }

</style>

""", unsafe_allow_html=True)



# ==========================================

# 1. DATA ENGINE

# ==========================================

@st.cache_data(ttl=3600)

def get_data(tickers, start, end):

    """Télécharge les NAVs Yahoo et les nettoie"""

    if not tickers or len(tickers) < 2:

        return pd.DataFrame()

    

    series_list = []

    ticker_names = []

    

    for ticker in tickers[:2]:  # On prend les 2 premiers

        try:

            df = yf.download(ticker, start=start, end=end, progress=False)

            if not df.empty:

                # Prendre la colonne Close (ou Adj Close si dispo)

                if 'Close' in df.columns:

                    series_list.append(df['Close'])

                elif 'Adj Close' in df.columns:

                    series_list.append(df['Adj Close'])

                else:

                    series_list.append(df.iloc[:, 0])

                ticker_names.append(ticker)

        except:

            continue

    

    if len(series_list) == 2:

        result = pd.concat(series_list, axis=1, keys=ticker_names)

        result.columns = ['X2', 'X1']  # X2 = Risk, X1 = Safe

        result = result.ffill().dropna()

        return result

    

    return pd.DataFrame()



# ==========================================

# 2. BACKTEST ENGINE (CLEAN VERSION)

# ==========================================

class BacktestEngine:

    @staticmethod

    def run_simulation(data, params):

        """

        Simule la stratégie d'arbitrage entre X2 (risk) et X1 (safe)

        Retourne des NAVs en base 100 pour comparison

        """

        # Extraire les prix bruts

        prices_x2 = data['X2'].values

        prices_x1 = data['X1'].values

        dates = data.index

        n = len(data)

        

        # Normaliser les benchmarks en base 100 (buy & hold pur)

        bench_x2 = (data['X2'] / data['X2'].iloc[0]) * 100

        bench_x1 = (data['X1'] / data['X1'].iloc[0]) * 100

        

        # Initialiser le portfolio de la stratégie

        portfolio_nav = 100.0

        position_x2 = 100.0  # Commence 100% en X2

        position_x1 = 0.0     # 0% en X1

        

        # Variables de régime

        current_regime = 'R0'  # OFFENSIF

        pending_regime = 'R0'

        confirm_count = 0

        

        # Variables de drawdown

        price_history = []

        peak_at_crash = 0.0

        trough = 0.0

        

        # Résultats

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

        

        # Boucle principale

        for i in range(n):

            # 1. Calculer le return du jour

            if i > 0:

                ret_x2 = (prices_x2[i] / prices_x2[i-1]) - 1

                ret_x1 = (prices_x1[i] / prices_x1[i-1]) - 1

                

                # Mettre à jour les positions

                position_x2 *= (1 + ret_x2)

                position_x1 *= (1 + ret_x1)

                portfolio_nav = position_x2 + position_x1

            

            # 2. Calculer le drawdown sur X2

            price_history.append(prices_x2[i])

            if len(price_history) > rolling_window:

                price_history.pop(0)

            

            peak = max(price_history)

            current_dd = ((prices_x2[i] - peak) / peak) * 100 if peak > 0 else 0

            

            # 3. Déterminer le régime cible

            target_regime = current_regime

            

            if current_regime != 'R2':

                if current_dd <= -panic:

                    target_regime = 'R2'

                elif current_dd <= -threshold:

                    target_regime = 'R1'

                else:

                    target_regime = 'R0'

            

            # Logique de recovery

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

            

            # 5. Exécution du trade

            if confirm_count >= confirm_days and pending_regime != current_regime:

                old_regime = current_regime

                current_regime = pending_regime

                

                # Déterminer l'allocation cible

                if current_regime == 'R2':

                    target_alloc_x1 = alloc_crash

                    label = "CRASH"

                elif current_regime == 'R1':

                    target_alloc_x1 = alloc_prudence

                    label = "PRUDENCE"

                else:

                    target_alloc_x1 = 0.0

                    label = "OFFENSIF"

                

                # Rebalancer

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

                    'portfolio_value': total

                })

                

                confirm_count = 0

            

            # 6. Stocker les résultats

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

                'regime': current_regime,

                'drawdown': current_dd

            })

        

        df = pd.DataFrame(results).set_index('date')

        return df, trades



# ==========================================

# 3. METRICS

# ==========================================

def calculate_metrics(series):

    """Calcule les métriques de performance"""

    if series.empty or len(series) < 2:

        return {"CAGR": 0, "MaxDD": 0, "Vol": 0, "Sharpe": 0, "Calmar": 0, "Cumul": 0}

    

    try:

        # Return total

        total_return = (series.iloc[-1] / series.iloc[0]) - 1

        

        # CAGR

        days = len(series)

        years = days / 252

        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) if years > 0 else 0

        

        # Drawdown

        cum_max = series.cummax()

        drawdown = (series - cum_max) / cum_max

        max_dd = drawdown.min()

        

        # Volatilité et Sharpe

        returns = series.pct_change().dropna()

        if len(returns) > 0:

            vol = returns.std() * np.sqrt(252)

            sharpe = cagr / vol if vol > 0 else 0

        else:

            vol = 0

            sharpe = 0

        

        # Calmar

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

# 5. MONTE CARLO

# ==========================================

class MonteCarloForecaster:

    @staticmethod

    def run_forecast(data, params, n_simulations=500, forecast_days=252):

        try:

            returns_x2 = data['X2'].pct_change().dropna()

            returns_x1 = data['X1'].pct_change().dropna()

            

            if returns_x2.empty or returns_x1.empty:

                return pd.DataFrame()

            

            df_res, _ = BacktestEngine.run_simulation(data, params)

            if df_res.empty:

                return pd.DataFrame()

            

            current_value = df_res['strategy'].iloc[-1]

            mu_x2, sigma_x2 = returns_x2.mean(), returns_x2.std()

            mu_x1, sigma_x1 = returns_x1.mean(), returns_x1.std()

            

            simulations = []

            

            for _ in range(n_simulations):

                try:

                    sim_rets_x2 = np.random.normal(mu_x2, sigma_x2, forecast_days)

                    sim_rets_x1 = np.random.normal(mu_x1, sigma_x1, forecast_days)

                    

                    sim_px_x2 = [data['X2'].iloc[-1]]

                    sim_px_x1 = [data['X1'].iloc[-1]]

                    

                    for i in range(forecast_days):

                        sim_px_x2.append(sim_px_x2[-1] * (1 + sim_rets_x2[i]))

                        sim_px_x1.append(sim_px_x1[-1] * (1 + sim_rets_x1[i]))

                    

                    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='D')[1:]

                    sim_data = pd.DataFrame({'X2': sim_px_x2[1:], 'X1': sim_px_x1[1:]}, index=future_dates)

                    

                    sim_result, _ = BacktestEngine.run_simulation(sim_data, params)

                    

                    if not sim_result.empty:

                        sim_result['strategy'] = (sim_result['strategy'] / sim_result['strategy'].iloc[0]) * current_value

                        simulations.append(sim_result['strategy'])

                except:

                    continue

            

            if len(simulations) == 0:

                return pd.DataFrame()

            

            forecast_df = pd.DataFrame(simulations).T

            

            return pd.DataFrame({

                'median': forecast_df.median(axis=1),

                'p5': forecast_df.quantile(0.05, axis=1),

                'p25': forecast_df.quantile(0.25, axis=1),

                'p75': forecast_df.quantile(0.75, axis=1),

                'p95': forecast_df.quantile(0.95, axis=1)

            })

        except:

            return pd.DataFrame()



# ==========================================

# 6. UI

# ==========================================

st.markdown("""

<div class="header-container">

    <div style="display:flex; justify-content:space-between; align-items:center;">

        <div>

            <span class="title-text">Predict</span><span class="title-dot">.</span>

            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V3.0 • CLEAN REWRITE</p>

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

    sel_period = st.selectbox("Period", period_options, index=4)  # Default to 2022

    

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

        start_d = st.date_input("Start", datetime(2022, 1, 1))

        end_d = st.date_input("End", datetime(2022, 12, 31))

    

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

            with st.spinner("Running grid search..."):

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

    

    st.markdown('</div>', unsafe_allow_html=True)



# --- MAIN ---

with col_main:

    data = get_data(tickers, start_d, end_d)

    

    if data.empty:

        st.error(f"❌ No data for {tickers}")

    else:

        # Run simulation

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

            # Calculate metrics

            met_strat = calculate_metrics(df_res['strategy'])

            met_x2 = calculate_metrics(df_res['bench_x2'])

            met_x1 = calculate_metrics(df_res['bench_x1'])

            

            # External modules

            risk_s = RiskMetrics.get_full_risk_profile(df_res['strategy']) if MODULES_STATUS["Risk"] else {}

            lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()

            arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()

            

            tabs = st.tabs(["📈 Performance", "⚠️ Risk", "🎯 Signals", "🛡️ Validation", "🔮 Forecast"])

            

            # --- TAB 1: PERFORMANCE ---

            with tabs[0]:

                k1, k2, k3, k4 = st.columns(4)

                k1.metric("CAGR", f"{met_strat['CAGR']:.1f}%", delta=f"{met_strat['CAGR']-met_x2['CAGR']:.1f}% vs X2")

                k2.metric("MaxDD", f"{met_strat['MaxDD']:.1f}%", delta=f"{met_strat['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")

                k3.metric("Sharpe", f"{met_strat['Sharpe']:.2f}")

                k4.metric("Trades", len(trades))

                

                if len(trades) == 0:

                    st.warning("⚠️ No trades executed - try lowering thresholds")

                

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)

                

                # Create dual chart

                fig = make_subplots(

                    rows=2, cols=1,

                    row_heights=[0.7, 0.3],

                    vertical_spacing=0.05,

                    subplot_titles=("NAV Base 100", "Allocation (%)"),

                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]

                )

                

                # TOP: NAV Chart

                fig.add_trace(go.Scatter(

                    x=df_res.index, y=df_res['strategy'],

                    name='STRATEGY', line=dict(color='#A855F7', width=3),

                    fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'

                ), row=1, col=1)

                

                fig.add_trace(go.Scatter(

                    x=df_res.index, y=df_res['bench_x2'],

                    name='X2 (Risk)', line=dict(color='#ef4444', width=2, dash='dot')

                ), row=1, col=1)

                

                fig.add_trace(go.Scatter(

                    x=df_res.index, y=df_res['bench_x1'],

                    name='X1 (Safe)', line=dict(color='#10b981', width=2, dash='dot')

                ), row=1, col=1)

                

                # Add trade markers

                for t in trades:

                    if t['date'] in df_res.index:

                        color = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')

                        fig.add_annotation(

                            x=t['date'], y=df_res.loc[t['date']]['strategy'],

                            text="▼" if t['to'] != 'R0' else "▲",

                            showarrow=False, font=dict(color=color, size=16),

                            row=1, col=1

                        )

                

                # BOTTOM: Allocation

                fig.add_trace(go.Scatter(

                    x=df_res.index, y=df_res['alloc_x2'],

                    name='X2 %', mode='none',

                    fillcolor='rgba(239, 68, 68, 0.6)', stackgroup='one'

                ), row=2, col=1)

                

                fig.add_trace(go.Scatter(

                    x=df_res.index, y=df_res['alloc_x1'],

                    name='X1 %', mode='none',

                    fillcolor='rgba(16, 185, 129, 0.6)', stackgroup='one'

                ), row=2, col=1)

                

                fig.update_xaxes(showgrid=False, linecolor='#333')

                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

                fig.update_yaxes(range=[0, 100], row=2, col=1)

                

                fig.update_layout(

                    paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',

                    font=dict(family="Inter", color='#E0E0E0'),

                    height=650, margin=dict(l=40, r=40, t=40, b=40),

                    hovermode="x unified", showlegend=True,

                    legend=dict(orientation="h", y=1.08, x=0)

                )

                

                st.plotly_chart(fig, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

                

                st.markdown("### 🏆 Performance Table")

                perf_data = {

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

                }

                st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

                

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

                    if not lev_beta.empty:

                        st.metric("Realized Beta", f"{lev_beta['Realized_Beta'].iloc[-1]:.2f}x")

                    else:

                        st.info("Leverage module not available")

                

                st.markdown("### 🌊 Drawdown Chart")

                dd_strat = (df_res['strategy'] / df_res['strategy'].cummax() - 1) * 100

                dd_x2 = (df_res['bench_x2'] / df_res['bench_x2'].cummax() - 1) * 100

                

                fig_dd = go.Figure()

                fig_dd.add_trace(go.Scatter(

                    x=dd_strat.index, y=dd_strat,

                    fill='tozeroy', name='Strategy',

                    line=dict(color='#A855F7', width=1),

                    fillcolor='rgba(168, 85, 247, 0.15)'

                ))

                fig_dd.add_trace(go.Scatter(

                    x=dd_x2.index, y=dd_x2,

                    name='X2', line=dict(color='#ef4444', width=1, dash='dot')

                ))

                fig_dd.update_layout(

                    paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',

                    font=dict(family="Inter", color='#E0E0E0'),

                    height=300, yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

                )

                st.plotly_chart(fig_dd, use_container_width=True)

            

            # --- TAB 3: SIGNALS ---

            with tabs[2]:

                if not arb_sig.empty and 'Z_Score' in arb_sig.columns:

                    st.markdown("### 🎯 Arbitrage Z-Score")

                    curr_z = arb_sig['Z_Score'].iloc[-1]

                    st.metric("Current Z", f"{curr_z:.2f}")

                    

                    fig_z = go.Figure()

                    fig_z.add_trace(go.Scatter(

                        x=arb_sig.index, y=arb_sig['Z_Score'],

                        line=dict(color='#3b82f6', width=2)

                    ))

                    fig_z.add_hline(y=2, line_dash="dot", line_color="#ef4444")

                    fig_z.add_hline(y=-2, line_dash="dot", line_color="#10b981")

                    fig_z.add_hline(y=0, line_dash="dot", line_color="#888")

                    fig_z.update_layout(

                        paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',

                        font=dict(family="Inter", color='#E0E0E0'), height=300

                    )

                    st.plotly_chart(fig_z, use_container_width=True)

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

                            

                            fig_hist = px.histogram(

                                mc_df, x="CAGR", nbins=20,

                                color_discrete_sequence=['#A855F7']

                            )

                            fig_hist.update_layout(

                                paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',

                                font=dict(color='#E0E0E0')

                            )

                            st.plotly_chart(fig_hist, use_container_width=True)

                        else:

                            st.error("Validation failed")

            

            # --- TAB 5: FORECAST ---

            with tabs[4]:

                st.markdown("### 🔮 Monte Carlo Forecast")

                

                n_sims = st.slider("Simulations", 100, 1000, 500, 100)

                horizon = st.slider("Horizon (days)", 60, 504, 252, 21)

                

                if st.button("▶️ RUN FORECAST"):

                    with st.spinner(f"Running {n_sims} simulations..."):

                        forecast = MonteCarloForecaster.run_forecast(

                            data, sim_params, n_simulations=n_sims, forecast_days=horizon

                        )

                        

                        if not forecast.empty:

                            fig_fan = go.Figure()

                            

                            fig_fan.add_trace(go.Scatter(

                                x=forecast.index, y=forecast['p95'],

                                mode='lines', line=dict(width=0), showlegend=False

                            ))

                            fig_fan.add_trace(go.Scatter(

                                x=forecast.index, y=forecast['p5'],

                                mode='lines', line=dict(width=0),

                                fill='tonexty', fillcolor='rgba(168, 85, 247, 0.1)',

                                name='90% CI'

                            ))

                            fig_fan.add_trace(go.Scatter(

                                x=forecast.index, y=forecast['median'],

                                mode='lines', line=dict(color='#A855F7', width=3),

                                name='Median'

                            ))

                            

                            fig_fan.update_layout(

                                paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',

                                font=dict(family="Inter", color='#E0E0E0'),

                                height=500, hovermode="x unified"

                            )

                            

                            st.plotly_chart(fig_fan, use_container_width=True)

                            

                            final_med = forecast['median'].iloc[-1]

                            final_p5 = forecast['p5'].iloc[-1]

                            final_p95 = forecast['p95'].iloc[-1]

                            current = df_res['strategy'].iloc[-1]

                            

                            c1, c2, c3 = st.columns(3)

                            c1.metric("Expected", f"{final_med:.1f}", delta=f"{((final_med/current)-1)*100:.1f}%")

                            c2.metric("Pessimistic (5%)", f"{final_p5:.1f}", delta=f"{((final_p5/current)-1)*100:.1f}%")

                            c3.metric("Optimistic (95%)", f"{final_p95:.1f}", delta=f"{((final_p95/current)-1)*100:.1f}%")

                        else:

                            st.error("Forecast failed")
