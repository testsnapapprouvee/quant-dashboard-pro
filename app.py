import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# ==========================================
# 0. CONFIGURATION & STYLES
# ==========================================
st.set_page_config(page_title="Predict. Pro", layout="wide", page_icon="⚡")

# --- MOCK CLASSES (Pour fonctionnement autonome sans fichiers externes) ---
class RiskMetrics:
    @staticmethod
    def get_full_risk_profile(series): return {"Ulcer_Index": 0, "VaR_95": 0, "CVaR_95": 0}

class LeverageDiagnostics:
    @staticmethod
    def calculate_realized_beta(data, window=21): 
        return pd.DataFrame({'Realized_Beta': [1.0]*len(data)}, index=data.index)

class ArbitrageSignals:
    @staticmethod
    def calculate_relative_strength(data, window=20): 
        return pd.DataFrame({'Z_Score': np.random.randn(len(data))}, index=data.index)

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
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1E1E2E; color: #A855F7; border: 1px solid #A855F7; }
    .stButton > button:hover { background-color: #A855F7; color: white; }
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """Télécharge et nettoie les données Yahoo Finance"""
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    
    series_list = []
    ticker_names = []
    
    for ticker in tickers[:2]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Gestion robuste MultiIndex vs SingleIndex
                if isinstance(df.columns, pd.MultiIndex):
                    # Essayer de récupérer Close ou Adj Close
                    try:
                        series = df.xs('Close', axis=1, level=0)[ticker]
                    except KeyError:
                        try:
                            series = df.xs('Adj Close', axis=1, level=0)[ticker]
                        except KeyError:
                            series = df.iloc[:, 0] # Fallback brutal
                else:
                    if 'Close' in df.columns:
                        series = df['Close']
                    elif 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    else:
                        series = df.iloc[:, 0]
                
                # Nettoyage basique
                series = series.replace(0, np.nan).ffill().dropna()
                series.name = ticker
                series_list.append(series)
                ticker_names.append(ticker)
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
            continue
    
    if len(series_list) == 2:
        result = pd.concat(series_list, axis=1)
        result.columns = ['X2', 'X1'] # Convention: Col 1 = Risk, Col 2 = Safe
        result = result.ffill().dropna()
        return result
    
    return pd.DataFrame()

# ==========================================
# 2. BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        # Préparation des données numpy pour vitesse
        prices_x2 = data['X2'].values
        prices_x1 = data['X1'].values
        dates = data.index
        n = len(data)
        
        # Benchmarks (Base 100)
        bench_x2 = (data['X2'] / data['X2'].iloc[0]) * 100
        bench_x1 = (data['X1'] / data['X1'].iloc[0]) * 100
        
        # État du Portefeuille
        position_x2 = 100.0  # Départ 100% Risk
        position_x1 = 0.0
        
        # Machine à états (Regimes)
        current_regime = 'R0'
        pending_regime = 'R0'
        confirm_count = 0
        
        # Suivi Drawdown
        price_history = []
        peak_at_crash = 0.0
        trough = 0.0
        
        results = []
        trades = []
        
        # Extraction Paramètres
        rolling_window = int(params['rollingWindow'])
        threshold = params['thresh']
        panic = params['panic']
        recovery = params['recovery']
        confirm_days = int(params['confirm'])
        alloc_prudence = params['allocPrudence'] / 100.0
        alloc_crash = params['allocCrash'] / 100.0
        tx_cost = params.get('cost', 0.001)
        
        for i in range(n):
            # 1. Mise à jour valeur portefeuille (Market Move)
            if i > 0:
                ret_x2 = (prices_x2[i] / prices_x2[i-1]) - 1
                ret_x1 = (prices_x1[i] / prices_x1[i-1]) - 1
                position_x2 *= (1 + ret_x2)
                position_x1 *= (1 + ret_x1)
            
            # 2. Calcul Drawdown (High Water Mark sur X2)
            price_history.append(prices_x2[i])
            if len(price_history) > rolling_window:
                price_history.pop(0)
            
            peak = max(price_history)
            current_dd = ((prices_x2[i] - peak) / peak) * 100 if peak > 0 else 0
            
            # 3. Détection Régime Cible
            target_regime = current_regime 
            
            # Logique de CHUTE (Offensif -> Defensif)
            if current_regime == 'R0':
                if current_dd <= -panic:
                    target_regime = 'R2'
                elif current_dd <= -threshold:
                    target_regime = 'R1'
            
            # Logique de RECOVERY (Defensif -> Offensif)
            elif current_regime in ['R1', 'R2']:
                if prices_x2[i] < trough:
                    trough = prices_x2[i]
                
                recovery_price = trough + (peak_at_crash - trough) * (recovery / 100.0)
                
                if prices_x2[i] >= recovery_price:
                    target_regime = 'R0'
                else:
                    # Aggravation possible même en recovery
                    if current_dd <= -panic and current_regime == 'R1':
                        target_regime = 'R2'

            # Init références crash si on est Offensif
            if current_regime == 'R0':
                peak_at_crash = peak
                trough = prices_x2[i]

            # 4. Confirmation
            if target_regime != current_regime:
                if target_regime == pending_regime:
                    confirm_count += 1
                else:
                    pending_regime = target_regime
                    confirm_count = 1
            else:
                confirm_count = 0
                pending_regime = current_regime
            
            # 5. Exécution Trade
            # Note: confirm_days >= 1 signifie qu'on attend au moins la clôture du jour de signal
            if confirm_count >= confirm_days and pending_regime != current_regime:
                old_regime = current_regime
                current_regime = pending_regime
                
                # Définir Allocation Cible
                if current_regime == 'R2':
                    target_alloc_x1 = alloc_crash
                    label = "CRASH (R2)"
                elif current_regime == 'R1':
                    target_alloc_x1 = alloc_prudence
                    label = "PRUDENCE (R1)"
                else:
                    target_alloc_x1 = 0.0
                    label = "OFFENSIF (R0)"
                
                # Rebalancing
                total_val = position_x2 + position_x1
                cost = total_val * tx_cost
                total_val -= cost # Appliquer frais
                
                position_x1 = total_val * target_alloc_x1
                position_x2 = total_val * (1 - target_alloc_x1)
                
                # Reset refs si entrée en défensif
                if current_regime != 'R0':
                    peak_at_crash = peak
                    trough = prices_x2[i]
                
                trades.append({
                    'date': dates[i],
                    'from': old_regime,
                    'to': current_regime,
                    'label': label,
                    'dd': current_dd,
                    'val': total_val
                })
                confirm_count = 0

            # 6. Sauvegarde Résultats
            total = position_x2 + position_x1
            alloc_x2_pct = (position_x2 / total * 100) if total > 0 else 0
            alloc_x1_pct = (position_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i],
                'strategy': total,
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
        vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = cagr / vol if vol > 0 else 0
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
        thresholds = [2, 3, 5, 8]
        panics = [8, 10, 15, 20]
        recoveries = [20, 30, 40]
        
        best_score = -np.inf
        best_params = {}
        
        for t in thresholds:
            for p in panics:
                if p <= t: continue
                for r in recoveries:
                    test_params = fixed_params.copy()
                    test_params.update({'thresh': t, 'panic': p, 'recovery': r})
                    try:
                        res, _ = BacktestEngine.run_simulation(data, test_params)
                        if res.empty: continue
                        metrics = calculate_metrics(res['strategy'])
                        
                        if profile == "DEFENSIVE": score = metrics['Calmar']
                        elif profile == "BALANCED": score = metrics['Sharpe']
                        else: score = metrics['CAGR'] if metrics['MaxDD'] > -35.0 else -1000
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'thresh': t, 'panic': p, 'recovery': r}
                    except: continue
        return best_params, best_score

# ==========================================
# 5. MONTE CARLO FORECASTER
# ==========================================
class MonteCarloForecaster:
    @staticmethod
    def run_forecast(data, params, n_simulations=200, forecast_days=252):
        try:
            # 1. Stats historiques
            returns_x2 = data['X2'].pct_change().dropna()
            returns_x1 = data['X1'].pct_change().dropna()
            
            if returns_x2.empty or returns_x1.empty: return pd.DataFrame()
            
            mu_x2, sigma_x2 = returns_x2.mean(), returns_x2.std()
            mu_x1, sigma_x1 = returns_x1.mean(), returns_x1.std()
            
            # Récupérer valeur actuelle du portefeuille via un run rapide
            hist_res, _ = BacktestEngine.run_simulation(data, params)
            start_value = hist_res['strategy'].iloc[-1]
            last_price_x2 = data['X2'].iloc[-1]
            last_price_x1 = data['X1'].iloc[-1]
            
            simulations = []
            future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='D')[1:]

            # 2. Boucle simulations
            for _ in range(n_simulations):
                # Génération chemins aléatoires (Geometric Brownian Motion simple)
                sim_rets_x2 = np.random.normal(mu_x2, sigma_x2, forecast_days)
                sim_rets_x1 = np.random.normal(mu_x1, sigma_x1, forecast_days)
                
                # Reconstruction prix
                sim_px_x2 = last_price_x2 * (1 + sim_rets_x2).cumprod()
                sim_px_x1 = last_price_x1 * (1 + sim_rets_x1).cumprod()
                
                sim_data = pd.DataFrame({'X2': sim_px_x2, 'X1': sim_px_x1}, index=future_dates)
                
                # Application Stratégie
                sim_res, _ = BacktestEngine.run_simulation(sim_data, params)
                
                if not sim_res.empty:
                    # Rebaser sur la vraie valeur actuelle
                    curve = (sim_res['strategy'] / sim_res['strategy'].iloc[0]) * start_value
                    simulations.append(curve)
            
            if not simulations: return pd.DataFrame()
            
            # 3. Agrégation
            forecast_df = pd.DataFrame(simulations).T
            return pd.DataFrame({
                'median': forecast_df.median(axis=1),
                'p5': forecast_df.quantile(0.05, axis=1),
                'p95': forecast_df.quantile(0.95, axis=1)
            })
        except Exception as e:
            st.error(f"MC Error: {e}")
            return pd.DataFrame()


# ==========================================
# 6. UI
# ==========================================
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span class="title-text">Predict</span><span class="title-dot">.</span>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V3.2 • FULL SUITE</p>
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
    
    period_options = ["YTD", "1Y", "3YR", "2022", "Custom"]
    sel_period = st.selectbox("Period", period_options, index=3) # Default 2022 pour voir l'action
    
    today = datetime.now()
    if sel_period == "YTD": start_d, end_d = datetime(today.year, 1, 1), today
    elif sel_period == "1Y": start_d, end_d = today - timedelta(days=365), today
    elif sel_period == "3YR": start_d, end_d = today - timedelta(days=365*3), today
    elif sel_period == "2022": start_d, end_d = datetime(2022, 1, 1), datetime(2022, 12, 31)
    else:
        start_d = st.date_input("Start", datetime(2022, 1, 1))
        end_d = st.date_input("End", datetime(2022, 12, 31))
    
    st.markdown("---")
    st.markdown("### ⚡ PARAMÈTRES")
    
    # --- FIXED DEFAULTS FOR REACTIVITY ---
    if 'params' not in st.session_state:
        st.session_state['params'] = {'thresh': 3.0, 'panic': 10.0, 'recovery': 20}
    
    thresh = st.slider("Threshold (%)", 1.0, 10.0, float(st.session_state['params']['thresh']), 0.5)
    panic = st.slider("Panic (%)", 5.0, 30.0, float(st.session_state['params']['panic']), 1.0)
    recov = st.slider("Recovery (%)", 10, 60, int(st.session_state['params']['recovery']), 5)
    
    st.markdown("---")
    alloc_prud = st.slider("Prudence (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash (X1%)", 0, 100, 100, 10)
    confirm = st.slider("Confirm (Days)", 1, 5, 1, 1) # 1 jour par défaut
    
    st.markdown("---")
    profile = st.selectbox("Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    if st.button(f"🎯 OPTIMIZE ({profile})"):
        opt_data = get_data(tickers, start_d, end_d)
        if not opt_data.empty:
            with st.spinner("Grid Search running..."):
                base_params = {
                    'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
                    'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001
                }
                best_p, score = Optimizer.run_grid_search(opt_data, profile, base_params)
                if best_p:
                    st.session_state['params'] = best_p
                    st.success(f"✅ Score: {score:.2f}")
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty:
        st.info(f"⏳ Waiting for data... (Checking {tickers})")
    else:
        sim_params = {
            'thresh': thresh, 'panic': panic, 'recovery': recov,
            'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
            'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001
        }
        
        df_res, trades = BacktestEngine.run_simulation(data, sim_params)
        
        if df_res.empty:
            st.error("❌ Simulation failed")
        else:
            met_strat = calculate_metrics(df_res['strategy'])
            met_x2 = calculate_metrics(df_res['bench_x2'])
            
            tabs = st.tabs(["📈 Performance", "⚠️ Risk & Debug", "📋 Data", "🔮 Forecast"])
            
            # --- TAB 1: PERFORMANCE ---
            with tabs[0]:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("CAGR", f"{met_strat['CAGR']:.1f}%", delta=f"{met_strat['CAGR']-met_x2['CAGR']:.1f}% vs X2")
                k2.metric("MaxDD", f"{met_strat['MaxDD']:.1f}%", delta=f"{met_strat['MaxDD']-met_x2['MaxDD']:.1f}%", delta_color="inverse")
                k3.metric("Trades", len(trades))
                k4.metric("Regime Actuel", df_res['regime'].iloc[-1])
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05, 
                                    shared_xaxes=True, subplot_titles=("NAV", "Allocation"))
                
                # NAV Area
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['strategy'], name='Strategy',
                                         line=dict(color='#A855F7', width=3), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['bench_x2'], name='Risk Asset',
                                         line=dict(color='#ef4444', width=1, dash='dot')), row=1, col=1)
                
                # Trade Markers
                for t in trades:
                    c = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                    fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['strategy'],
                                       text="▼" if t['to'] != 'R0' else "▲", showarrow=False,
                                       font=dict(color=c, size=18), row=1, col=1)

                # Allocation Stacked
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['alloc_x2'], name='Risk %', stackgroup='one', 
                                         mode='none', fillcolor='rgba(239, 68, 68, 0.5)'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_res.index, y=df_res['alloc_x1'], name='Safe %', stackgroup='one', 
                                         mode='none', fillcolor='rgba(16, 185, 129, 0.5)'), row=2, col=1)
                
                fig.update_layout(height=600, paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',
                                  font=dict(family="Inter", color='#E0E0E0'), hovermode="x unified",
                                  margin=dict(l=40, r=40, t=40, b=40))
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- TAB 2: RISK ---
            with tabs[1]:
                st.markdown("### 🌊 Drawdown vs Seuils")
                st.caption("Visualisez quand le Drawdown (violet) franchit vos seuils.")
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=df_res.index, y=df_res['drawdown'], name='Drawdown',
                                            line=dict(color='#A855F7', width=2), fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.2)'))
                
                # VISUAL THRESHOLDS
                fig_dd.add_hline(y=-thresh, line_dash="dot", line_color="orange", annotation_text=f"Threshold -{thresh}%")
                fig_dd.add_hline(y=-panic, line_dash="dot", line_color="red", annotation_text=f"Panic -{panic}%")
                
                fig_dd.update_layout(height=400, paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',
                                     font=dict(color='#E0E0E0'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                st.plotly_chart(fig_dd, use_container_width=True)
                
                if len(trades) > 0:
                    st.markdown("### 📋 Historique des Trades")
                    st.dataframe(pd.DataFrame(trades), use_container_width=True)

            # --- TAB 3: DATA ---
            with tabs[2]:
                st.dataframe(df_res, use_container_width=True)

            # --- TAB 4: FORECAST ---
            with tabs[3]:
                st.markdown("### 🔮 Monte Carlo Forecast")
                st.caption(f"Projection sur {252} jours basée sur la volatilité récente.")
                
                col_btn, col_sl = st.columns([1, 2])
                with col_btn:
                    run_mc = st.button("▶️ LANCER FORECAST", type="primary")
                with col_sl:
                    n_sims = st.slider("Nombre simulations", 100, 1000, 200, 100)
                
                if run_mc:
                    with st.spinner("Simulation des futurs possibles..."):
                        fc_df = MonteCarloForecaster.run_forecast(data, sim_params, n_simulations=n_sims)
                        
                        if not fc_df.empty:
                            curr_val = df_res['strategy'].iloc[-1]
                            med = fc_df['median'].iloc[-1]
                            p5 = fc_df['p5'].iloc[-1]
                            p95 = fc_df['p95'].iloc[-1]
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Attendu (Median)", f"{med:.0f}", delta=f"{((med/curr_val)-1)*100:.1f}%")
                            c2.metric("Pessimiste (5%)", f"{p5:.0f}", delta=f"{((p5/curr_val)-1)*100:.1f}%", delta_color="inverse")
                            c3.metric("Optimiste (95%)", f"{p95:.0f}", delta=f"{((p95/curr_val)-1)*100:.1f}%")
                            
                            fig_mc = go.Figure()
                            # Intervalle de confiance
                            fig_mc.add_trace(go.Scatter(x=fc_df.index, y=fc_df['p95'], mode='lines', line=dict(width=0), showlegend=False))
                            fig_mc.add_trace(go.Scatter(x=fc_df.index, y=fc_df['p5'], mode='lines', line=dict(width=0), fill='tonexty', 
                                                        fillcolor='rgba(168, 85, 247, 0.1)', name='90% Conf.'))
                            # Median
                            fig_mc.add_trace(go.Scatter(x=fc_df.index, y=fc_df['median'], mode='lines', line=dict(color='#A855F7', width=2), name='Median'))
                            
                            fig_mc.update_layout(height=500, paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F',
                                                 font=dict(color='#E0E0E0'), title="Éventail des probabilités")
                            st.plotly_chart(fig_mc, use_container_width=True)
