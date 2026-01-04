import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import gc

# ==========================================
# 0. SYSTEM HARDENING (ANTI CRASH)
# ==========================================
import resource
try:
    # Tente d'augmenter la limite de fichiers ouverts du système (Linux/Mac)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
except Exception:
    pass

# Import du Cache pour éviter les appels réseaux infinis
try:
    import requests_cache
    # Crée une session qui stocke les réponses dans un fichier 'yfinance_cache.sqlite'
    session = requests_cache.CachedSession('yfinance_cache', expire_after=3600)
except ImportError:
    # Fallback si l'utilisateur n'a pas installé la lib, mais moins robuste
    import requests
    session = requests.Session()

# Headers pour éviter le blocage Yahoo
session.headers['User-agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

st.set_page_config(page_title="Predict. Institutional", layout="wide", page_icon="⚡")

# ==========================================
# 1. MODULES (Mocks pour éviter les erreurs d'import)
# ==========================================
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
# 2. CSS & DESIGN
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { background-color: #0A0A0F; font-family: 'Inter', sans-serif; color: #E0E0E0; }
    h1, h2, h3, h4, p, div, span, label { color: #E0E0E0; }
    
    .header-container {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A3E 100%);
        border-radius: 12px; padding: 25px; 
        border: 1px solid rgba(255,255,255,0.08); 
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .title-text { font-weight: 800; font-size: 32px; letter-spacing: -1px; color: #FFFFFF; }
    .title-dot { color: #A855F7; font-size: 32px; font-weight: 800; }
    
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; gap: 25px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border: none; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #A855F7 !important; border-bottom: 2px solid #A855F7 !important; font-weight: 600; }
    
    .glass-card { 
        background: rgba(30, 30, 46, 0.6); 
        border-radius: 12px; padding: 20px; 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        margin-bottom: 20px; backdrop-filter: blur(10px);
    }
    
    .stButton > button { width: 100%; border-radius: 6px; font-weight: 600; background-color: #1E1E2E; color: #A855F7; border: 1px solid #A855F7; transition: all 0.3s; }
    .stButton > button:hover { background-color: #A855F7; color: white; border: 1px solid #A855F7; }
    
    header, footer { visibility: hidden; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MOTEUR VECTORISÉ (RAPIDE)
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        dates = data.index
        # Numpy arrays pour performance
        px_x2 = data['X2'].values
        px_x1 = data['X1'].values
        
        w = int(params['rollingWindow'])
        thresh = -params['thresh'] / 100.0
        panic = -params['panic'] / 100.0
        recov_factor = params['recovery'] / 100.0
        confirm = int(params['confirm'])
        alloc_crash = params['allocCrash'] / 100.0
        alloc_prud = params['allocPrudence'] / 100.0
        cost_rate = params.get('cost', 0.001)

        # Indicateurs
        roll_max = data['X2'].rolling(w, min_periods=1).max().values
        dd = np.zeros_like(px_x2)
        mask = roll_max != 0
        dd[mask] = (px_x2[mask] / roll_max[mask]) - 1.0
        
        # Logique Régime
        n = len(data)
        regimes = np.zeros(n, dtype=int) 
        
        curr_reg = 0
        peak = px_x2[0]
        trough = px_x2[0]
        pending = 0
        conf_count = 0
        
        for i in range(1, n):
            price = px_x2[i]
            cur_dd = dd[i]
            target = curr_reg
            
            if curr_reg != 2: 
                if cur_dd <= panic: target = 2
                elif cur_dd <= thresh: target = 1
                else: target = 0
            
            if curr_reg in [1, 2]:
                if price < trough: trough = price
                recov_price = trough + (peak - trough) * recov_factor
                if price >= recov_price: target = 0
                else:
                    if cur_dd <= panic: target = 2
                    elif cur_dd <= thresh and curr_reg != 2: target = 1
            else:
                peak = roll_max[i]
                trough = price

            if target == pending: conf_count += 1
            else: pending = target; conf_count = 0
            
            if conf_count >= confirm and pending != curr_reg:
                curr_reg = pending
                conf_count = 0
                if curr_reg != 0: peak = roll_max[i]; trough = price
            
            regimes[i] = curr_reg

        # Allocation
        alloc_x1 = np.where(regimes == 2, alloc_crash, np.where(regimes == 1, alloc_prud, 0.0))
        alloc_x1 = np.roll(alloc_x1, 1); alloc_x1[0] = 0.0
        alloc_x2 = 1.0 - alloc_x1
        
        ret_x2 = np.zeros_like(px_x2); ret_x1 = np.zeros_like(px_x1)
        valid_x2 = (px_x2[:-1] != 0)
        ret_x2[1:][valid_x2] = (px_x2[1:][valid_x2] / px_x2[:-1][valid_x2]) - 1
        valid_x1 = (px_x1[:-1] != 0)
        ret_x1[1:][valid_x1] = (px_x1[1:][valid_x1] / px_x1[:-1][valid_x1]) - 1
        
        delta_alloc = np.abs(np.diff(alloc_x1, prepend=0))
        costs = delta_alloc * cost_rate
        
        strat_ret = (alloc_x2 * ret_x2) + (alloc_x1 * ret_x1) - costs
        curve_strat = 100 * np.cumprod(1 + strat_ret)
        curve_x2 = 100 * np.cumprod(1 + ret_x2)
        curve_x1 = 100 * np.cumprod(1 + ret_x1)
        
        df_res = pd.DataFrame({
            'portfolio': curve_strat,
            'benchX2': curve_x2,
            'benchX1': curve_x1,
            'regime': regimes,
            'weight_x2': alloc_x2,
            'weight_x1': alloc_x1,
            'drawdown': dd
        }, index=dates)
        
        trades = []
        regime_changes = np.where(np.diff(regimes) != 0)[0] + 1
        labels = {0: "OFFENSIF", 1: "PRUDENCE", 2: "CRASH"}
        
        for idx in regime_changes:
            if idx < len(dates):
                trades.append({'date': dates[idx], 'to': regimes[idx], 'label': labels.get(regimes[idx], "UNK")})
                
        return df_res, trades

# ==========================================
# 4. ANALYTICS & OPTIMIZER
# ==========================================
class AnalyticsEngine:
    @staticmethod
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

    @staticmethod
    def monte_carlo_forecast(series, n_sims=200, horizon=252):
        returns = series.pct_change().dropna()
        recent_returns = returns.tail(500).values 
        last_price = series.iloc[-1]
        paths = np.zeros((horizon, n_sims))
        
        for i in range(n_sims):
            sim_rets = np.random.choice(recent_returns, size=horizon, replace=True)
            paths[:, i] = last_price * np.cumprod(1 + sim_rets)
        return paths

class SmartOptimizer:
    @staticmethod
    def run(data, objective_mode, current_params):
        n_iter = 30 # Réduit pour éviter la surcharge lors des tests
        best_score, best_p = -np.inf, current_params.copy()
        
        for _ in range(n_iter):
            t = np.random.uniform(2.0, 12.0)
            p = np.random.uniform(t + 5.0, 40.0)
            r = np.random.choice([20, 30, 40, 50, 60])
            test_p = current_params.copy()
            test_p.update({'thresh': t, 'panic': p, 'recovery': r})
            
            res, _ = BacktestEngine.run_simulation(data, test_p)
            met = AnalyticsEngine.calculate_metrics(res['portfolio'])
            
            score = 0
            if objective_mode == "Max Performance (CAGR)":
                score = met['CAGR'] if met['MaxDD'] > -50 else -1000
            elif objective_mode == "Min Drawdown (Calmar)":
                score = met['Calmar']
            elif objective_mode == "Max Sharpe (Risk-Adj)":
                score = met['Sharpe']
            
            if score > best_score:
                best_score, best_p = score, test_p
                
        return best_p, best_score

# ==========================================
# 5. DATA FETCHING (CACHE ACTIVÉ)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """
    Récupère les données avec requests_cache pour éviter l'erreur 'Too many open files'.
    """
    if not tickers: return pd.DataFrame()
    price_map = {}
    clean = [t.strip().upper() for t in tickers]
    
    # 1. Tentative Bulk
    try:
        tickers_str = " ".join(clean)
        # On passe la session CACHÉE ici
        df = yf.download(tickers_str, start=start, end=end, progress=False, auto_adjust=True, session=session)
        if df.empty:
             df = yf.download(tickers_str, start=start, end=end, progress=False, auto_adjust=False, session=session)
    except:
        df = pd.DataFrame()

    # 2. Parsing
    if len(clean) == 1:
        t = clean[0]
        if not df.empty:
            if 'Close' in df.columns: price_map[t] = df['Close']
            elif 'Adj Close' in df.columns: price_map[t] = df['Adj Close']
            else: price_map[t] = df.iloc[:, 0]
    else:
        for t in clean:
            try:
                # Gestion MultiIndex complexe de yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    # Essai 1: df['Close']['AAPL']
                    if 'Close' in df.columns.get_level_values(0) and t in df['Close'].columns:
                        price_map[t] = df['Close'][t]
                    # Essai 2: df['AAPL']['Close']
                    elif t in df.columns.get_level_values(0):
                        price_map[t] = df[t]['Close']
                else:
                    if t in df.columns: price_map[t] = df[t]
            except: continue

    # Assemblage
    if len(price_map) >= 2:
        df_final = pd.concat(price_map.values(), axis=1)
        df_final.columns = clean[:len(df_final.columns)] # Force rename based on input order
        
        final_cols = {}
        if len(clean) >= 2:
            final_cols[clean[0]] = 'X2'
            final_cols[clean[1]] = 'X1'
            df_final.rename(columns=final_cols, inplace=True)
            
            if 'X2' in df_final.columns and 'X1' in df_final.columns:
                # Nettoyage mémoire
                gc.collect()
                return df_final.ffill().dropna()
                
    return pd.DataFrame()

# ==========================================
# 6. LAYOUT UI
# ==========================================

st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span class="title-text">Predict</span><span class="title-dot">.</span>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.0 • PREDICTIVE ANALYTICS • INSTITUTIONAL</p>
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
    presets = {"Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"], "S&P 500 (US)": ["SSO", "SPY"], "Custom": []}
    preset = st.selectbox("Universe", list(presets.keys()))
    if preset == "Custom":
        t_in = st.text_input("Tickers", "LQQ.PA, PUST.PA")
        tickers = [x.strip() for x in t_in.split(',')]
    else:
        tickers = presets[preset]
        st.caption(f"Risk: **{tickers[0]}** | Safe: **{tickers[1]}**")
    
    per = st.selectbox("Period", ["YTD", "1Y", "3YR", "5YR", "2022", "2008", "Custom"], index=2)
    today = datetime.now()
    if per == "YTD": start_d = datetime(today.year, 1, 1)
    elif per == "1Y": start_d = today - timedelta(days=365)
    elif per == "3YR": start_d = today - timedelta(days=365*3)
    elif per == "5YR": start_d = today - timedelta(days=365*5)
    elif per == "2022": start_d = datetime(2022,1,1); end_d = datetime(2022,12,31)
    elif per == "2008": start_d = datetime(2008,1,1); end_d = datetime(2008,12,31)
    else: start_d = datetime(2022,1,1)
    
    if per == "Custom": 
        start_d = st.date_input("Start", datetime(2022,1,1))
        end_d = st.date_input("End", today)
    elif per not in ["2022", "2008"]: 
        end_d = today
        
    st.markdown("---")
    st.markdown("### ⚡ PARAMS")
    if 'p' not in st.session_state: st.session_state['p'] = {'thresh': 5.0, 'panic': 15, 'recovery': 30, 'allocPrudence': 50, 'allocCrash': 100, 'rollingWindow': 60, 'confirm': 2}
    pp = st.session_state['p']
    
    thresh = st.slider("Threshold (%)", 2.0, 10.0, float(pp['thresh']), 0.5)
    panic = st.slider("Panic (%)", 10, 40, int(pp['panic']), 1)
    recov = st.slider("Recovery (%)", 10, 60, int(pp['recovery']), 5)
    
    with st.expander("Avancé"):
        a_prud = st.slider("Alloc Prudence (%)", 0, 100, int(pp['allocPrudence']))
        a_crash = st.slider("Alloc Crash (%)", 0, 100, int(pp['allocCrash']))
        confirm = st.slider("Confirm (Jours)", 1, 5, int(pp['confirm']))
        cost = st.number_input("Frais (%)", 0.0, 1.0, 0.1) / 100

    st.markdown("---")
    st.markdown("### 🧠 AUTO TUNE")
    obj_mode = st.selectbox("Cible", ["Max Performance (CAGR)", "Min Drawdown (Calmar)", "Max Sharpe (Risk-Adj)"])
    
    if st.button("LANCER L'OPTIMISATION"):
        d_opt = get_data(tickers, start_d, end_d)
        if not d_opt.empty:
            with st.spinner(f"Recherche optimale pour : {obj_mode}..."):
                curr = {'thresh': thresh, 'panic': panic, 'recovery': recov, 'allocPrudence': a_prud, 'allocCrash': a_crash, 'rollingWindow': 60, 'confirm': confirm, 'cost': cost}
                best, score = SmartOptimizer.run(d_opt, obj_mode, curr)
                st.session_state['p'] = best
                st.success(f"Optimisé ! Score: {score:.2f}")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN ---
with col_main:
    data = get_data(tickers, start_d, end_d)
    
    if data.empty or len(data) < 10:
        st.error("❌ NO DATA. Please verify tickers or date range.")
    else:
        sim_p = {'thresh': thresh, 'panic': panic, 'recovery': recov, 'allocPrudence': a_prud, 'allocCrash': a_crash, 'rollingWindow': 60, 'confirm': confirm, 'cost': cost}
        df_res, trades = BacktestEngine.run_simulation(data, sim_p)
        m_strat = AnalyticsEngine.calculate_metrics(df_res['portfolio'])
        m_bench = AnalyticsEngine.calculate_metrics(df_res['benchX2'])
        
        # Modules Externes
        risk_s = RiskMetrics.get_full_risk_profile(df_res['portfolio']) if MODULES_STATUS["Risk"] else {}
        lev_beta = LeverageDiagnostics.calculate_realized_beta(data) if MODULES_STATUS["Leverage"] else pd.DataFrame()
        arb_sig = ArbitrageSignals.calculate_relative_strength(data) if MODULES_STATUS["Arbitrage"] else pd.DataFrame()

        t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "📈 Performance", "⚙️ Risk & Leverage", "🎯 Signals", "🔮 Forecast"])
        
        # --- TAB 1: DASHBOARD ---
        with t1:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR", f"{m_strat['CAGR']:.1f}%", delta=f"{m_strat['CAGR']-m_bench['CAGR']:.1f}%")
            k2.metric("Max Drawdown", f"{m_strat['MaxDD']:.1f}%", delta=f"{m_strat['MaxDD']-m_bench['MaxDD']:.1f}%", delta_color="inverse")
            k3.metric("Sharpe", f"{m_strat['Sharpe']:.2f}")
            k4.metric("Trades", len(trades))
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # SUBPLOTS
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # 1. Perf
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['portfolio'], name='STRATÉGIE', 
                                     line=dict(color='#A855F7', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['benchX2'], name='Risk (X2)', 
                                     line=dict(color='#ef4444', width=1, dash='dot')), row=1, col=1)
            
            for t in trades:
                c = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
                fig.add_annotation(x=t['date'], y=df_res.loc[t['date']]['portfolio'], text="▼" if t['to']!=0 else "▲", 
                                   showarrow=False, font=dict(color=c, size=14), row=1, col=1)

            # 2. Alloc
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['weight_x2']*100, name='Alloc X2',
                                     stackgroup='one', line=dict(width=0), fillcolor='rgba(239, 68, 68, 0.3)'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['weight_x1']*100, name='Alloc X1',
                                     stackgroup='one', line=dict(width=0), fillcolor='rgba(16, 185, 129, 0.3)'), row=2, col=1)

            fig.update_layout(
                paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', 
                font=dict(family="Inter", color='#E0E0E0'), height=550, 
                margin=dict(l=40, r=40, t=20, b=40), 
                xaxis2=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis2=dict(range=[0, 100], title="Alloc %"),
                hovermode="x unified", showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div style="display:flex; justify-content:center; gap:20px; font-size:12px; color:#888;">
                <span style="color:#10b981">▲ Buy Risk</span>
                <span style="color:#ef4444">▼ Panic Sell (X1)</span>
                <span style="color:#f59e0b">▼ Prudence</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: PERFORMANCE ---
        with t2:
            st.markdown("### 🏆 Tableau de Bord")
            p_data = {
                "Metric": ["CAGR", "Vol (Ann)", "Sharpe", "MaxDD", "Calmar", "Cumul"],
                "Strategy": [f"{m_strat['CAGR']:.1f}%", f"{m_strat['Vol']:.1f}%", f"{m_strat['Sharpe']:.2f}", f"{m_strat['MaxDD']:.1f}%", f"{m_strat['Calmar']:.2f}", f"{m_strat['Cumul']:.1f}%"],
                "Benchmark": [f"{m_bench['CAGR']:.1f}%", f"{m_bench['Vol']:.1f}%", f"{m_bench['Sharpe']:.2f}", f"{m_bench['MaxDD']:.1f}%", f"{m_bench['Calmar']:.2f}", f"{m_bench['Cumul']:.1f}%"]
            }
            st.markdown(pd.DataFrame(p_data).style.hide(axis="index").set_properties(**{'background-color': '#0A0A0F', 'color': '#eee', 'border-color': '#333'}).to_html(), unsafe_allow_html=True)

        # --- TAB 3: RISK ---
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
                if not lev_beta.empty:
                    st.metric("Realized Beta", f"{lev_beta['Realized_Beta'].iloc[-1]:.2f}x")
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=lev_beta.index, y=lev_beta['Realized_Beta'], line=dict(color='#A855F7')))
                    fig_l.add_hline(y=2.0, line_dash="dot", line_color="white")
                    fig_l.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'), height=200, margin=dict(t=10,b=10))
                    st.plotly_chart(fig_l, use_container_width=True)
                else: st.info("Leverage Module missing")

        # --- TAB 4: SIGNALS ---
        with t4:
            if not arb_sig.empty:
                st.markdown("### 🎯 Arbitrage Z-Score")
                curr_z = arb_sig['Z_Score'].iloc[-1]
                st.metric("Z-Score", f"{curr_z:.2f}", delta="Rich" if curr_z>0 else "Cheap", delta_color="inverse")
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=arb_sig.index, y=arb_sig['Z_Score'], line=dict(color='#3b82f6', width=2)))
                fig_z.add_hrect(y0=2.0, y1=5.0, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0)
                fig_z.add_hrect(y0=-5.0, y1=-2.0, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0)
                fig_z.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(color='#E0E0E0'), height=300, margin=dict(t=10,b=10), yaxis=dict(title="Sigma", showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-3.5, 3.5]))
                st.plotly_chart(fig_z, use_container_width=True)
                
                st.markdown("""
                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; font-size: 12px; margin-top: 10px; color: #aaa;">
                    <strong>📊 Comprendre le Z-Score :</strong><br>
                    • <strong>> 2.0 (Zone Rouge)</strong> : Le marché est "Surchauffé". Risque de correction.<br>
                    • <strong>< -2.0 (Zone Verte)</strong> : Le marché est "Bradé". Opportunité d'achat potentiel.<br>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else: st.info("Arbitrage Module missing")

        # --- TAB 5: FORECAST ---
        with t5:
            st.markdown("### 🔮 Prévisions de Marché (Fan Chart)")
            st.caption("Projection probabiliste du portefeuille sur les 252 prochains jours (1 an).")
            
            if st.button("Générer les Scénarios (200 Simulations)"):
                with st.spinner("Calcul des trajectoires futures..."):
                    paths = AnalyticsEngine.monte_carlo_forecast(df_res['portfolio'], n_sims=200, horizon=252)
                    
                    final_prices = paths[-1, :]
                    median_price = np.median(final_prices)
                    p95 = np.percentile(final_prices, 95)
                    p05 = np.percentile(final_prices, 5)
                    start_price = df_res['portfolio'].iloc[-1]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prix Médian (1A)", f"{median_price:.0f}", delta=f"{(median_price/start_price-1)*100:.1f}%")
                    c2.metric("Optimiste (95%)", f"{p95:.0f}", delta=f"{(p95/start_price-1)*100:.1f}%")
                    c3.metric("Pessimiste (5%)", f"{p05:.0f}", delta=f"{(p05/start_price-1)*100:.1f}%", delta_color="inverse")
                    
                    fig_mc = go.Figure()
                    x_axis = np.arange(252)
                    y_p95 = np.percentile(paths, 95, axis=1)
                    y_p05 = np.percentile(paths, 5, axis=1)
                    y_median = np.median(paths, axis=1)
                    
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=y_p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=y_p05, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(168, 85, 247, 0.2)', name='Intervalle 95%'))
                    fig_mc.add_trace(go.Scatter(x=x_axis, y=y_median, mode='lines', name='Trajectoire Médiane', line=dict(color='#A855F7', width=3)))
                    fig_mc.add_hline(y=start_price, line_dash="dot", line_color="white", annotation_text="Prix Actuel")

                    fig_mc.update_layout(paper_bgcolor='#0A0A0F', plot_bgcolor='#0A0A0F', font=dict(family="Inter", color='#E0E0E0'), height=450, title="Projection Future (Cône d'Incertitude)", xaxis_title="Jours Ouvrés", yaxis_title="Valeur")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Nettoyage
                    gc.collect()
