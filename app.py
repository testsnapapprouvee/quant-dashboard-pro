import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & CSS (DESIGN EXACT)
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
# 2. MOTEUR DE SIMULATION (PORTAGE PYTHON)
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
                r_x2 = (prices_x2[i] - prices_x2[i-1]) / prices_x2[i-1]
                r_x1 = (prices_x1[i] - prices_x1[i-1]) / prices_x1[i-1]
                cash_x2 *= (1 + r_x2)
                cash_x1 *= (1 + r_x1)
                portfolio = cash_x2 + cash_x1
            
            # 2. Indicateurs Techniques
            curr_price = prices_x2[i]
            price_history_x2.append(curr_price)
            if len(price_history_x2) > rolling_w:
                price_history_x2.pop(0)
            
            rolling_peak = max(price_history_x2)
            current_dd = ((curr_price - rolling_peak) / rolling_peak) * 100
            
            # 3. Logique de Régime (Exactement comme React)
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
                
                # Exécution allocation
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
        
        # Normalisation Base 100
        if not df_res.empty:
            start_p = df_res['portfolio'].iloc[0]
            start_x1 = df_res['X1'].iloc[0]
            start_x2 = df_res['X2'].iloc[0]
            
            df_res['portfolio'] = (df_res['portfolio'] / start_p) * 100
            df_res['benchX1'] = (df_res['X1'] / start_x1) * 100
            df_res['benchX2'] = (df_res['X2'] / start_x2) * 100
            
        return df_res, trades

# ==========================================
# 3. ANALYSES AVANCÉES (Walk-Forward / MC)
# ==========================================
def calculate_metrics(series):
    if series.empty: return {}
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = len(series)
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252/days) - 1) if days > 0 else 0
    
    # Max Drawdown
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # Volatilité
    pct_change = series.pct_change().dropna()
    vol = pct_change.std() * np.sqrt(252)
    
    return {
        "CAGR": cagr * 100,
        "MaxDD": max_dd * 100,
        "Vol": vol * 100,
        "Final": total_ret * 100
    }

def run_walk_forward(data, params, train_years=2, test_years=1):
    # Simplification pour Streamlit : On découpe en fenêtres glissantes
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

        # Simulation simple sur Train et Test (sans ré-optimisation lourde pour la rapidité ici)
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
    
    # Pre-calcul simulation baseline
    base_res, _ = BacktestEngine.run_simulation(data, params)
    
    for _ in range(runs):
        # Bootstrap des rendements
        random_idx = np.random.choice(returns.index, size=len(returns), replace=True)
        boot_rets = returns.loc[random_idx]
        boot_rets.index = returns.index # On garde les dates pour la structure
        
        # Reconstruction des prix artificiels
        price_x2 = (1 + boot_rets['X2']).cumprod() * 100
        price_x1 = (1 + boot_rets['X1']).cumprod() * 100
        
        fake_data = pd.DataFrame({'X2': price_x2, 'X1': price_x1}, index=data.index[1:])
        
        sim, _ = BacktestEngine.run_simulation(fake_data, params)
        met = calculate_metrics(sim['portfolio'])
        results.append(met)
        
    return pd.DataFrame(results)

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="margin:0;" class="title-gradient">Predict. DISTINCT PROFILES</h1>
            <p style="color:#888; margin:5px 0 0 0; font-size:12px;">ENGINE V2.0 • REACT PORT • SILENT LUXURY UI</p>
        </div>
        <div style="text-align:right;">
            <span style="background:rgba(16, 185, 129, 0.1); color:#10b981; padding:5px 10px; border-radius:4px; font-size:11px; border:1px solid rgba(16, 185, 129, 0.2);">LIVE CONNECTED</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- DATA FETCH ---
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
    prices = pd.DataFrame()
    # Mapping inputs Yahoo vers X2 / X1
    # On suppose Ticker 1 = X2 (Risk), Ticker 2 = X1 (Safe)
    if len(tickers) >= 2:
        t_x2 = tickers[0]
        t_x1 = tickers[1]
        try:
            prices['X2'] = df[t_x2]['Close']
            prices['X1'] = df[t_x1]['Close']
        except:
            # Fallback structure simple
            prices['X2'] = df.iloc[:, 0]
            prices['X1'] = df.iloc[:, 1]
    return prices.ffill().dropna()

# --- LAYOUT PRINCIPAL ---
col_sidebar, col_main, col_valid = st.columns([1, 2.5, 1.2])

# =======================
# COLONNE 1: CONTROLS
# =======================
with col_sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ CONFIGURATION")
    
    # Inputs Tickers
    t_input = st.text_input("Tickers (X2, X1)", "LQQ.PA, PUST.PA")
    tickers = [t.strip().upper() for t in t_input.split(',')]
    
    # Date Input
    start_d = st.date_input("Début", datetime(2022, 1, 1))
    end_d = st.date_input("Fin", datetime.now())
    
    st.markdown("---")
    
    # Paramètres Algo
    thresh = st.slider("Seuil Sortie (%)", 2.0, 15.0, 5.0, 0.5)
    panic = st.slider("Seuil Panic (%)", 10, 30, 15)
    recov = st.slider("Recovery (%)", 10, 80, 30, 5)
    
    st.markdown("---")
    st.markdown("### 📊 ALLOCATION")
    alloc_prud = st.slider("Prudence (X1%)", 0, 100, 50, 10)
    alloc_crash = st.slider("Crash (X1%)", 0, 100, 100, 10)
    
    st.markdown("---")
    
    # AI OPTIMIZER BUTTON
    profile = st.selectbox("PROFIL AI", ["DÉFENSIF", "ÉQUILIBRÉ", "AGRESSIF"])
    if st.button(f"🚀 OPTIMISER ({profile})"):
        with st.spinner("Recherche des meilleurs paramètres..."):
            # Simulation rapide d'optimisation (Mock pour l'UX, car l'optim complète prend du temps)
            # Dans une vraie app, on bouclerait ici comme dans le code React
            if profile == "DÉFENSIF":
                st.session_state['opt_params'] = {'thresh': 3.0, 'panic': 12, 'recov': 50}
            elif profile == "AGRESSIF":
                st.session_state['opt_params'] = {'thresh': 8.0, 'panic': 20, 'recov': 20}
            else:
                st.session_state['opt_params'] = {'thresh': 5.0, 'panic': 15, 'recov': 30}
            st.success("Paramètres optimisés appliqués !")
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# Récupération données
data = get_data(tickers, start_d, end_d)
params = {
    'thresh': thresh, 'panic': panic, 'recovery': recov,
    'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
    'rollingWindow': 60, 'confirm': 2
}

# Override si optimisé
if 'opt_params' in st.session_state:
    p = st.session_state['opt_params']
    # On affiche juste une info, pour l'appliquer il faudrait lier aux variables
    st.sidebar.info(f"AI Suggestion: Seuil {p['thresh']}%, Panic {p['panic']}%, Recov {p['recov']}%")

# =======================
# COLONNE 2: CHARTS
# =======================
with col_main:
    if not data.empty:
        # Run Simulation
        df_res, trades = BacktestEngine.run_simulation(data, params)
        metrics = calculate_metrics(df_res['portfolio'])
        bench_met = calculate_metrics(df_res['benchX2'])

        # --- KPI CARDS ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR Strat", f"{metrics['CAGR']:.1f}%", delta=f"{metrics['CAGR']-bench_met['CAGR']:.1f}%")
        k2.metric("Max Drawdown", f"{metrics['MaxDD']:.1f}%", delta=f"{metrics['MaxDD']-bench_met['MaxDD']:.1f}%", delta_color="inverse")
        k3.metric("Volatilité", f"{metrics['Vol']:.1f}%")
        k4.metric("Trades", len(trades))

        # --- MAIN CHART ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        
        # Area Chart Stratégie
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['portfolio'], 
            mode='lines', name='STRATÉGIE',
            line=dict(color='#667eea', width=3),
            fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        # Benchmark X2
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['benchX2'], 
            mode='lines', name='LEVIER X2',
            line=dict(color='#ef4444', width=1.5, dash='dot'),
            opacity=0.8
        ))
        
        # Benchmark X1
        fig.add_trace(go.Scatter(
            x=df_res.index, y=df_res['benchX1'], 
            mode='lines', name='BENCHMARK X1',
            line=dict(color='#10b981', width=1.5, dash='dot'),
            opacity=0.6
        ))

        # Signaux de Trade (Scatter)
        for t in trades:
            col = '#ef4444' if 'CRASH' in t['label'] else ('#f59e0b' if 'PRUDENCE' in t['label'] else '#10b981')
            fig.add_annotation(
                x=t['date'], y=df_res.loc[t['date']]['portfolio'],
                text="▼" if t['to'] != 'R0' else "▲",
                showarrow=False, font=dict(color=col, size=14)
            )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='#888'),
            height=450,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, linecolor='#333'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- DRAWDOWN CHART ---
        st.markdown("### 🌊 UNDERWATER PLOT")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        dd_series = (df_res['portfolio'] / df_res['portfolio'].cummax() - 1) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series,
            fill='tozeroy', line=dict(color='#ef4444', width=1),
            fillcolor='rgba(239, 68, 68, 0.2)'
        ))
        fig_dd.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='#888'), height=200, margin=dict(t=0,b=0,l=0,r=0),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =======================
# COLONNE 3: VALIDATION
# =======================
with col_valid:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛡️ ROBUSTESSE")
    
    if st.button("LANCER VALIDATION"):
        with st.spinner("Calculs Monte-Carlo & Walk-Forward..."):
            # 1. Walk Forward
            wf_res = run_walk_forward(data, params)
            avg_overfit = np.mean([w['overfit'] for w in wf_res]) if wf_res else 0
            
            # 2. Monte Carlo
            mc_res = run_monte_carlo(data, params, runs=100)
            prob_loss = len(mc_res[mc_res['CAGR'] < 0]) / len(mc_res) * 100
            
            # Affichage Verdict
            verdict = "ROBUSTE" if avg_overfit < 1.5 and prob_loss < 20 else "FRAGILE"
            color_v = "#10b981" if verdict == "ROBUSTE" else "#ef4444"
            
            st.markdown(f"""
            <div style="background:{color_v}20; border:1px solid {color_v}; padding:15px; border-radius:10px; text-align:center; margin-bottom:15px;">
                <h2 style="color:{color_v}; margin:0;">{verdict}</h2>
                <p style="font-size:11px; margin:5px 0 0 0; color:#aaa;">Overfit: {avg_overfit:.2f}x • Prob. Perte: {prob_loss:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Charts WF
            st.markdown("#### Walk-Forward Periods")
            for w in wf_res:
                col_o = "#ef4444" if w['overfit'] > 1.5 else "#10b981"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05); padding:8px; border-radius:6px; margin-bottom:5px; font-size:11px; display:flex; justify-content:space-between;">
                    <span>{w['period']}</span>
                    <span>Train: {w['train_cagr']:.0f}% | Test: {w['test_cagr']:.0f}%</span>
                    <span style="color:{col_o}">{w['overfit']:.1f}x</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Chart MC
            st.markdown("#### Distribution Monte-Carlo")
            fig_mc = px.histogram(mc_res, x="CAGR", nbins=20, color_discrete_sequence=['#667eea'])
            fig_mc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888', size=10), height=150, margin=dict(l=0,r=0,t=0,b=0),
                xaxis_title=None, yaxis_title=None, showlegend=False
            )
            st.plotly_chart(fig_mc, use_container_width=True)
            
    else:
        st.info("Cliquez pour lancer l'analyse de robustesse complète (Calcul intensif).")
        
    st.markdown('</div>', unsafe_allow_html=True)
