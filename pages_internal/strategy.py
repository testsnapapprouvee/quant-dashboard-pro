# pages_internal/strategy.py

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Modules internes
from modules.arbitrage_signals import ArbitrageSignals  # signals arbitrage
from modules.data_engine import get_data  # moteur de données
from modules.risk_metrics import RiskMetrics
from modules.leverage_diagnostics import LeverageDiagnostics
from modules.backtest_engine import VectorizedBacktester  # ton engine existant
from modules.optimizer import Optimizer  # ton optimizer existant

# ==========================================
# PAGE STRATEGIE
# ==========================================
st.set_page_config(page_title="Strategy", layout="wide")

st.title("📈 Strategy & Arbitrage")

# --------------------------
# 1️⃣ Paramètres Utilisateur
# --------------------------
col1, col2 = st.columns([1, 2])

with col1:
    presets = {
        "Nasdaq 100 (Amundi)": ["LQQ.PA", "PUST.PA"],
        "S&P 500 (US)": ["SSO", "SPY"],
        "Custom": []
    }
    sel_preset = st.selectbox("Universe", list(presets.keys()))
    if sel_preset == "Custom":
        tickers_input = st.text_input("Tickers (Risk, Safe)", "LQQ.PA, PUST.PA")
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
    else:
        tickers = presets[sel_preset]

    period_options = ["YTD", "1Y", "3YR", "5YR", "Custom"]
    sel_period = st.selectbox("Period", period_options, index=3)

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
    else:
        start_d = st.date_input("Start", datetime(2022,1,1))
        end_d = st.date_input("End", datetime.now())

# --------------------------
# 2️⃣ Récupération des données
# --------------------------
data = get_data(tickers, start_d, end_d)

if data.empty or len(data) < 10:
    st.error("❌ Pas assez de données. Vérifiez vos tickers ou la période.")
    st.stop()

# --------------------------
# 3️⃣ Paramètres de Simulation
# --------------------------
if 'params' not in st.session_state:
    st.session_state['params'] = {
        'thresh': 5, 'panic': 15, 'recovery': 30,
        'allocPrudence': 50, 'allocCrash': 100, 'rollingWindow': 60, 'confirm': 2
    }

params = st.session_state['params']

params['thresh'] = st.slider("Threshold (%)", 2, 10, params['thresh'], 0.5)
params['panic'] = st.slider("Panic (%)", 10, 30, params['panic'], 1)
params['recovery'] = st.slider("Recovery (%)", 20, 60, params['recovery'], 5)
params['allocPrudence'] = st.slider("Prudence (X1%)", 0, 100, params['allocPrudence'], 10)
params['allocCrash'] = st.slider("Crash (X1%)", 0, 100, params['allocCrash'], 10)
params['confirm'] = st.slider("Confirm (Days)", 1, 3, params['confirm'], 1)

profile = st.selectbox("Objective", ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])

# --------------------------
# 4️⃣ Run Simulation
# --------------------------
df_res, trades = BacktestEngine.run_simulation(data, params)

# --------------------------
# 5️⃣ Onglets Streamlit
# --------------------------
tabs = st.tabs(["Performance", "Risk & Leverage", "Arbitrage Signals"])

# --- TAB 1: Performance ---
with tabs[0]:
    st.subheader("Performance Portfolio")
    st.line_chart(df_res['portfolio'])

# --- TAB 2: Risk & Leverage ---
with tabs[1]:
    st.subheader("Risk Metrics")
    risk_s = RiskMetrics.get_full_risk_profile(df_res['portfolio'])
    st.json(risk_s)

    st.subheader("Leverage Diagnostics")
    lev_beta = LeverageDiagnostics.calculate_realized_beta(data)
    st.line_chart(lev_beta['Realized_Beta'] if not lev_beta.empty else pd.Series())

# --- TAB 3: Arbitrage Signals ---
with tabs[2]:
    st.subheader("Arbitrage Signals")
    if not data.empty:
        arb_sig = ArbitrageSignals.calculate_relative_strength(data)
        st.dataframe(arb_sig.head(20))

        signal_status = ArbitrageSignals.get_signal_status(arb_sig)
        st.json(signal_status)
    else:
        st.info("Module Arbitrage non disponible ou données insuffisantes")
