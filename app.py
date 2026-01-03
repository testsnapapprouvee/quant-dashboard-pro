import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from quant_engine import MarketData, RiskMetrics, VectorBacktester

# --- CONFIGURATION INITIALE ---
st.set_page_config(
    page_title="Quant Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM (POUR LE DESIGN DEFINI) ---
st.markdown("""
    <style>
    /* Force le style des métriques */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        font-size: 24px;
        color: #00FFAA;
    }
    /* Style global */
    .stApp {
        background-color: #0E1117;
    }
    /* Bouton principal */
    div.stButton > button {
        background-color: #00FFAA;
        color: black;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ QUANT LAB")
    st.caption("Engine v1.0 | Vectorized")
    st.markdown("---")
    
    ticker = st.text_input("ACTIF (Ticker Yahoo)", value="BTC-USD").upper()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Début", value=datetime(2020, 1, 1))
    end_date = col_d2.date_input("Fin", value=datetime.now())
    
    st.markdown("### 🛠 Stratégie (SMA)")
    short_w = st.number_input("Moyenne Courte", min_value=5, max_value=100, value=20)
    long_w = st.number_input("Moyenne Longue", min_value=50, max_value=365, value=50)
    
    run = st.button("LANCER LE BACKTEST", use_container_width=True)

# --- FONCTION D'AFFICHAGE GRAPHIQUE ---
def plot_chart(df):
    fig = go.Figure()
    
    # Courbe Benchmark (Buy & Hold)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Benchmark_Curve'],
        mode='lines', name='Buy & Hold',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    # Courbe Stratégie
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Strategy_Curve'],
        mode='lines', name='Stratégie SMA',
        line=dict(color='#00FFAA', width=2)
    ))
    
    fig.update_layout(
        title="Performance Comparée (Base 100)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        xaxis_title="",
        yaxis_title="Valeur Portefeuille",
        hovermode="x unified"
    )
    return fig

# --- MAIN LOGIC ---
if run:
    try:
        with st.spinner('Chargement des données & Calculs vectoriels...'):
            # 1. Ingestion
            engine = MarketData(ticker, start_date, end_date)
            df_raw = engine.fetch()
            
            if df_raw.empty:
                st.error("❌ Aucune donnée trouvée. Vérifie le ticker.")
            else:
                # 2. Backtest
                bt = VectorBacktester(df_raw)
                df_res = bt.run_sma_strategy(short_w, long_w)
                
                if df_res.empty:
                    st.warning("⚠️ Pas assez de données pour calculer ces moyennes mobiles.")
                else:
                    # 3. Calculs Métriques
                    metrics_calc = RiskMetrics(df_res['Strategy_Returns'])
                    kpis = metrics_calc.get_metrics()
                    mdd = RiskMetrics.max_drawdown(df_res['Strategy_Curve'])
                    
                    # --- DASHBOARD ---
                    
                    # Ligne 1 : Les chiffres clés
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Rendement Total", f"{(df_res['Strategy_Curve'].iloc[-1]/100 - 1):.2%}")
                    kpi2.metric("Ratio de Sharpe", f"{kpis['Sharpe']:.2f}")
                    kpi3.metric("Max Drawdown", f"{mdd:.2%}", delta_color="inverse")
                    kpi4.metric("Volatilité", f"{kpis['Volatilité']:.2%}")
                    
                    st.markdown("---")
                    
                    # Ligne 2 : Le Graphique Principal
                    st.plotly_chart(plot_chart(df_res), use_container_width=True)
                    
                    # Ligne 3 : Analyse détaillée
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.subheader("📊 Distribution")
                        st.bar_chart(df_res['Strategy_Returns'])
                    with c2:
                        st.subheader("📝 Données récentes")
                        st.dataframe(df_res[['Price', 'SMA_Short', 'SMA_Long', 'Signal', 'Strategy_Curve']].tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur critique : {str(e)}")

else:
    st.info("👈 Configure la stratégie à gauche pour démarrer.")
    st.markdown("""
        <div style='text-align: center; color: gray; margin-top: 50px;'>
        Waiting for input...
        </div>
    """, unsafe_allow_html=True)
