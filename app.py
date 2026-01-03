import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from datetime import datetime, timedelta

# --- 1. CONFIGURATION DESIGN (NOIR & VIOLET) ---
st.set_page_config(page_title="Predict", layout="wide", page_icon="🔮")

# Injection CSS : Noir Profond & Accents Violets (#8b5cf6)
st.markdown("""
<style>
    /* Fond principal noir */
    .stApp { background-color: #050505; color: #e0e0e0; }
    
    /* Titres et textes */
    h1, h2, h3 { color: #ffffff !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* Métriques (Cartes) */
    div[data-testid="stMetric"] {
        background-color: #121212;
        border: 1px solid #333;
        border-left: 5px solid #8b5cf6;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetricLabel"] { color: #a0a0a0; }
    div[data-testid="stMetricValue"] { color: #c4b5fd; text-shadow: 0 0 10px rgba(139, 92, 246, 0.3); }

    /* Inputs et Sidebar */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
    .stTextInput > div > div > input { background-color: #1a1a1a; color: white; border-color: #8b5cf6; }
    
    /* Boutons et Sliders */
    div.stSlider > div[data-baseweb="slider"] > div > div { background-color: #8b5cf6 !important; }
    button[kind="secondary"] { border-color: #8b5cf6; color: #8b5cf6; }
    button[kind="primary"] { background-color: #8b5cf6; border: none; }
</style>
""", unsafe_allow_html=True)

# --- 2. FONCTIONS (MOTEUR) ---

@st.cache_data(ttl=3600)
def get_data_robust(tickers, start_date):
    """Télécharge les données et gère les erreurs silencieusement"""
    try:
        # On force unique=True pour éviter les doublons si l'utilisateur tape 2 fois le même
        tickers = list(set(tickers))
        df = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=True)
        prices = pd.DataFrame()

        # Gestion format Yahoo (MultiIndex vs Single Index)
        if len(tickers) == 1:
            t = tickers[0]
            if 'Close' in df.columns: prices[t] = df['Close']
            elif t in df.columns: prices[t] = df[t]['Close']
        else:
            for t in tickers:
                if t in df.columns: prices[t] = df[t]['Close']
                
        prices = prices.fillna(method='ffill').dropna()
        return prices
    except Exception:
        return pd.DataFrame()

def optimize(returns):
    """Optimisation Max Sharpe"""
    n = len(returns.columns)
    def neg_sharpe(w):
        r = np.sum(returns.mean()*w)*252
        v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
        return -r/v if v > 0 else 0
    
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    bnds = tuple((0,1) for _ in range(n))
    res = minimize(neg_sharpe, [1/n]*n, bounds=bnds, constraints=cons)
    return res.x

# --- 3. INTERFACE PREDICT ---

st.title("PREDICT 🔮")
st.caption("Plateforme d'Analyse Quantitative & Arbitrage")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres")
    
    # Champ de recherche
    default_tickers = "PUST.PA, LQQ.PA"
    tickers_input = st.text_input("Tickers (Yahoo)", default_tickers, help="Séparez les tickers par une virgule")
    
    # Nettoyage de l'input et suppression des vides
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip() != ""]
    
    # Période
    years = st.slider("Historique (Années)", 1, 10, 3)
    start_date = datetime.now() - timedelta(days=years*365)
    
    st.divider()
    
    # Stratégie
    mode = st.radio("Mode Allocation", ["Manuel", "Optimisation AI"])
    weights = []
    
    if mode == "Manuel":
        # --- FIX CRASH INDEXERROR ---
        if len(tickers) >= 2:
            w = st.slider(f"Poids {tickers[0]}", 0, 100, 50)
            weights = [w/100, 1-(w/100)]
            st.write(f"🟣 {tickers[0]}: **{weights[0]:.0%}**")
            st.write(f"⚪ {tickers[1]}: **{weights[1]:.0%}**")
        elif len(tickers) == 1:
            st.warning("⚠️ Ajoutez un 2ème ticker pour l'arbitrage (ex: , LQQ.PA)")
            weights = [1.0] # 100% sur le seul actif
        else:
            st.error("❌ Entrez au moins un ticker")
            weights = []

# --- MAIN LOGIC ---

if len(tickers) > 0:
    # 1. Chargement
    data = get_data_robust(tickers, start_date)

    if not data.empty and len(data.columns) > 0:
        # Petit indicateur de succès
        st.sidebar.success(f"✅ Données chargées : {len(data)} jours")
        
        returns = data.pct_change().dropna()
        
        # 2. Optimisation (si activée)
        if mode == "Optimisation AI" and len(data.columns) >= 2:
            with st.spinner("🔮 L'IA calcule l'allocation optimale..."):
                weights = optimize(returns)
            st.sidebar.markdown(f"**Optimisé :**")
            # Affichage sécurisé des poids optimisés
            for i, col in enumerate(data.columns):
                 st.sidebar.info(f"{col}: {weights[i]:.1%}")
        
        # Sécurité taille poids (si l'utilisateur change les tickers en cours de route)
        if len(weights) != len(data.columns):
            weights = [1/len(data.columns)] * len(data.columns)

        # 3. Calculs Portefeuille
        port_returns = returns.dot(weights)
        
        # Création des séries de prix cumulés (Base 100) pour calculs KPIs
        cum_port = (1 + port_returns).cumprod() * 100
        
        # KPIs
        total_ret = cum_port.iloc[-1]/100 - 1
        cagr = (total_ret + 1)**(252/len(data)) - 1
        vol = port_returns.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        
        # --- DASHBOARD ---
        
        # Ligne des KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAGR", f"{cagr:.2%}")
        k2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        k3.metric("Volatilité", f"{vol:.2%}")
        k4.metric("Perf Totale", f"{total_ret:.2%}")

        st.divider()

        # --- SECTION GRAPHIQUE AVANCÉE ---
        
        col_opt, _ = st.columns([1, 4])
        with col_opt:
            show_base100 = st.toggle("Voir en Base 100", value=True)

        fig = go.Figure()

        if show_base100:
            # MODE BASE 100
            # 1. Le Portefeuille (Violet brillant + Remplissage)
            fig.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port, 
                name="PREDICT PORTFOLIO", 
                mode='lines',
                line=dict(color='#8b5cf6', width=3),
                fill='tozeroy', 
                fillcolor='rgba(139, 92, 246, 0.1)' 
            ))
            
            # 2. Les ETFs individuels
            colors = ['#a0a0a0', '#4b5563', '#d1d5db'] 
            for i, col in enumerate(data.columns):
                cum_asset = (1 + returns[col]).cumprod() * 100
                fig.add_trace(go.Scatter(
                    x=cum_asset.index, y=cum_asset, 
                    name=f"{col} (Base 100)",
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot')
                ))
            
            title_graph = "Performance Comparée (Base 100)"
            y_title = "Valeur (Base 100)"
            
        else:
            # MODE PRIX RÉELS
            for i, col in enumerate(data.columns):
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col], 
                    name=f"Prix {col}",
                    mode='lines',
                    line=dict(width=2)
                ))
                
            title_graph = "Historique des Prix Réels"
            y_title = "Prix (€/$)"

        fig.update_layout(
            title=title_graph,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            yaxis=dict(title=y_title, gridcolor='#333'),
            xaxis=dict(gridcolor='#333'),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- PARTIE ANALYSE CORRÉLATION ---
        with st.expander("📊 Matrice de Corrélation"):
            if len(data.columns) > 1:
                corr = returns.corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Purples', zmin=-1, zmax=1)
                fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_corr)
            else:
                st.info("Il faut au moins 2 actifs pour afficher la corrélation.")

    else:
        st.error("❌ Ticker introuvable ou données vides.")
        st.info("Astuce : Vérifiez qu'il y a bien une virgule entre les tickers (ex: 'PUST.PA, LQQ.PA').")
else:
    st.info("👋 Bienvenue sur Predict. Entrez des tickers à gauche pour commencer.")
