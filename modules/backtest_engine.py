# modules/backtest_engine.py
import pandas as pd
import numpy as np

class VectorizedBacktester:
    """
    Moteur de backtest vectorisé haute performance.
    Gère la détection de régime et l'allocation d'actifs sans boucles Python lentes.
    """
    
    def __init__(self, data: pd.DataFrame, params: dict):
        self.data = data.copy()
        self.params = params
        self.results = None
        self.trades = None

    def run(self):
        # 1. Préparation des données
        df = self.data.copy()
        rolling_w = int(self.params['rollingWindow'])
        
        # 2. Calcul des Drawdowns (Vectorisé)
        # On utilise le prix X2 (Risk Asset) pour déterminer le stress du marché
        df['Rolling_Max'] = df['X2'].rolling(window=rolling_w, min_periods=1).max()
        df['Drawdown'] = (df['X2'] / df['Rolling_Max']) - 1.0
        
        # 3. Détection des Régimes (Logique Vectorielle)
        # R0 (Offensif) par défaut
        # R1 (Prudence) si DD < threshold
        # R2 (Crash) si DD < panic
        
        # On initialise tout à R0 (0)
        # 0: Offensif, 1: Prudence, 2: Crash
        conditions = [
            (df['Drawdown'] <= -self.params['panic'] / 100.0),
            (df['Drawdown'] <= -self.params['thresh'] / 100.0)
        ]
        choices = [2, 1] # R2, R1
        
        # Calcul du régime brut
        df['Regime_Raw'] = np.select(conditions, choices, default=0)
        
        # 4. Logique de Recovery (Hystérésis)
        # C'est la seule partie difficilement vectorisable purement sans scan, 
        # mais on peut l'approximer ou utiliser numba pour la vitesse si besoin.
        # Pour rester en pandas pur et rapide, on applique un shift et une logique de confirmation.
        
        # Confirmation (Lissage du signal)
        df['Regime_Smooth'] = df['Regime_Raw'].rolling(int(self.params['confirm'])).max() # Prend le pire cas sur la fenêtre
        df['Regime_Final'] = df['Regime_Smooth'].fillna(0).astype(int)

        # 5. Allocation (Vectorisée)
        # Définition des poids cibles pour X1 (Safe)
        alloc_map = {
            0: 0.0,  # R0: 0% Safe, 100% Risk
            1: self.params['allocPrudence'] / 100.0,
            2: self.params['allocCrash'] / 100.0
        }
        
        df['Weight_X1'] = df['Regime_Final'].map(alloc_map)
        df['Weight_X2'] = 1.0 - df['Weight_X1']
        
        # 6. Calcul des Rendements Portefeuille
        # On décale les poids de 1 jour (Lag) car on trade à la clôture pour le lendemain (ou à l'ouverture suivante)
        df['Weight_X1_Shift'] = df['Weight_X1'].shift(1).fillna(0.0)
        df['Weight_X2_Shift'] = df['Weight_X2'].shift(1).fillna(1.0)
        
        df['Ret_X1'] = df['X1'].pct_change().fillna(0.0)
        df['Ret_X2'] = df['X2'].pct_change().fillna(0.0)
        
        # Frais de transaction
        # On détecte les changements d'allocation
        df['Delta_Alloc'] = df['Weight_X1_Shift'].diff().abs()
        cost_bps = self.params.get('cost', 0.001)
        df['Tx_Cost'] = df['Delta_Alloc'] * cost_bps
        
        # Rendement Net
        df['Strategy_Ret'] = (df['Weight_X1_Shift'] * df['Ret_X1']) + \
                             (df['Weight_X2_Shift'] * df['Ret_X2']) - \
                             df['Tx_Cost']
                             
        # Courbe d'équité
        df['Portfolio'] = (1 + df['Strategy_Ret']).cumprod() * 100.0
        
        # Benchmarks Base 100
        df['Bench_X2'] = (1 + df['Ret_X2']).cumprod() * 100.0
        df['Bench_X1'] = (1 + df['Ret_X1']).cumprod() * 100.0
        
        self.results = df
        
        return df

    def get_trades(self):
        """Extrait le journal des trades à partir des changements de régime"""
        if self.results is None: return []
        
        changes = self.results[self.results['Regime_Final'].diff() != 0].dropna()
        trades = []
        regime_labels = {0: "OFFENSIF", 1: "PRUDENCE", 2: "CRASH"}
        
        for date, row in changes.iterrows():
            new_regime = int(row['Regime_Final'])
            prev_regime = 0 # Par défaut ou à récupérer
            
            trades.append({
                'date': date,
                'regime': regime_labels.get(new_regime, "N/A"),
                'alloc_x1': f"{row['Weight_X1']:.0%}",
                'cost': row['Tx_Cost']
            })
        return trades
