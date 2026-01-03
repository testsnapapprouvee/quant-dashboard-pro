# modules/arbitrage_signals.py
import pandas as pd
import numpy as np
from typing import Dict

class ArbitrageSignals:
    """
    Détecte les opportunités tactiques d'entrée/sortie basées sur la valeur relative
    entre l'actif à risque (X2) et l'actif refuge (X1).
    """

    @staticmethod
    def calculate_relative_strength(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calcule le ratio de prix X2/X1 et son Z-Score (Écart à la moyenne).
        C'est un indicateur de Mean Reversion classique en Pair Trading.
        """
        if 'X2' not in data.columns or 'X1' not in data.columns:
            return pd.DataFrame()

        res = pd.DataFrame(index=data.index)
        
        # 1. Ratio de Prix (Relative Strength)
        res['Ratio'] = data['X2'] / data['X1']
        
        # 2. Moyenne Mobile et Écart-Type du Ratio
        rolling_mean = res['Ratio'].rolling(window=window).mean()
        rolling_std = res['Ratio'].rolling(window=window).std()
        
        # 3. Z-Score : (Prix Actuel - Moyenne) / Volatilité
        # Indique à combien d'écarts-types nous sommes de la "normale"
        res['Z_Score'] = (res['Ratio'] - rolling_mean) / rolling_std
        
        return res.dropna()

    @staticmethod
    def get_signal_status(z_score_series: pd.Series) -> Dict[str, str]:
        """
        Interprète le dernier Z-Score pour le Comité d'Investissement.
        """
        if z_score_series.empty: return {}
        
        last_z = z_score_series.iloc[-1]
        
        status = "NEUTRE"
        color = "gray"
        action = "Hold"
        
        if last_z > 2.0:
            status = "OVERBOUGHT (X2 Expensive)"
            color = "#ef4444" # Red
            action = "Prise de profits ?"
        elif last_z < -2.0:
            status = "OVERSOLD (X2 Cheap)"
            color = "#10b981" # Green
            action = "Opportunité d'achat Aggressive ?"
        elif last_z > 1.0:
            status = "MILDLY RICH"
            color = "#f59e0b"
        elif last_z < -1.0:
            status = "MILDLY CHEAP"
            color = "#3b82f6"
            
        return {
            "Current_Z": last_z,
            "Status": status,
            "Color": color,
            "Action": action
        }
