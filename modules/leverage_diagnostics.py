# modules/leverage_diagnostics.py
import pandas as pd
import numpy as np
from typing import Dict

class LeverageDiagnostics:
    """
    Analyse la relation structurelle entre un actif à levier (X2) et son sous-jacent (X1).
    Utile quand X2 et X1 partagent le même sous-jacent (ex: SPY vs SSO).
    """

    @staticmethod
    def calculate_realized_beta(data: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """
        Calcule le Beta roulant (Levier Réalisé) de X2 par rapport à X1.
        Si le Beta chute (ex: passe de 2.0 à 1.5), le levier perd de son efficacité.
        """
        if 'X2' not in data.columns or 'X1' not in data.columns:
            return pd.DataFrame()

        rets = data[['X2', 'X1']].pct_change().dropna()
        
        # Covariance Roulante / Variance Roulante X1 = Beta
        rolling_cov = rets['X2'].rolling(window).cov(rets['X1'])
        rolling_var = rets['X1'].rolling(window).var()
        
        beta = rolling_cov / rolling_var
        
        res = pd.DataFrame(index=data.index)
        res['Realized_Beta'] = beta
        return res.dropna()

    @staticmethod
    def calculate_leverage_health(data: pd.DataFrame) -> Dict[str, float]:
        """
        Diagnostic instantané pour le comité d'investissement.
        """
        if data.empty: return {}
        
        rets = data[['X2', 'X1']].pct_change().dropna()
        
        # Levier Moyen sur la période
        total_beta = rets['X2'].cov(rets['X1']) / rets['X1'].var()
        
        # Volatility Ratio (Est-ce que X2 est bien 2x plus volatil que X1 ?)
        vol_ratio = rets['X2'].std() / rets['X1'].std()
        
        # Tracking Error (Écart type de la différence des rendements ajustés)
        # Idéalement proche de 0
        diff = rets['X2'] - (rets['X1'] * 2) 
        tracking_err = diff.std()

        return {
            "Realized_Leverage": total_beta,
            "Vol_Ratio": vol_ratio,
            "Tracking_Error": tracking_err
        }

    @staticmethod
    def detect_decay_regime(data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Détecte si nous sommes dans un régime favorable au levier (Trend) 
        ou défavorable (Decay/Sideways).
        
        Si X1 fait 0% sur la période mais X2 fait -5%, c'est du pure Decay.
        """
        if data.empty: return pd.DataFrame()
        
        res = pd.DataFrame(index=data.index)
        
        # Rendement roulant
        r_x1 = data['X1'].pct_change(window)
        r_x2 = data['X2'].pct_change(window)
        
        # Decay Spread: La différence entre la perf réelle de X2 et la perf théorique (2 * X1)
        # Note: C'est une approximation simplifiée pour l'affichage
        res['Decay_Spread'] = r_x2 - (2 * r_x1)
        
        return res.dropna()
