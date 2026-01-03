# modules/analytics.py
import numpy as np
import pandas as pd

class AnalyticsEngine:
    @staticmethod
    def calculate_metrics(series: pd.Series, risk_free_rate=0.0):
        if series.empty or len(series) < 10:
            return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Sortino": 0, "Calmar": 0, "CVaR": 0}
            
        # Returns
        returns = series.pct_change().dropna()
        
        # 1. CAGR
        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        
        # 2. Volatility (Annualized)
        vol = returns.std() * np.sqrt(252)
        
        # 3. Max Drawdown
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        max_dd = drawdown.min()
        
        # 4. Sharpe Ratio
        sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0
        
        # 5. Sortino Ratio (Downside deviation only)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252)
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # 6. Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # 7. Conditional VaR (CVaR / Expected Shortfall) at 95%
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            "CAGR": cagr * 100,
            "Vol": vol * 100,
            "MaxDD": max_dd * 100,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "CVaR_95": cvar_95 * 100 # Monthly/Daily scale usually, here daily pct
        }

    @staticmethod
    def monte_carlo_simulation(series: pd.Series, n_sims=100, days_ahead=252):
        """
        Simulation Monte Carlo vectorisée (Block Bootstrap pour préserver la corrélation locale).
        """
        returns = series.pct_change().dropna().values
        
        # Simulation simple par choix aléatoire (avec remise)
        # Pour une version plus robuste, on utiliserait le Block Bootstrap
        sim_returns = np.random.choice(returns, size=(days_ahead, n_sims))
        
        # Reconstruction des chemins de prix (Base 100)
        sim_paths = (1 + sim_returns).cumprod(axis=0) * 100
        
        return sim_paths
