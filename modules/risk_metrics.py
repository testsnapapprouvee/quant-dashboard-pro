# modules/risk_metrics.py

import numpy as np
import pandas as pd

class RiskMetrics:
    @staticmethod
    def get_metrics(returns_series):
        """
        Calculate core risk metrics for a portfolio.
        Input: pd.Series of portfolio returns
        Output: dict with Vol, Sharpe, VaR, Max Drawdown, CAGR
        """
        if returns_series.empty:
            return {"Vol": 0, "Sharpe": 0, "VaR": 0, "MaxDD": 0, "CAGR": 0}

        # Annualized volatility
        vol = returns_series.std() * np.sqrt(252)
        mean_ret = returns_series.mean() * 252

        # Sharpe ratio
        sharpe = mean_ret / vol if vol != 0 else 0

        # VaR 95%
        var_95 = returns_series.quantile(0.05)

        # Max Drawdown
        cum = (1 + returns_series).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        max_dd = dd.min()

        # CAGR
        total_ret = cum.iloc[-1] / cum.iloc[0] - 1
        cagr = (1 + total_ret) ** (252 / len(returns_series)) - 1

        return {"Vol": vol, "Sharpe": sharpe, "VaR": var_95, "MaxDD": max_dd, "CAGR": cagr}
