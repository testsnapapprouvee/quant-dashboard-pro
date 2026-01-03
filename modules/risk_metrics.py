# modules/risk_metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional

class RiskMetrics:
    """
    Institutional-grade risk metrics for Leveraged Arbitrage strategies.
    Focuses on tail risk (VaR/CVaR) and drawdown duration (Ulcer Index).
    """

    @staticmethod
    def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        """
        Calculates annualized rolling volatility.
        Standard convention: 21 trading days (1 month).
        """
        return returns.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculates Historical Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Why Historical?
        Leveraged ETFs exhibit significant skewness and kurtosis. 
        Parametric (Normal distribution) assumptions underestimate tail risk in LETFs.
        
        Args:
            returns (pd.Series): Daily logarithmic or percentage returns.
            confidence_level (float): 0.95 or 0.99.
            
        Returns:
            Dict containing VaR and CVaR values (positive float representation of loss).
        """
        if returns.empty:
            return {"VaR": 0.0, "CVaR": 0.0}

        # Invert confidence level for percentile calculation (e.g., 95% conf -> 5th percentile)
        alpha = 1.0 - confidence_level
        
        # Calculate VaR (The threshold)
        var_value = returns.quantile(alpha)
        
        # Calculate CVaR (Mean of returns falling below the VaR threshold)
        cvar_value = returns[returns <= var_value].mean()
        
        return {
            f"VaR_{int(confidence_level*100)}": abs(var_value),  # Returned as positive loss magnitude
            f"CVaR_{int(confidence_level*100)}": abs(cvar_value) # Returned as positive loss magnitude
        }

    @staticmethod
    def calculate_ulcer_index(prices: pd.Series) -> float:
        """
        Calculates the Ulcer Index (UI).
        
        Rationale:
        Standard deviation penalizes upside volatility. UI only penalizes downside.
        It measures the depth and duration of drawdowns (Root Mean Squared Drawdown).
        Crucial for assessing the 'psychological cost' of the X2 leverage regime.
        """
        if prices.empty:
            return 0.0
            
        # 1. Calculate percentage drawdown from rolling max
        rolling_max = prices.cummax()
        drawdowns = (prices - rolling_max) / rolling_max
        
        # 2. Square the drawdowns
        drawdowns_squared = drawdowns ** 2
        
        # 3. Mean of squares
        mean_squared = drawdowns_squared.mean()
        
        # 4. Square root
        ulcer_index = np.sqrt(mean_squared)
        
        return ulcer_index * 100  # Return as percentage

    @staticmethod
    def get_full_risk_profile(prices: pd.Series) -> Dict[str, float]:
        """
        Aggregates all risk metrics for the Strategy vs Benchmark comparison.
        """
        if prices.empty or len(prices) < 2:
            return {}

        returns = prices.pct_change().dropna()
        
        # 1. Tail Risk
        var_95 = RiskMetrics.calculate_var_cvar(returns, 0.95)
        var_99 = RiskMetrics.calculate_var_cvar(returns, 0.99)
        
        # 2. Stress Metrics
        ulcer = RiskMetrics.calculate_ulcer_index(prices)
        
        # 3. Standard Metrics (for reference)
        vol_ann = returns.std() * np.sqrt(252)
        
        return {
            "Vol_Ann": vol_ann,
            "Ulcer_Index": ulcer,
            "VaR_95": var_95["VaR_95"],
            "CVaR_95": var_95["CVaR_95"],
            "VaR_99": var_99["VaR_99"],
            "CVaR_99": var_99["CVaR_99"]
        }
