import pandas as pd
import numpy as np

class RiskMetrics:
    @staticmethod
    def get_metrics(returns_series):
        if returns_series.empty: return {"Sharpe": 0, "Vol": 0, "VaR": 0, "MaxDD": 0}
        
        # Volatilité Annualisée
        vol = returns_series.std() * np.sqrt(252)
        
        # Sharpe (Rf=0 simplifie)
        mean_ret = returns_series.mean() * 252
        sharpe = mean_ret / vol if vol != 0 else 0
        
        # VaR 95%
        var_95 = returns_series.quantile(0.05)
        
        # Max Drawdown
        cum_ret = (1 + returns_series).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        return {
            "Volatilité": vol,
            "Sharpe": sharpe,
            "VaR (95%)": var_95,
            "Max Drawdown": max_dd
        }

class VectorBacktester:
    def __init__(self, price_series):
        # On s'attend à une Series pandas (une seule colonne de prix)
        self.df = pd.DataFrame(price_series)
        self.df.columns = ['Price']
        self.df['Log_Returns'] = np.log(self.df['Price'] / self.df['Price'].shift(1))

    def run_sma_strategy(self, short_window, long_window):
        # 1. Indicateurs
        self.df['SMA_Short'] = self.df['Price'].rolling(window=short_window).mean()
        self.df['SMA_Long'] = self.df['Price'].rolling(window=long_window).mean()

        # 2. Signal (1 = Achat, 0 = Cash)
        self.df['Signal'] = np.where(self.df['SMA_Short'] > self.df['SMA_Long'], 1, 0)
        
        # 3. Position (Shifté de 1 jour)
        self.df['Position'] = self.df['Signal'].shift(1)
        
        # 4. Performance
        self.df['Strategy_Returns'] = self.df['Log_Returns'] * self.df['Position']
        self.df['Strategy_Curve'] = (1 + self.df['Strategy_Returns']).cumprod() * 100
        self.df['BuyHold_Curve'] = (1 + self.df['Log_Returns']).cumprod() * 100
        
        return self.df.dropna()
