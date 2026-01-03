import pandas as pd
import numpy as np

class RiskMetrics:
    @staticmethod
    def get_metrics(returns_series):
        if returns_series.empty: return {}
        vol = returns_series.std() * np.sqrt(252)
        mean_ret = returns_series.mean() * 252
        sharpe = mean_ret / vol if vol != 0 else 0
        var_95 = returns_series.quantile(0.05)
        
        # Max Drawdown
        cum = (1 + returns_series).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        max_dd = dd.min()
        
        return {"Vol": vol, "Sharpe": sharpe, "VaR": var_95, "MaxDD": max_dd}

class VectorBacktester:
    def __init__(self, price_series):
        self.df = pd.DataFrame(price_series)
        self.df.columns = ['Price']
        self.df['Log_Ret'] = np.log(self.df['Price'] / self.df['Price'].shift(1))

    def run_strategy(self, short_w, long_w):
        self.df['SMA_S'] = self.df['Price'].rolling(short_w).mean()
        self.df['SMA_L'] = self.df['Price'].rolling(long_w).mean()
        self.df['Signal'] = np.where(self.df['SMA_S'] > self.df['SMA_L'], 1, 0)
        self.df['Pos'] = self.df['Signal'].shift(1)
        self.df['Strat_Ret'] = self.df['Log_Ret'] * self.df['Pos']
        self.df['Curve'] = (1 + self.df['Strat_Ret']).cumprod() * 100
        self.df['BH_Curve'] = (1 + self.df['Log_Ret']).cumprod() * 100
        return self.df.dropna()
