import pandas as pd
import numpy as np
import yfinance as yf

class MarketData:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date

    def fetch(self):
        """Récupère et nettoie les données"""
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
            if df.empty:
                return pd.DataFrame()
            
            # Gestion multi-index de yfinance parfois capricieuse
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Adj Close', level=0, axis=1) if 'Adj Close' in df.columns.levels[0] else df['Close']
            else:
                df = df[['Adj Close']] if 'Adj Close' in df.columns else df[['Close']]

            df.columns = ['Price']
            df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"Erreur data: {e}")
            return pd.DataFrame()

class RiskMetrics:
    def __init__(self, returns_series, risk_free_rate=0.0):
        self.returns = returns_series
        self.rf = risk_free_rate

    def get_metrics(self):
        """Retourne un dictionnaire avec tous les KPIs"""
        if self.returns.empty: return {}
        
        vol = self.returns.std() * np.sqrt(252)
        
        # Sharpe
        excess_ret = self.returns.mean() * 252 - self.rf
        sharpe = excess_ret / vol if vol != 0 else 0
        
        # Sortino (Downside risk uniquement)
        neg_ret = self.returns[self.returns < 0]
        downside_std = neg_ret.std() * np.sqrt(252)
        sortino = excess_ret / downside_std if downside_std != 0 else 0

        # VaR 95%
        var_95 = self.returns.quantile(0.05)

        return {
            "Volatilité": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "VaR (95%)": var_95
        }

    @staticmethod
    def max_drawdown(price_series):
        rolling_max = price_series.cummax()
        drawdown = (price_series - rolling_max) / rolling_max
        return drawdown.min()

class VectorBacktester:
    def __init__(self, data):
        self.df = data.copy()

    def run_sma_strategy(self, short_window, long_window):
        # 1. Calculs Indicateurs
        self.df['SMA_Short'] = self.df['Price'].rolling(window=short_window).mean()
        self.df['SMA_Long'] = self.df['Price'].rolling(window=long_window).mean()

        # 2. Logique de Signal (Crossover)
        self.df['Signal'] = 0
        self.df.loc[self.df['SMA_Short'] > self.df['SMA_Long'], 'Signal'] = 1
        
        # 3. Position (Shifté de 1 jour pour éviter le biais de futur)
        self.df['Position'] = self.df['Signal'].shift(1)
        
        # 4. Rendements Stratégie
        self.df['Strategy_Returns'] = self.df['Log_Returns'] * self.df['Position']
        
        # 5. Courbes cumulées (Base 100)
        self.df['Benchmark_Curve'] = (1 + self.df['Log_Returns']).cumprod() * 100
        self.df['Strategy_Curve'] = (1 + self.df['Strategy_Returns']).cumprod() * 100
        
        return self.df.dropna()
