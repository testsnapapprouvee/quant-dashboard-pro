import pandas as pd
import numpy as np
import yfinance as yf

# 1. Le Récupérateur de Données
class MarketData:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date

    def fetch(self):
        try:
            # Récupération silencieuse pour ne pas polluer l'app
            df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
            if df.empty: return pd.DataFrame()
            
            # Nettoyage et standardisation
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Adj Close', level=0, axis=1) if 'Adj Close' in df.columns.levels[0] else df['Close']
            else:
                df = df[['Adj Close']] if 'Adj Close' in df.columns else df[['Close']]

            df.columns = ['Price']
            # Calcul des rendements Log (Base des maths fi)
            df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
            df.dropna(inplace=True)
            return df
        except:
            return pd.DataFrame()

# 2. Le Calculateur de Risque
class RiskMetrics:
    @staticmethod
    def get_metrics(returns_series):
        if returns_series.empty: return {"Sharpe": 0, "Volatilité": 0, "Var95": 0}
        
        # Volatilité Annualisée
        vol = returns_series.std() * np.sqrt(252)
        
        # Sharpe Ratio (Simplifié avec taux sans risque à 0 pour l'instant)
        mean_ret = returns_series.mean() * 252
        sharpe = mean_ret / vol if vol != 0 else 0
        
        # Value at Risk (95%)
        var_95 = returns_series.quantile(0.05)
        
        return {
            "Volatilité": vol,
            "Sharpe": sharpe,
            "VaR (95%)": var_95
        }

# 3. Le Backtester Rapide
class VectorBacktester:
    def __init__(self, data):
        self.df = data.copy()

    def run_strategy(self, short_window, long_window):
        # Indicateurs
        self.df['SMA_Short'] = self.df['Price'].rolling(window=short_window).mean()
        self.df['SMA_Long'] = self.df['Price'].rolling(window=long_window).mean()

        # Signal (1 = Achat, 0 = Cash)
        self.df['Signal'] = np.where(self.df['SMA_Short'] > self.df['SMA_Long'], 1, 0)
        
        # On décale d'un jour pour simuler le trading réel
        self.df['Position'] = self.df['Signal'].shift(1)
        
        # Performance
        self.df['Strategy_Returns'] = self.df['Log_Returns'] * self.df['Position']
        self.df['Equity_Curve'] = (1 + self.df['Strategy_Returns']).cumprod()
        
        return self.df.dropna()
