# modules/monte_carlo.py

import pandas as pd
import numpy as np

class MonteCarloSimulator:
    def __init__(self, returns_series):
        self.returns = returns_series

    def run_simulation(self, n_simulations=1000, horizon_days=252):
        """
        Simulate portfolio evolution using Geometric Brownian Motion
        Returns a DataFrame: each column = 1 simulation
        """
        mean = self.returns.mean()
        vol = self.returns.std()
        last_price = 100  # base 100
        sim_df = pd.DataFrame()

        for i in range(n_simulations):
            rand_returns = np.random.normal(loc=mean, scale=vol, size=horizon_days)
            price_path = last_price * np.cumprod(1 + rand_returns)
            sim_df[f"Sim_{i+1}"] = price_path

        return sim_df
