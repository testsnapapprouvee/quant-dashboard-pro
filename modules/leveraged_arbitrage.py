# modules/leveraged_arbitrage.py

import pandas as pd
import numpy as np

class LeveragedArbitrage:
    @staticmethod
    def compute_optimal_weights(prices_df, roles_dict):
        """
        Simple example:
        - Assign weight to maximize exposure to leveraged ETF while keeping normal ETF as hedge
        - roles_dict: {ticker: "Normal"/"X2"/"Short"}
        """
        n = len(prices_df.columns)
        weights = np.zeros(n)
        if n == 0: return {}

        # Arbitrage logic (simple, extensible)
        levered_idx = [i for i, t in enumerate(prices_df.columns) if roles_dict[t] == "X2"]
        normal_idx = [i for i, t in enumerate(prices_df.columns) if roles_dict[t] == "Normal"]
        short_idx = [i for i, t in enumerate(prices_df.columns) if roles_dict[t] == "Short"]

        if levered_idx:
            weights[levered_idx] = 0.6 / len(levered_idx)
        if normal_idx:
            weights[normal_idx] = 0.3 / len(normal_idx)
        if short_idx:
            weights[short_idx] = 0.1 / len(short_idx)

        # Normalize to sum 1
        weights /= weights.sum()
        return dict(zip(prices_df.columns, weights))
