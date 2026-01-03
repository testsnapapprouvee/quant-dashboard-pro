# modules/portfolio_optimizer.py

import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    @staticmethod
    def optimize_max_sharpe(returns_df):
        """
        Optimize portfolio weights to maximize Sharpe ratio
        """
        n = len(returns_df.columns)
        if n == 0:
            return []

        def neg_sharpe(w):
            port_ret = np.sum(returns_df.mean() * w) * 252
            port_vol = np.sqrt(np.dot(w.T, np.dot(returns_df.cov() * 252, w)))
            return -port_ret / port_vol if port_vol > 0 else 0

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(neg_sharpe, [1/n]*n, bounds=bounds, constraints=cons)
        return res.x if res.success else [1/n]*n
