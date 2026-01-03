# modules/optimization.py
import numpy as np
import pandas as pd
from .backtest_engine import VectorizedBacktester
from .analytics import AnalyticsEngine

class SmartOptimizer:
    @staticmethod
    def run_random_search(data, n_iter=50, objective='Calmar'):
        """
        Optimisation par recherche aléatoire (Random Search).
        Souvent plus efficace que Grid Search pour trouver des optima globaux.
        """
        results = []
        
        # Espace de paramètres (Bounds)
        param_space = {
            'thresh': np.arange(2.0, 12.0, 0.5),
            'panic': np.arange(10.0, 40.0, 1.0),
            'recovery': np.arange(20, 60, 5),
            'allocPrudence': [20, 30, 40, 50, 60, 70],
            'allocCrash': [80, 90, 100],
            'rollingWindow': [40, 50, 60, 80, 100],
            'confirm': [1, 2, 3],
            'cost': [0.001]
        }
        
        best_score = -np.inf
        best_params = {}
        
        for _ in range(n_iter):
            # Sampling aléatoire
            current_params = {k: np.random.choice(v) for k, v in param_space.items()}
            
            # Contrainte logique : Panic doit être > Threshold
            if current_params['panic'] <= current_params['thresh']:
                continue
                
            # Backtest
            bt = VectorizedBacktester(data, current_params)
            res = bt.run()
            metrics = AnalyticsEngine.calculate_metrics(res['Portfolio'])
            
            # Score avec pénalité sur le nombre de trades (Overtrading)
            n_switches = len(bt.get_trades())
            penalty = n_switches * 0.05 # Pénalise légèrement le churn excessif
            
            score = metrics.get(objective, 0) - penalty
            
            if score > best_score:
                best_score = score
                best_params = current_params
                
        return best_params, best_score
