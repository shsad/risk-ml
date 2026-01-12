from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix

def expected_cost(y_true, y_prob, threshold: float, cost_fn: float = 5.0, cost_fp: float = 1.0) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return cost_fn * fn + cost_fp * fp

def find_best_threshold(y_true, y_prob, cost_fn: float = 5.0, cost_fp: float = 1.0, n_grid: int = 200):
    thresholds = np.linspace(0.01, 0.99, n_grid)
    costs = [expected_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in thresholds]
    idx = int(np.argmin(costs))
    return float(thresholds[idx]), float(costs[idx])