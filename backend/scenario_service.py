"""
Scenario predictions and portfolio VaR.
Scenario: conditional forecast under vol shock or market move.
Portfolio VaR: value-at-risk for a set of positions using predictions/correlations.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def scenario_adjust_prediction(
    base_predicted_return: float,
    scenario: str,
    vol_multiplier: Optional[float] = None,
    market_shock_pct: Optional[float] = None,
) -> float:
    """
    Adjust a single-period predicted return for a scenario.
    base_predicted_return: e.g. 0.02 for +2%
    scenario: 'base' | 'high_vol' | 'market_down_5' | 'market_up_2'
    """
    if scenario == "base" or (not vol_multiplier and not market_shock_pct):
        return base_predicted_return
    out = base_predicted_return
    if scenario == "high_vol" and vol_multiplier:
        # Scale return by vol (e.g. more uncertainty, dampen magnitude)
        out = out / vol_multiplier
    if scenario == "market_down_5" and market_shock_pct is not None:
        out = out + (market_shock_pct / 100.0)  # e.g. -5% market
    if scenario == "market_up_2" and market_shock_pct is not None:
        out = out + (market_shock_pct / 100.0)
    return out


def portfolio_var_historical(
    positions: List[Dict],
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """
    positions: list of { "symbol": str, "weight": float } (weights sum to 1)
    returns_matrix: (n_days, n_assets) daily returns.
    Returns VaR (positive number = loss at confidence level) and optional components.
    """
    if not positions or returns_matrix is None or returns_matrix.size == 0:
        return {"var_pct": 0.0, "var_amount": 0.0, "confidence": confidence}
    weights = np.array([p.get("weight", 1.0 / len(positions)) for p in positions])
    weights = weights / weights.sum()
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(-1, 1)
    if returns_matrix.shape[1] != len(weights):
        # If single asset, broadcast
        if returns_matrix.shape[1] == 1 and len(weights) == 1:
            pass
        else:
            weights = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]
    portfolio_returns = returns_matrix @ weights
    var_pct = float(np.percentile(portfolio_returns, (1 - confidence) * 100))
    # VaR as positive loss
    var_pct = abs(min(0, var_pct))
    return {
        "var_pct": round(var_pct, 4),
        "var_amount": None,  # caller can multiply by portfolio value
        "confidence": confidence,
        "horizon_days": 1,
    }


def portfolio_var_from_predictions(
    symbols: List[str],
    predicted_returns: List[float],
    volatility_scale: float = 1.5,
    confidence: float = 0.95,
) -> Dict:
    """
    Simplified VaR using predicted returns and a volatility scale (e.g. 1.5x for stress).
    Assumes diagonal (no correlation). For demo; use historical correlation in production.
    """
    if not symbols or not predicted_returns:
        return {"var_pct": 0.0, "confidence": confidence}
    n = len(symbols)
    weights = np.ones(n) / n
    # Simulate distribution: mean = predicted return, std = volatility_scale * abs(predicted)
    mu = np.mean(predicted_returns)
    sigma = max(0.01, np.std(predicted_returns) * volatility_scale)
    # 1-day VaR from normal approx
    z = 1.65 if confidence >= 0.95 else 1.28
    var_pct = max(0.0, -(mu - z * sigma))
    return {
        "var_pct": round(float(var_pct) * 100, 4),
        "confidence": confidence,
        "volatility_scale": volatility_scale,
    }
