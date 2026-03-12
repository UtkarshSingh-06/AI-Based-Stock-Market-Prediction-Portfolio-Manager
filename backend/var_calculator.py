"""
Value at Risk (VaR) Calculator

Two methods:
1. Historical VaR: Sort historical returns, find the 5th percentile (95% VaR).
   = The loss you won't exceed 95% of the time.
2. Parametric VaR: Uses mean (μ) and standard deviation (σ), assumes normal distribution.
   VaR = -(μ - z*σ) where z is the standard normal quantile (e.g. 1.65 for 95%).
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_portfolio_returns(
    symbols: List[str],
    weights: Optional[List[float]] = None,
    lookback_days: int = 252,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Download historical price data and compute daily returns for each symbol.
    Returns (portfolio_returns, individual_returns_df).
    """
    if not symbols:
        raise ValueError("symbols cannot be empty")
    symbols = [s.strip().upper() for s in symbols]
    n = len(symbols)
    weights = np.array(weights) if weights and len(weights) == n else np.ones(n) / n
    weights = weights / weights.sum()

    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    returns_list = []
    valid_symbols = []
    valid_weights = []

    for i, sym in enumerate(symbols):
        try:
            df = yf.download(sym, start=start_str, end=end_str, progress=False, threads=False, auto_adjust=True)
            if df is None or df.empty or "Close" not in df.columns:
                logger.warning(f"No data for {sym}")
                continue
            ret = df["Close"].pct_change().dropna()
            if len(ret) < 30:
                continue
            returns_list.append(ret)
            valid_symbols.append(sym)
            valid_weights.append(weights[i])
        except Exception as e:
            logger.warning(f"Failed to download {sym}: {e}")
            continue

    if not returns_list:
        raise ValueError("Could not download returns for any symbol")

    valid_weights = np.array(valid_weights) / sum(valid_weights)
    combined = pd.concat(returns_list, axis=1, join="inner")
    combined.columns = valid_symbols
    portfolio_returns = (combined * valid_weights).sum(axis=1).values
    return portfolio_returns, combined


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """
    Historical VaR: sort returns, find the (1-confidence)*100 percentile.
    E.g. 95% VaR = 5th percentile = loss you won't exceed 95% of the time.
    Returns VaR as positive % (e.g. 2.5 means 2.5% max loss).
    """
    if returns is None or len(returns) == 0:
        return {"var_pct": 0.0, "method": "historical", "percentile": (1 - confidence) * 100}
    percentile = (1 - confidence) * 100
    var_raw = np.percentile(returns, percentile)
    var_pct = abs(min(0, var_raw)) * 100
    return {
        "var_pct": round(float(var_pct), 4),
        "method": "historical",
        "confidence": confidence,
        "percentile": percentile,
        "samples": len(returns),
    }


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """
    Parametric VaR: assume returns ~ N(μ, σ²).
    VaR = -(μ - z*σ) where z = quantile of standard normal.
    E.g. 95% confidence → z ≈ 1.65.
    """
    if returns is None or len(returns) == 0:
        return {"var_pct": 0.0, "method": "parametric"}
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma <= 0:
        sigma = 1e-8
    z_map = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z = z_map.get(confidence, 1.65)
    var_raw = -(mu - z * sigma)
    var_pct = max(0, var_raw) * 100
    return {
        "var_pct": round(float(var_pct), 4),
        "method": "parametric",
        "confidence": confidence,
        "mean_daily_return_pct": round(mu * 100, 4),
        "std_daily_return_pct": round(sigma * 100, 4),
        "z_score": z,
        "samples": len(returns),
    }


def var_calculator(
    symbols: List[str],
    weights: Optional[List[float]] = None,
    confidence: float = 0.95,
    lookback_days: int = 252,
) -> Dict:
    """
    Download portfolio returns, compute both Historical and Parametric VaR,
    and return comparison.
    """
    portfolio_returns, individual_df = download_portfolio_returns(symbols, weights, lookback_days)
    hist = historical_var(portfolio_returns, confidence)
    param = parametric_var(portfolio_returns, confidence)
    diff_pct = round(param["var_pct"] - hist["var_pct"], 4)
    return {
        "symbols": symbols,
        "weights": weights,
        "confidence": confidence,
        "lookback_days": lookback_days,
        "historical_var": hist,
        "parametric_var": param,
        "comparison": {
            "historical_var_pct": hist["var_pct"],
            "parametric_var_pct": param["var_pct"],
            "difference_pct": diff_pct,
            "parametric_minus_historical": diff_pct,
            "note": "Parametric assumes normal returns; Historical uses actual distribution. "
            "Parametric often underestimates tail risk in real markets.",
        },
        "sample_count": len(portfolio_returns),
        "portfolio_value_var_note": "Multiply portfolio value by var_pct/100 for dollar VaR.",
    }
