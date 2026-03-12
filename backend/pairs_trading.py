"""
Pairs Trading Strategy Backtest

1. Find two stocks in the same sector (e.g., Coca-Cola & Pepsi)
2. Test for cointegration using statsmodels
3. Calculate the spread between them (price_A - hedge_ratio * price_B)
4. When spread deviates >2 std: long the underperformer, short the outperformer
5. Backtest to see if it's profitable

Resources: statsmodels cointegration, Backtrader
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_pair_data(
    symbol1: str,
    symbol2: str,
    lookback_days: int = 504,
) -> pd.DataFrame:
    """Download OHLCV for both symbols and align on common dates."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    df1 = yf.download(symbol1.strip().upper(), start=start_str, end=end_str, progress=False, threads=False, auto_adjust=True)
    df2 = yf.download(symbol2.strip().upper(), start=start_str, end=end_str, progress=False, threads=False, auto_adjust=True)

    if df1 is None or df1.empty:
        raise ValueError(f"No data for {symbol1}")
    if df2 is None or df2.empty:
        raise ValueError(f"No data for {symbol2}")
    p1 = df1["Close"].copy() if "Close" in df1.columns else df1.iloc[:, 0].copy()
    p2 = df2["Close"].copy() if "Close" in df2.columns else df2.iloc[:, 0].copy()
    if isinstance(p1, pd.DataFrame):
        p1 = p1.iloc[:, 0]
    if isinstance(p2, pd.DataFrame):
        p2 = p2.iloc[:, 0]
    p1 = p1.rename(symbol1)
    p2 = p2.rename(symbol2)
    combined = pd.concat([p1, p2], axis=1).dropna()
    if len(combined) < 100:
        raise ValueError("Insufficient overlapping data for pair")
    return combined


def test_cointegration(
    prices1: np.ndarray,
    prices2: np.ndarray,
) -> Tuple[bool, float, float]:
    """
    Test for cointegration using Engle-Granger (statsmodels).
    Returns (is_cointegrated, p_value, hedge_ratio).
    """
    try:
        from statsmodels.tsa.stattools import coint
        from statsmodels.regression.linear_model import OLS
    except ImportError:
        logger.warning("statsmodels not installed; using OLS hedge ratio and skip coint test")
        # Fallback: compute hedge ratio via OLS, assume cointegrated if correlation high
        x = np.column_stack([np.ones_like(prices2), prices2])
        beta = np.linalg.lstsq(x, prices1, rcond=None)[0]
        hedge_ratio = beta[1]
        corr = np.corrcoef(prices1, prices2)[0, 1]
        return corr > 0.7, 0.05, float(hedge_ratio)

    # Engle-Granger cointegration test
    score, p_value, _ = coint(prices1, prices2)
    is_cointegrated = p_value < 0.05

    # Hedge ratio: regress price1 on price2
    x = np.column_stack([np.ones_like(prices2), prices2])
    beta = np.linalg.lstsq(x, prices1, rcond=None)[0]
    hedge_ratio = float(beta[1])

    return is_cointegrated, float(p_value), hedge_ratio


def calculate_spread(
    prices1: np.ndarray,
    prices2: np.ndarray,
    hedge_ratio: float,
) -> np.ndarray:
    """Spread = price1 - hedge_ratio * price2."""
    return prices1 - hedge_ratio * prices2


def backtest_pairs_strategy(
    symbol1: str,
    symbol2: str,
    threshold_std: float = 2.0,
    lookback_days: int = 504,
    initial_capital: float = 100000,
    commission: float = 0.001,
) -> Dict:
    """
    Backtest pairs trading: when spread deviates >threshold_std, long the underperformer
    and short the outperformer. Mean-revert when spread returns to mean.
    """
    df = download_pair_data(symbol1, symbol2, lookback_days)
    p1 = df[symbol1].values
    p2 = df[symbol2].values
    dates = df.index

    is_coint, p_value, hedge_ratio = test_cointegration(p1, p2)
    spread = calculate_spread(p1, p2, hedge_ratio)
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    if std_spread <= 0:
        std_spread = 1e-8
    z_score = (spread - mean_spread) / std_spread

    # Position: +1 = long spread (long symbol1, short symbol2), -1 = short spread
    # When z > +2: spread high, symbol1 outperformed -> short spread (short symbol1, long symbol2)
    # When z < -2: spread low, symbol1 underperformed -> long spread (long symbol1, short symbol2)
    position = np.zeros(len(spread))
    pos = 0
    for i in range(1, len(z_score)):
        if z_score[i] > threshold_std and pos <= 0:
            pos = -1
        elif z_score[i] < -threshold_std and pos >= 0:
            pos = 1
        elif -0.5 < z_score[i] < 0.5:
            pos = 0
        position[i] = pos

    # PnL: 1 unit spread = long 1 share symbol1, short hedge_ratio shares symbol2
    # PnL per unit = spread_change. Scale by units = capital * 0.2 / (std_spread * avg_price)
    spread_change = np.diff(spread)
    units = (initial_capital * 0.2) / (std_spread + 1e-10)
    position_pnl = position[1:] * spread_change * units
    portfolio_values = [initial_capital]
    capital = initial_capital
    trades = []

    for i in range(1, len(position)):
        prev_pos = position[i - 1]
        curr_pos = position[i]
        if curr_pos != prev_pos and prev_pos != 0:
            trades.append({"date": str(dates[i]), "action": "close", "prev_pos": int(prev_pos)})
        if curr_pos != prev_pos and curr_pos != 0:
            trades.append({"date": str(dates[i]), "action": "open", "pos": int(curr_pos)})
        if i <= len(position_pnl):
            capital += position_pnl[i - 1]
        portfolio_values.append(capital)

    final_value = portfolio_values[-1]
    total_return_pct = (final_value - initial_capital) / initial_capital * 100
    pv = np.array(portfolio_values)
    rets = np.diff(pv) / (pv[:-1] + 1e-10)
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 0 and np.std(rets) > 0 else 0
    max_dd = 0
    peak = pv[0]
    for v in pv:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "cointegration": {
            "is_cointegrated": is_coint,
            "p_value": round(p_value, 6),
            "hedge_ratio": round(hedge_ratio, 6),
        },
        "spread_stats": {
            "mean": float(mean_spread),
            "std": float(std_spread),
        },
        "backtest": {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "num_trades": len(trades),
            "threshold_std": threshold_std,
        },
        "trades_sample": trades[:10] if trades else [],
        "profitable": total_return_pct > 0,
    }


def backtest_with_backtrader(
    symbol1: str,
    symbol2: str,
    threshold_std: float = 2.0,
    lookback_days: int = 504,
) -> Dict:
    """
    Backtest using Backtrader (optional). Falls back to custom backtest if not installed.
    """
    try:
        import backtrader as bt  # noqa: F401
        # Full Backtrader integration requires custom data feeds for pairs
        # Use custom backtest as default; extend later for full Backtrader support
        pass
    except ImportError:
        pass
    result = backtest_pairs_strategy(symbol1, symbol2, threshold_std, lookback_days)
    result["engine"] = "custom"
    return result
