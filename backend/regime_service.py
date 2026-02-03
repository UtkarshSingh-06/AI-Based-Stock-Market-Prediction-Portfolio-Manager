"""
Market regime detection: volatility (VIX-style) and trend for conditioning predictions.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# VIX symbol for broad market volatility
VIX_SYMBOL = "^VIX"
SPY_SYMBOL = "SPY"

# Regime labels
REGIME_LOW_VOL = "low_vol"
REGIME_HIGH_VOL = "high_vol"
REGIME_TRENDING_UP = "trending_up"
REGIME_TRENDING_DOWN = "trending_down"
REGIME_CRISIS = "crisis"
REGIME_UNKNOWN = "unknown"


def get_vix_level() -> Optional[float]:
    """Fetch current VIX level. Returns None if unavailable."""
    try:
        t = yf.Ticker(VIX_SYMBOL)
        hist = t.history(period="5d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug(f"VIX fetch failed: {e}")
    return None


def volatility_20d_from_prices(close: pd.Series) -> float:
    """Annualized 20-day volatility (std of returns * sqrt(252))."""
    if close is None or len(close) < 20:
        return 0.0
    ret = close.pct_change().dropna()
    if len(ret) < 20:
        return 0.0
    return float(ret.tail(20).std() * (252 ** 0.5) * 100)


def trend_signal_from_prices(close: pd.Series, short: int = 10, long: int = 50) -> Optional[float]:
    """Simple trend: (SMA_short - SMA_long) / SMA_long. Positive = uptrend."""
    if close is None or len(close) < long:
        return None
    sma_short = close.rolling(short).mean().iloc[-1]
    sma_long = close.rolling(long).mean().iloc[-1]
    if sma_long and sma_long != 0:
        return (sma_short - sma_long) / sma_long
    return None


def detect_regime(
    symbol: Optional[str] = None,
    as_of_date: Optional[datetime] = None,
    vix_level: Optional[float] = None,
    volatility_20d: Optional[float] = None,
    trend_signal: Optional[float] = None,
) -> Dict:
    """
    Detect market regime. If symbol is provided, fetches data for that symbol.
    Otherwise uses VIX for broad market. Returns regime label and metrics.
    """
    as_of = as_of_date or datetime.utcnow()
    result = {
        "symbol": symbol,
        "snapshot_date": as_of.isoformat(),
        "regime": REGIME_UNKNOWN,
        "vix_level": None,
        "volatility_20d": None,
        "trend_signal": None,
        "metadata": {},
        "computed_at": datetime.utcnow().isoformat(),
    }

    # VIX (broad market)
    if vix_level is None:
        vix_level = get_vix_level()
    result["vix_level"] = vix_level

    # Symbol-specific vol and trend
    if symbol:
        try:
            end = as_of + timedelta(days=1)
            start = end - timedelta(days=120)
            df = yf.download(
                symbol.strip().upper(),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                threads=False,
                auto_adjust=True,
            )
            if df is not None and not df.empty and "Close" in df.columns:
                close = df["Close"].astype(float)
                if volatility_20d is None:
                    volatility_20d = volatility_20d_from_prices(close)
                if trend_signal is None:
                    trend_signal = trend_signal_from_prices(close)
        except Exception as e:
            logger.debug(f"Regime data for {symbol} failed: {e}")

    result["volatility_20d"] = volatility_20d
    result["trend_signal"] = trend_signal

    # Classify regime
    vol = vix_level if vix_level is not None else (volatility_20d or 0)
    if vol is not None:
        if vol >= 35:
            result["regime"] = REGIME_CRISIS
        elif vol >= 22:
            result["regime"] = REGIME_HIGH_VOL
        else:
            result["regime"] = REGIME_LOW_VOL

    if trend_signal is not None:
        if trend_signal > 0.02 and result["regime"] == REGIME_LOW_VOL:
            result["regime"] = REGIME_TRENDING_UP
        elif trend_signal < -0.02 and result["regime"] == REGIME_LOW_VOL:
            result["regime"] = REGIME_TRENDING_DOWN

    return result
