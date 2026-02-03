"""
Data quality service: per-symbol scoring and gating for predictions.
Ensures we only predict when data is complete, fresh, and free of major gaps/outliers.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Minimum score (0-100) to allow predictions
MIN_QUALITY_SCORE_FOR_PREDICTION = 60.0
# Max staleness in hours
MAX_STALENESS_HOURS = 48
# Max allowed gap days (trading days)
MAX_GAP_DAYS = 3


def compute_data_quality(symbol: str, lookback_days: int = 90) -> Dict:
    """
    Compute data quality score for a symbol.
    Returns dict with score (0-100), completeness_pct, staleness_hours, has_gaps, details.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return _low_quality_result("Invalid symbol", 0)

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            threads=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning(f"Data download failed for {symbol}: {e}")
        return _low_quality_result(f"Download failed: {e}", 0)

    if df is None or df.empty or "Close" not in df.columns:
        return _low_quality_result("No data or missing Close", 0)

    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    details = {}

    # 1) Completeness: expected trading days vs actual
    expected_days = _trading_days_between(df.index.min(), df.index.max())
    actual_days = len(df)
    completeness_pct = (actual_days / expected_days * 100) if expected_days > 0 else 0
    details["expected_days"] = expected_days
    details["actual_days"] = actual_days
    details["completeness_pct"] = round(completeness_pct, 2)

    # 2) Staleness: hours since last data point
    last_ts = df.index.max()
    if hasattr(last_ts, "tzinfo") and last_ts.tzinfo:
        now = datetime.utcnow().replace(tzinfo=last_ts.tzinfo)
    else:
        now = datetime.utcnow()
    if last_ts.tzinfo:
        from datetime import timezone
        now = now.replace(tzinfo=timezone.utc)
    try:
        delta = now - last_ts
        staleness_hours = delta.total_seconds() / 3600.0
    except Exception:
        staleness_hours = 999.0
    details["last_data_at"] = str(last_ts)
    details["staleness_hours"] = round(staleness_hours, 2)

    # 3) Gaps: consecutive missing trading days
    sorted_index = df.sort_index().index
    gaps = 0
    max_gap = 0
    for i in range(1, len(sorted_index)):
        diff = (sorted_index[i] - sorted_index[i - 1]).days
        if diff > 1:
            gaps += 1
            if diff > max_gap:
                max_gap = diff
    has_gaps = gaps > 0 or max_gap > MAX_GAP_DAYS
    details["gap_count"] = gaps
    details["max_gap_days"] = max_gap

    # 4) Outliers: extreme returns
    close = df["Close"].astype(float)
    returns = close.pct_change().dropna()
    if len(returns) > 0:
        threshold = 0.15  # 15% single-day move is suspicious
        outlier_count = int((returns.abs() > threshold).sum())
    else:
        outlier_count = 0
    details["outlier_count"] = outlier_count

    # Score: weighted combination
    score = 100.0
    if completeness_pct < 80:
        score -= (80 - completeness_pct) * 0.5
    if staleness_hours > 24:
        score -= min(40, staleness_hours / 6)
    if has_gaps:
        score -= 15
    if outlier_count > 5:
        score -= min(20, outlier_count * 2)
    score = max(0.0, min(100.0, score))

    return {
        "symbol": symbol,
        "score": round(score, 2),
        "completeness_pct": round(completeness_pct, 2),
        "staleness_hours": round(staleness_hours, 2),
        "has_gaps": has_gaps,
        "gap_count": gaps,
        "outlier_count": outlier_count,
        "details": details,
        "computed_at": datetime.utcnow().isoformat(),
        "allowed_for_prediction": score >= MIN_QUALITY_SCORE_FOR_PREDICTION,
    }


def _trading_days_between(start, end) -> int:
    """Approximate trading days between two dates."""
    if start is None or end is None:
        return 0
    try:
        delta = (end - start).days
        # ~252 trading days per year
        return max(1, int(delta * 252 / 365))
    except Exception:
        return 1


def _low_quality_result(reason: str, score: float = 0.0) -> Dict:
    return {
        "symbol": "",
        "score": score,
        "completeness_pct": 0.0,
        "staleness_hours": 999.0,
        "has_gaps": True,
        "gap_count": 0,
        "outlier_count": 0,
        "details": {"error": reason},
        "computed_at": datetime.utcnow().isoformat(),
        "allowed_for_prediction": False,
    }


def is_prediction_allowed(symbol: str, lookback_days: int = 90) -> Tuple[bool, Optional[Dict]]:
    """
    Returns (allowed: bool, quality_result: dict or None).
    If quality check fails, quality_result explains why.
    """
    result = compute_data_quality(symbol, lookback_days=lookback_days)
    return result.get("allowed_for_prediction", False), result
