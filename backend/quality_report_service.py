"""
Prediction quality report: prediction vs actual, vs baseline (naive), direction hit rate.
Computes and stores PredictionQualityMetric for the Quality Report API.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def compute_mape(actual: List[float], predicted: List[float]) -> float:
    """Mean absolute percentage error."""
    if not actual or not predicted or len(actual) != len(predicted):
        return 0.0
    n = len(actual)
    s = 0.0
    for a, p in zip(actual, predicted):
        if a and a != 0:
            s += abs((a - p) / a)
    return (s / n) * 100.0 if n else 0.0


def compute_mae(actual: List[float], predicted: List[float]) -> float:
    """Mean absolute error."""
    if not actual or not predicted or len(actual) != len(predicted):
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def direction_hit_rate(actual: List[float], predicted: List[float]) -> float:
    """Fraction of steps where predicted direction matched actual (up/down)."""
    if not actual or not predicted or len(actual) != len(predicted) or len(actual) < 2:
        return 0.0
    hits = 0
    total = 0
    for i in range(1, len(actual)):
        actual_dir = 1 if actual[i] > actual[i - 1] else -1 if actual[i] < actual[i - 1] else 0
        pred_dir = 1 if predicted[i] > predicted[i - 1] else -1 if predicted[i] < predicted[i - 1] else 0
        if actual_dir != 0:
            total += 1
            if actual_dir == pred_dir:
                hits += 1
    return (hits / total * 100.0) if total else 0.0


def naive_forecast_mape(actual: List[float]) -> float:
    """Naive forecast: next = last. Returns MAPE of that baseline."""
    if not actual or len(actual) < 2:
        return 0.0
    naive_pred = actual[:-1]
    actual_next = actual[1:]
    return compute_mape(actual_next, naive_pred)


def compute_vs_naive_improvement(actual: List[float], predicted: List[float]) -> Optional[float]:
    """
    % improvement of our MAPE over naive (positive = we are better).
    improvement = (naive_mape - our_mape) / naive_mape * 100
    """
    our_mape = compute_mape(actual, predicted)
    naive_mape = naive_forecast_mape(actual)
    if naive_mape and naive_mape > 0:
        return round((naive_mape - our_mape) / naive_mape * 100.0, 2)
    return None


def compute_quality_metrics(
    actual: List[float],
    predicted: List[float],
    symbol: str,
    horizon_days: int,
    period_start: datetime,
    period_end: datetime,
    abstention_count: int = 0,
) -> Dict:
    """Compute all metrics for a symbol/horizon window."""
    mape = compute_mape(actual, predicted) if (actual and predicted) else None
    mae = compute_mae(actual, predicted) if (actual and predicted) else None
    hit_rate = direction_hit_rate(actual, predicted) if (actual and predicted) else None
    vs_naive = compute_vs_naive_improvement(actual, predicted) if (actual and predicted) else None
    return {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "period_start": period_start,
        "period_end": period_end,
        "sample_count": len(actual) if actual else 0,
        "mape": round(mape, 4) if mape is not None else None,
        "mae": round(mae, 4) if mae is not None else None,
        "direction_hit_rate": round(hit_rate, 2) if hit_rate is not None else None,
        "vs_naive_improvement": vs_naive,
        "abstention_count": abstention_count,
    }


def persist_quality_metric(db: Session, metric: Dict) -> None:
    """Persist a PredictionQualityMetric row (use database.PredictionQualityMetric)."""
    from database import PredictionQualityMetric
    row = PredictionQualityMetric(
        symbol=metric["symbol"],
        horizon_days=metric["horizon_days"],
        period_start=metric["period_start"],
        period_end=metric["period_end"],
        sample_count=metric["sample_count"],
        mape=metric.get("mape"),
        mae=metric.get("mae"),
        direction_hit_rate=metric.get("direction_hit_rate"),
        vs_naive_improvement=metric.get("vs_naive_improvement"),
        vs_buy_hold_note=metric.get("vs_buy_hold_note"),
        abstention_count=metric.get("abstention_count", 0),
    )
    db.add(row)
    db.commit()
