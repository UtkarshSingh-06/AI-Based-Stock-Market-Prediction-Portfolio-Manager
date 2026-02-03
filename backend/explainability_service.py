"""
Model explainability: feature importance (SHAP-style) for why a prediction went up/down.
Lightweight implementation using gradient or permutation when full SHAP is not available.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def feature_importance_permutation(
    predict_fn,
    X: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    baseline_pred: Optional[float] = None,
) -> Dict[str, float]:
    """
    Permutation importance: shuffle each feature and see change in prediction.
    predict_fn(X) -> scalar or array; we use last prediction if array.
    Returns dict of feature_name -> importance (positive = increases prediction when higher).
    """
    if X.ndim == 2:
        X = X.reshape(1, *X.shape)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        base = predict_fn(X)
        base = float(np.asarray(base).flat[-1]) if base is not None else 0.0
    except Exception as e:
        logger.warning(f"Baseline prediction failed: {e}")
        base = baseline_pred or 0.0

    n_features = X.shape[-1]
    if len(feature_names) != n_features:
        feature_names = [f"f{i}" for i in range(n_features)]
    importances = np.zeros(n_features)
    X_perm = X.copy()

    for f in range(n_features):
        delta = 0.0
        for _ in range(n_repeats):
            np.random.shuffle(X_perm[:, :, f] if X_perm.ndim == 3 else X_perm[:, f])
            try:
                p = predict_fn(X_perm)
                p = float(np.asarray(p).flat[-1]) if p is not None else 0.0
                delta += (p - base)
            except Exception:
                pass
            # Restore
            X_perm[:, :, f] = X[:, :, f] if X.ndim == 3 else X[:, f]
        importances[f] = delta / n_repeats if n_repeats else 0.0

    out = {}
    for i, name in enumerate(feature_names):
        out[name] = round(float(importances[i]), 6)
    return out


def explain_prediction(
    feature_importance: Dict[str, float],
    top_n: int = 5,
) -> Dict:
    """
    Build human-readable explanation from feature importance.
    Returns top drivers (positive/negative) and a short summary.
    """
    sorted_f = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top = sorted_f[:top_n]
    positive = [t for t in top if t[1] > 0]
    negative = [t for t in top if t[1] < 0]
    return {
        "top_drivers": [{"feature": k, "impact": v} for k, v in top],
        "increased_prediction": [{"feature": k, "impact": v} for k, v in positive],
        "decreased_prediction": [{"feature": k, "impact": v} for k, v in negative],
        "summary": _summary_text(positive, negative),
    }


def _summary_text(positive: List, negative: List) -> str:
    if not positive and not negative:
        return "No strong feature drivers."
    parts = []
    if positive:
        names = [p[0] for p in positive[:2]]
        parts.append(f"Higher {', '.join(names)} pushed the prediction up.")
    if negative:
        names = [n[0] for n in negative[:2]]
        parts.append(f"Lower {', '.join(names)} pushed the prediction down.")
    return " ".join(parts)
