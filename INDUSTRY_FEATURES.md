# Industry-Level Features

This document describes the industry-level features added to the Stock Prediction platform to differentiate it from typical academic/side projects and address real-world issues.

## 1. Data Quality & Gating

- **Service**: `backend/data_quality_service.py`
- **API**: `GET /api/v1/data-quality/{symbol}?lookback_days=90`
- **Behavior**: Computes a 0–100 data quality score per symbol (completeness, staleness, gaps, outliers). Predictions are **gated**: if the score is below the threshold (default 60), the predict endpoint returns **503** with `data_quality` details instead of running the model.
- **Config**: `MIN_QUALITY_SCORE_FOR_PREDICTION`, `MAX_STALENESS_HOURS`, `MAX_GAP_DAYS` in the service.

## 2. Market Regime Detection

- **Service**: `backend/regime_service.py`
- **API**: `GET /api/v1/regime?symbol=AAPL` (optional symbol)
- **Behavior**: Detects current regime (e.g. `low_vol`, `high_vol`, `trending_up`, `trending_down`, `crisis`) using VIX and optional symbol-specific volatility/trend. Stored in each prediction’s **passport** for audit.

## 3. Prediction Passport & Abstention

- **Storage**: Each `Prediction` row can store `passport` (JSON) and `abstained`, `abstention_reason`.
- **Passport** includes: `model_version`, `model_type`, `feature_set`, `data_start`, `data_end`, `regime`, `vix_level`.
- **Abstention**: When confidence interval width is large relative to price, the API still returns a prediction but sets `abstained: true` and `abstention_reason: "low_confidence"`. Clients can choose to treat this as “no trade” or “hold only”.

## 4. Prediction vs Actual & Quality Report

- **Service**: `backend/quality_report_service.py`
- **APIs**:
  - `GET /api/v1/quality-report?symbol=AAPL&horizon_days=7` – list stored quality metrics (MAPE, direction hit rate, vs naive improvement, abstention count).
  - `POST /api/v1/quality-report/compute?symbol=AAPL&horizon_days=7` (Premium) – compute and persist metrics for the last 90 days.
- **Metrics**: MAPE, MAE, direction hit rate, % improvement vs naive forecast, abstention count.

## 5. Model Degradation Alerts

- **Storage**: `ModelDegradationAlert` table (symbol, metric_name, previous_value, current_value, threshold, triggered_at, acknowledged).
- **Task**: Celery task `check_model_degradation` runs on a schedule (e.g. daily). Compares recent-window MAPE to previous window; if current MAPE exceeds threshold and is >20% worse than previous, an alert is created.
- **API**: `GET /api/v1/admin/degradation-alerts` (Admin only).

## 6. Explainability (Feature Importance)

- **Service**: `backend/explainability_service.py`
- **API**: `GET /api/v1/predict/{prediction_id}/explain`
- **Behavior**: Returns permutation-based feature importance and a short text summary (which features pushed the prediction up/down). Useful for “why did the model say this?”.

## 7. Scenario Predictions & Portfolio VaR

- **Service**: `backend/scenario_service.py`, `backend/var_calculator.py`
- **APIs**:
  - `POST /api/v1/predict/scenario` – body: `symbol`, `scenario` (`base` | `high_vol` | `market_down_5` | `market_up_2`), optional `vol_multiplier`, `market_shock_pct`. Returns scenario-adjusted return and price.
  - `POST /api/v1/portfolio/var` – body: `symbols`, optional `weights`, `confidence`, `volatility_scale`. Returns portfolio VaR (e.g. 95% one-day loss %).
  - **`POST /api/v1/portfolio/var/calculator`** – **VaR Calculator**: Downloads historical returns, computes **Historical VaR** (5th percentile = loss you won't exceed 95% of the time) and **Parametric VaR** (mean + std, normal assumption). Returns both and comparison.

## 8. Pairs Trading Strategy Backtest

- **Service**: `backend/pairs_trading.py`
- **API**: `POST /api/v1/pairs/backtest`
- **Behavior**: For two stocks (e.g., KO & PEP, XOM & CVX): (1) test cointegration via statsmodels, (2) compute spread = price_A - hedge_ratio × price_B, (3) when spread deviates >2 std, long underperformer and short outperformer, (4) backtest for profitability. Returns cointegration stats, backtest metrics (total return, Sharpe, max drawdown), and trades sample.

## 9. Webhooks

- **Storage**: `WebhookSubscription` (user_id, url, secret, events, is_active).
- **APIs**:
  - `POST /api/v1/webhooks` – create subscription (events: `prediction_created`, `prediction_updated`, `alert_triggered`).
  - `GET /api/v1/webhooks` – list user’s webhooks.
  - `DELETE /api/v1/webhooks/{webhook_id}` – deactivate.
- **Delivery**: On `prediction_created`, the app POSTs a JSON payload to each subscriber’s URL with optional `X-Webhook-Signature` (HMAC-SHA256) if `secret` is set.

## 10. Prediction Endpoint Integration

- **Predict** (`POST /api/v1/predict`):
  1. Checks data quality; if below threshold → 503 with `data_quality`.
  2. Runs prediction with regime and passport; sets abstention when confidence is wide.
  3. Saves prediction with `passport`, `abstained`, `abstention_reason`.
  4. Fires webhooks for `prediction_created`.

## Database (main.py vs database.py)

- **main.py** defines its own SQLAlchemy `Base` and tables (including the new ones: `DataQualityScore`, `MarketRegimeSnapshot`, `WebhookSubscription`, `ModelDegradationAlert`, `PredictionQualityMetric`) and adds `passport`, `abstained`, `abstention_reason` to `Prediction`.
- **database.py** (used by Celery/tasks) has the same new models and Prediction fields; ensure migrations or `create_all` are run so both apps see the new columns/tables.

## Quick Test

1. **Data quality**: `GET /api/v1/data-quality/AAPL` (with auth).
2. **Regime**: `GET /api/v1/regime`.
3. **Predict**: `POST /api/v1/predict` – response includes `passport`, `abstained`, `abstention_reason`.
4. **Quality report**: `GET /api/v1/quality-report`; then `POST /api/v1/quality-report/compute?symbol=AAPL&horizon_days=7` (Premium).
5. **Webhook**: `POST /api/v1/webhooks` with `{"url": "https://webhook.site/...", "events": ["prediction_created"]}`.
6. **Scenario**: `POST /api/v1/predict/scenario` with `{"symbol": "AAPL", "scenario": "high_vol"}`.
7. **VaR**: `POST /api/v1/portfolio/var` with `{"symbols": ["AAPL", "MSFT"], "confidence": 0.95}`. **VaR Calculator** (Historical + Parametric): `POST /api/v1/portfolio/var/calculator` with `{"symbols": ["AAPL", "MSFT"], "weights": [0.6, 0.4], "confidence": 0.95, "lookback_days": 252}`.
8. **Explain**: `GET /api/v1/predict/{prediction_id}/explain`.
9. **Degradation**: Run Celery task `check_model_degradation`; then `GET /api/v1/admin/degradation-alerts` (admin).
10. **Pairs trading**: `POST /api/v1/pairs/backtest` with `{"symbol1": "KO", "symbol2": "PEP", "threshold_std": 2, "lookback_days": 504}`.
