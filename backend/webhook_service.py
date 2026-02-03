"""
Webhook delivery for prediction and alert events.
"""
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


def sign_payload(payload: bytes, secret: str) -> str:
    """HMAC-SHA256 signature of payload for X-Webhook-Signature header."""
    if not secret:
        return ""
    return "sha256=" + hmac.new(
        secret.encode("utf-8") if isinstance(secret, str) else secret,
        payload,
        hashlib.sha256,
    ).hexdigest()


def deliver_webhook(
    url: str,
    payload: Dict[str, Any],
    secret: Optional[str] = None,
    timeout: int = 10,
) -> tuple[int, Optional[str]]:
    """
    POST JSON payload to url. Optionally sign with secret.
    Returns (status_code, error_message). error_message is None on success.
    """
    try:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "StockPrediction-Webhook/1.0",
            "X-Webhook-Delivery": datetime.utcnow().isoformat() + "Z",
        }
        if secret:
            headers["X-Webhook-Signature"] = sign_payload(body, secret)
        req = Request(url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), None
    except HTTPError as e:
        return e.code, str(e)
    except URLError as e:
        return 0, str(e.reason) if getattr(e, "reason", None) else str(e)
    except Exception as e:
        return 0, str(e)


def notify_webhooks(
    db_session,
    event: str,
    payload: Dict[str, Any],
    user_id: Optional[int] = None,
) -> None:
    """
    Find active webhooks subscribed to event and deliver. Update last_triggered_at, failure_count.
    """
    from database import WebhookSubscription
    subs = db_session.query(WebhookSubscription).filter(
        WebhookSubscription.is_active == True,
    )
    if user_id is not None:
        subs = subs.filter(WebhookSubscription.user_id == user_id)
    subs = subs.all()
    for sub in subs:
        events = sub.events or []
        if event not in events:
            continue
        payload["event"] = event
        payload["user_id"] = user_id
        code, err = deliver_webhook(sub.url, payload, sub.secret)
        sub.last_triggered_at = datetime.utcnow()
        sub.last_status_code = code
        if code is None or code >= 400:
            sub.failure_count = (sub.failure_count or 0) + 1
        db_session.commit()
