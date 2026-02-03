# backend/tasks.py
"""
Celery tasks for background processing
Includes stock monitoring and SMS notification tasks
"""
from celery import Celery
from celery.schedules import crontab
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf

from database import (
    SessionLocal, User, SMSSubscription, SMSNotification, 
    StockMovement, Watchlist, Prediction, ModelDegradationAlert,
)
from sms_service import sms_service
from stock_explainer import stock_explainer

logger = logging.getLogger(__name__)

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "stock_predictor",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
)

# ========== SCHEDULED TASKS ==========

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks"""
    # Monitor stocks every hour during market hours (9:30 AM - 4:00 PM ET)
    sender.add_periodic_task(
        crontab(minute=0),  # Every hour
        monitor_stocks.s(),
        name='monitor-stocks-hourly'
    )
    
    # Reset daily notification counters at midnight UTC
    sender.add_periodic_task(
        crontab(hour=0, minute=0),
        reset_daily_notification_counters.s(),
        name='reset-notification-counters'
    )
    # Check model degradation daily (e.g. 3 AM UTC)
    sender.add_periodic_task(
        crontab(hour=3, minute=0),
        check_model_degradation.s(),
        name='check-model-degradation'
    )

# ========== STOCK MONITORING TASKS ==========

@celery_app.task(name="monitor_stocks", bind=True, max_retries=3)
def monitor_stocks(self):
    """
    Monitor all tracked stocks and send SMS notifications for significant movements
    Runs every hour during market hours
    """
    db = SessionLocal()
    try:
        logger.info("Starting stock monitoring task")
        
        # Get all active SMS subscriptions
        subscriptions = db.query(SMSSubscription).filter(
            SMSSubscription.is_active == True
        ).all()
        
        if not subscriptions:
            logger.info("No active SMS subscriptions found")
            return {"status": "success", "subscriptions_checked": 0}
        
        total_notifications = 0
        total_errors = 0
        
        for subscription in subscriptions:
            try:
                # Check if user has reached daily limit
                if subscription.notifications_sent_today >= subscription.daily_notification_limit:
                    logger.info(f"User {subscription.user_id} has reached daily notification limit")
                    continue
                
                # Get tracked symbols
                tracked_symbols = subscription.tracked_symbols or []
                if not tracked_symbols:
                    # If no symbols specified, check user's watchlists
                    user = db.query(User).filter(User.id == subscription.user_id).first()
                    if user:
                        watchlists = db.query(Watchlist).filter(Watchlist.user_id == user.id).all()
                        for watchlist in watchlists:
                            if watchlist.symbols:
                                tracked_symbols.extend(watchlist.symbols)
                        tracked_symbols = list(set(tracked_symbols))  # Remove duplicates
                
                if not tracked_symbols:
                    continue
                
                # Monitor each symbol
                for symbol in tracked_symbols:
                    try:
                        result = check_and_notify_stock_movement_task.delay(
                            subscription.user_id, symbol, subscription.min_change_threshold
                        ).get(timeout=30)
                        if result:
                            total_notifications += 1
                            subscription.notifications_sent_today += 1
                    except Exception as e:
                        logger.error(f"Error monitoring {symbol} for user {subscription.user_id}: {e}")
                        total_errors += 1
                
                db.commit()
                
            except Exception as e:
                logger.error(f"Error processing subscription for user {subscription.user_id}: {e}")
                total_errors += 1
                db.rollback()
        
        logger.info(f"Stock monitoring completed. Notifications sent: {total_notifications}, Errors: {total_errors}")
        return {
            "status": "success",
            "subscriptions_checked": len(subscriptions),
            "notifications_sent": total_notifications,
            "errors": total_errors
        }
        
    except Exception as e:
        logger.error(f"Error in monitor_stocks task: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        db.close()

@celery_app.task(name="check_and_notify_stock_movement")
def check_and_notify_stock_movement_task(
    user_id: int,
    symbol: str,
    min_change_threshold: float
) -> bool:
    """
    Check if stock has moved significantly and send notification if needed
    
    Args:
        user_id: User ID
        symbol: Stock symbol
        min_change_threshold: Minimum percentage change to trigger notification
    
    Returns:
        bool: True if notification was sent
    """
    db = SessionLocal()
    try:
        # Get subscription
        subscription = db.query(SMSSubscription).filter(
            SMSSubscription.user_id == user_id,
            SMSSubscription.is_active == True
        ).first()
        
        if not subscription:
            logger.warning(f"No active subscription for user {user_id}")
            return False
        
        # Get stock movement explanation
        movement_data = stock_explainer.explain_movement(symbol)
        
        if not movement_data:
            logger.warning(f"Could not get movement data for {symbol}")
            return False
        
        change_pct = abs(movement_data['change_pct'])
        
        # Check if movement exceeds threshold
        if change_pct < min_change_threshold:
            return False
        
        # Check if we already notified about this movement today
        today = datetime.utcnow().date()
        existing_notification = db.query(SMSNotification).filter(
            SMSNotification.user_id == user_id,
            SMSNotification.symbol == symbol,
            SMSNotification.created_at >= datetime.combine(today, datetime.min.time())
        ).first()
        
        if existing_notification:
            # Only notify again if movement is significantly larger
            if change_pct < abs(existing_notification.change_pct) * 1.5:
                return False
        
        # Get user phone number
        if not subscription.phone_number:
            logger.warning(f"No phone number for user {user_id}")
            return False
        
        # Send SMS notification
        success = sms_service.send_stock_alert(
            to_phone=subscription.phone_number,
            symbol=symbol,
            change_pct=movement_data['change_pct'],
            explanation=movement_data['explanation'],
            current_price=movement_data['current_price']
        )
        
        # Record notification
        notification = SMSNotification(
            user_id=user_id,
            symbol=symbol,
            phone_number=subscription.phone_number,
            message=f"{symbol} {movement_data['change_pct']:+.2f}% - {movement_data['explanation']}",
            status="sent" if success else "failed",
            change_pct=movement_data['change_pct'],
            current_price=movement_data['current_price'],
            explanation=movement_data['explanation'],
            sent_at=datetime.utcnow() if success else None
        )
        db.add(notification)
        
        # Record stock movement
        movement = StockMovement(
            symbol=symbol,
            current_price=movement_data['current_price'],
            previous_price=movement_data.get('previous_price', movement_data['current_price']),
            change_pct=movement_data['change_pct'],
            change_amount=movement_data.get('change_amount', 0),
            volume=movement_data.get('volume', 0),
            volume_ratio=movement_data.get('volume_ratio', 1.0),
            explanation=movement_data['explanation']
        )
        db.add(movement)
        
        db.commit()
        
        logger.info(f"Notification sent for {symbol} to user {user_id}")
        return success
        
    except Exception as e:
        logger.error(f"Error in check_and_notify_stock_movement: {e}")
        db.rollback()
        return False
    finally:
        db.close()

@celery_app.task(name="reset_daily_notification_counters")
def reset_daily_notification_counters():
    """Reset daily notification counters for all subscriptions"""
    db = SessionLocal()
    try:
        subscriptions = db.query(SMSSubscription).all()
        today = datetime.utcnow().date()
        
        for subscription in subscriptions:
            # Reset if last reset was not today
            if subscription.last_reset_date.date() < today:
                subscription.notifications_sent_today = 0
                subscription.last_reset_date = datetime.utcnow()
        
        db.commit()
        logger.info(f"Reset notification counters for {len(subscriptions)} subscriptions")
        return {"status": "success", "subscriptions_reset": len(subscriptions)}
        
    except Exception as e:
        logger.error(f"Error resetting notification counters: {e}")
        db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        db.close()

# ========== MANUAL TASKS ==========

@celery_app.task(name="send_test_sms")
def send_test_sms(user_id: int, phone_number: str, message: str = None):
    """Send a test SMS to verify phone number"""
    if not message:
        message = "Test message from Stock Tracker Agent. Your SMS notifications are working!"
    
    success = sms_service.send_sms(phone_number, message)
    
    db = SessionLocal()
    try:
        notification = SMSNotification(
            user_id=user_id,
            phone_number=phone_number,
            message=message,
            status="sent" if success else "failed",
            sent_at=datetime.utcnow() if success else None
        )
        db.add(notification)
        db.commit()
    except Exception as e:
        logger.error(f"Error recording test SMS: {e}")
        db.rollback()
    finally:
        db.close()
    
    return {"success": success, "message": "Test SMS sent" if success else "Failed to send test SMS"}

@celery_app.task(name="monitor_single_stock")
def monitor_single_stock(symbol: str):
    """Monitor a single stock and return movement data"""
    movement_data = stock_explainer.explain_movement(symbol)
    return movement_data


# ========== MODEL DEGRADATION ALERTS ==========

def _mape_for_predictions(predictions_list):
    """Compute MAPE from list of (predicted_price, actual_price)."""
    if not predictions_list:
        return None
    total = 0
    n = 0
    for pred, actual in predictions_list:
        if actual and actual > 0:
            total += abs(pred - actual) / actual * 100
            n += 1
    return total / n if n else None


@celery_app.task(name="check_model_degradation")
def check_model_degradation(symbol: str = None, mape_threshold: float = 15.0):
    """
    Compare recent prediction accuracy to previous window; create ModelDegradationAlert if worse.
    If symbol is None, check all symbols that have recent predictions with actuals.
    """
    db = SessionLocal()
    try:
        from datetime import timedelta
        now = datetime.utcnow()
        recent_end = now - timedelta(days=7)
        recent_start = now - timedelta(days=14)
        previous_start = now - timedelta(days=21)
        q = db.query(Prediction.symbol).filter(
            Prediction.actual_price.isnot(None),
            Prediction.prediction_date >= previous_start,
        ).distinct()
        if symbol:
            q = q.filter(Prediction.symbol == symbol)
        symbols_to_check = [r[0] for r in q.all()]
        created = 0
        for sym in symbols_to_check:
            recent_rows = db.query(Prediction.predicted_price, Prediction.actual_price).filter(
                Prediction.symbol == sym,
                Prediction.actual_price.isnot(None),
                Prediction.actual_price > 0,
                Prediction.prediction_date >= recent_start,
                Prediction.prediction_date < recent_end,
            ).all()
            previous_rows = db.query(Prediction.predicted_price, Prediction.actual_price).filter(
                Prediction.symbol == sym,
                Prediction.actual_price.isnot(None),
                Prediction.actual_price > 0,
                Prediction.prediction_date >= previous_start,
                Prediction.prediction_date < recent_start,
            ).all()
            recent_mape = _mape_for_predictions(recent_rows)
            previous_mape = _mape_for_predictions(previous_rows)
            if recent_mape is None or previous_mape is None:
                continue
            if recent_mape > mape_threshold and recent_mape > previous_mape * 1.2:
                alert = ModelDegradationAlert(
                    symbol=sym,
                    model_type="gru",
                    metric_name="mape",
                    previous_value=previous_mape,
                    current_value=recent_mape,
                    threshold=mape_threshold,
                    details={"window": "7d", "previous_window": "7d"},
                )
                db.add(alert)
                created += 1
        db.commit()
        return {"status": "success", "alerts_created": created, "symbols_checked": len(symbols_to_check)}
    except Exception as e:
        logger.error(f"Model degradation check failed: {e}")
        db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        db.close()
