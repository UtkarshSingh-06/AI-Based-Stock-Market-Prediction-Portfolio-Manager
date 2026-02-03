# backend/database.py
"""
Database configuration, models, and migrations
PostgreSQL with SQLAlchemy ORM
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import os

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://stockuser:securepassword@localhost:5432/stockprediction"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ========== DATABASE MODELS ==========

class User(Base):
    """User accounts with authentication and subscription"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    first_name = Column(String(100))
    last_name = Column(String(100))
    phone = Column(String(20))
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    # Subscription
    subscription_tier = Column(String(20), default="free")  # free, premium, enterprise
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    
    # API access
    api_key = Column(String(64), unique=True, index=True)
    api_calls_count = Column(Integer, default=0)
    api_calls_limit = Column(Integer, default=100)
    api_calls_reset_date = Column(DateTime, default=datetime.utcnow)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    last_failed_login = Column(DateTime, nullable=True)
    account_locked_until = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    watchlists = relationship("Watchlist", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    alerts = relationship("PriceAlert", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    sms_subscription = relationship("SMSSubscription", back_populates="user", uselist=False)
    sms_notifications = relationship("SMSNotification", back_populates="user")

class Prediction(Base):
    """Stock price predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Stock info
    symbol = Column(String(10), index=True, nullable=False)
    exchange = Column(String(20))
    
    # Prediction details
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    target_date = Column(DateTime, index=True)
    horizon_days = Column(Integer)  # 1, 7, 30, etc.
    
    # Predictions
    predicted_price = Column(Float, nullable=False)
    predicted_return = Column(Float)
    actual_price = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)
    
    # Uncertainty
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    confidence_level = Column(Float, default=0.95)
    prediction_std = Column(Float)
    
    # Model info
    model_type = Column(String(20))  # gru, lstm, transformer, ensemble
    model_version = Column(String(50))
    features_used = Column(JSON)
    
    # Performance tracking
    accuracy_score = Column(Float, nullable=True)
    absolute_error = Column(Float, nullable=True)
    percentage_error = Column(Float, nullable=True)
    
    # Additional metadata
    metadata = Column(JSON)
    notes = Column(Text)
    
    # Prediction passport (compliance / explainability)
    passport = Column(JSON)  # model_version, feature_set, data_start, data_end, regime
    # Abstention: when uncertainty or data quality is poor
    abstained = Column(Boolean, default=False)
    abstention_reason = Column(String(100), nullable=True)  # low_confidence, low_data_quality
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="predictions")

class ModelMetrics(Base):
    """Model performance metrics and training history"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    model_type = Column(String(20), index=True)
    model_version = Column(String(50))
    
    # Training metrics
    train_rmse = Column(Float)
    train_mape = Column(Float)
    train_mae = Column(Float)
    train_r2 = Column(Float)
    
    # Validation metrics
    val_rmse = Column(Float)
    val_mape = Column(Float)
    val_mae = Column(Float)
    val_r2 = Column(Float)
    
    # Trading metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    calmar_ratio = Column(Float)
    win_rate = Column(Float)
    
    # Training details
    trained_at = Column(DateTime, default=datetime.utcnow, index=True)
    training_duration_seconds = Column(Float)
    num_epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    
    # Model architecture
    architecture = Column(JSON)  # Store model config
    hyperparameters = Column(JSON)
    
    # Data info
    data_start_date = Column(DateTime)
    data_end_date = Column(DateTime)
    features_count = Column(Integer)
    
    # Status
    status = Column(String(20), default="active")  # active, deprecated, archived
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Watchlist(Base):
    """User watchlists for tracking stocks"""
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    name = Column(String(100), nullable=False)
    description = Column(Text)
    symbols = Column(JSON)  # List of stock symbols
    color = Column(String(7))  # Hex color for UI
    
    is_default = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="watchlists")

class Portfolio(Base):
    """User portfolios and positions"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Portfolio value
    initial_capital = Column(Float, default=10000.0)
    current_value = Column(Float)
    cash_balance = Column(Float)
    
    # Performance
    total_return = Column(Float)
    total_return_pct = Column(Float)
    daily_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Settings
    currency = Column(String(3), default="USD")
    is_paper_trading = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")
    trades = relationship("Trade", back_populates="portfolio")

class Position(Base):
    """Current positions in portfolios"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True)
    
    symbol = Column(String(10), index=True, nullable=False)
    shares = Column(Float, nullable=False)
    
    # Entry details
    avg_entry_price = Column(Float)
    total_cost = Column(Float)
    entry_date = Column(DateTime)
    
    # Current value
    current_price = Column(Float)
    current_value = Column(Float)
    
    # Performance
    unrealized_pnl = Column(Float)
    unrealized_pnl_pct = Column(Float)
    realized_pnl = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="positions")

class Trade(Base):
    """Trade history"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), index=True)
    
    symbol = Column(String(10), index=True, nullable=False)
    trade_type = Column(String(10), nullable=False)  # BUY, SELL
    
    # Trade details
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float)
    
    # Costs
    commission = Column(Float, default=0)
    slippage = Column(Float, default=0)
    
    # Execution
    order_type = Column(String(20))  # market, limit, stop
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Strategy
    strategy = Column(String(50))  # threshold, ml_confidence, etc.
    signal_strength = Column(Float)
    
    # P&L (for closing trades)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    
    # Metadata
    notes = Column(Text)
    metadata = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="trades")

class PriceAlert(Base):
    """Price alerts for users"""
    __tablename__ = "price_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    symbol = Column(String(10), index=True, nullable=False)
    alert_type = Column(String(20), nullable=False)  # price_above, price_below, prediction_change
    
    # Alert conditions
    target_price = Column(Float)
    condition = Column(String(50))
    
    # Status
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime, nullable=True)
    times_triggered = Column(Integer, default=0)
    
    # Notification
    notification_method = Column(String(20), default="email")  # email, sms, push
    message = Column(Text)
    
    # Auto-disable
    expires_at = Column(DateTime, nullable=True)
    trigger_once = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="alerts")

class MarketData(Base):
    """Cached market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    
    # OHLCV
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    # Adjusted
    adj_close = Column(Float)
    
    # Technical indicators (cached)
    sma_20 = Column(Float)
    ema_50 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class NewsArticle(Base):
    """Cached news articles for sentiment analysis"""
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=True)
    
    title = Column(String(500))
    content = Column(Text)
    source = Column(String(100))
    author = Column(String(200))
    url = Column(Text, unique=True)
    
    # Sentiment
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))  # positive, negative, neutral
    
    # Metadata
    published_at = Column(DateTime, index=True)
    category = Column(String(50))
    tags = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    """Comprehensive audit trail"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    
    # Action details
    action = Column(String(100), index=True, nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(Integer)
    
    # Request details
    ip_address = Column(String(45))
    user_agent = Column(Text)
    endpoint = Column(String(200))
    method = Column(String(10))
    
    # Response
    status_code = Column(Integer)
    response_time_ms = Column(Float)
    
    # Additional context
    details = Column(JSON)
    error_message = Column(Text)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    user = relationship("User", back_populates="audit_logs")

class SystemMetrics(Base):
    """System performance and health metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    metric_type = Column(String(50), index=True)  # api_latency, model_accuracy, cache_hit_rate
    metric_name = Column(String(100))
    value = Column(Float)
    
    # Context
    dimensions = Column(JSON)  # Additional dimensions (region, model_type, etc.)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class APIUsage(Base):
    """Track API usage for billing and analytics"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    endpoint = Column(String(200), index=True)
    method = Column(String(10))
    
    # Usage
    request_count = Column(Integer, default=1)
    tokens_used = Column(Integer)
    compute_time_ms = Column(Float)
    
    # Cost estimation
    estimated_cost = Column(Float)
    
    date = Column(DateTime, default=datetime.utcnow, index=True)

class StockMovement(Base):
    """Track stock price movements for SMS notifications"""
    __tablename__ = "stock_movements"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    
    # Price data
    current_price = Column(Float, nullable=False)
    previous_price = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    change_amount = Column(Float)
    
    # Volume
    volume = Column(Float)
    volume_ratio = Column(Float)
    
    # Explanation
    explanation = Column(Text)
    
    # Timestamp
    movement_time = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Metadata
    metadata = Column(JSON)

class SMSNotification(Base):
    """Track SMS notifications sent to users"""
    __tablename__ = "sms_notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Notification details
    symbol = Column(String(10), index=True)
    phone_number = Column(String(20))
    message = Column(Text)
    
    # Status
    status = Column(String(20), default="pending")  # pending, sent, failed
    twilio_sid = Column(String(100))  # Twilio message SID
    error_message = Column(Text)
    
    # Movement data
    change_pct = Column(Float)
    current_price = Column(Float)
    explanation = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    sent_at = Column(DateTime, nullable=True)
    
    # Metadata
    metadata = Column(JSON)

class SMSSubscription(Base):
    """User subscriptions for SMS stock alerts"""
    __tablename__ = "sms_subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, unique=True)
    
    # Phone number
    phone_number = Column(String(20), nullable=False)
    phone_verified = Column(Boolean, default=False)
    
    # Subscription settings
    is_active = Column(Boolean, default=True)
    min_change_threshold = Column(Float, default=1.0)  # Minimum % change to trigger alert
    notification_frequency = Column(String(20), default="daily")  # daily, hourly, realtime
    
    # Tracked symbols
    tracked_symbols = Column(JSON)  # List of stock symbols to track
    
    # Notification limits
    daily_notification_limit = Column(Integer, default=10)
    notifications_sent_today = Column(Integer, default=0)
    last_reset_date = Column(DateTime, default=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sms_subscription")

# ========== INDUSTRY FEATURES: DATA QUALITY, REGIME, WEBHOOKS, DEGRADATION ==========

class DataQualityScore(Base):
    """Per-symbol data quality score for gating predictions"""
    __tablename__ = "data_quality_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    
    score = Column(Float, nullable=False)  # 0-100
    completeness_pct = Column(Float)  # % of expected bars present
    staleness_hours = Column(Float)  # hours since last data point
    has_gaps = Column(Boolean, default=False)
    gap_count = Column(Integer, default=0)
    outlier_count = Column(Integer, default=0)
    
    details = Column(JSON)  # raw checks
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)

class MarketRegimeSnapshot(Base):
    """Market regime (volatility/trend) for a date or symbol"""
    __tablename__ = "market_regime_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=True)  # null = broad market
    snapshot_date = Column(DateTime, index=True, nullable=False)
    
    regime = Column(String(30), nullable=False)  # low_vol, high_vol, trending_up, trending_down, crisis
    vix_level = Column(Float, nullable=True)
    volatility_20d = Column(Float, nullable=True)
    trend_signal = Column(Float, nullable=True)  # e.g. SMA crossover
    metadata = Column(JSON)
    computed_at = Column(DateTime, default=datetime.utcnow)

class WebhookSubscription(Base):
    """User webhooks for prediction/alert events"""
    __tablename__ = "webhook_subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    
    url = Column(Text, nullable=False)
    secret = Column(String(64))  # for HMAC signing
    events = Column(JSON, nullable=False)  # ["prediction_created", "prediction_updated", "alert_triggered"]
    is_active = Column(Boolean, default=True)
    
    last_triggered_at = Column(DateTime, nullable=True)
    last_status_code = Column(Integer, nullable=True)
    failure_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelDegradationAlert(Base):
    """Alerts when model accuracy drops below threshold"""
    __tablename__ = "model_degradation_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    model_type = Column(String(20), nullable=True)
    
    metric_name = Column(String(50), nullable=False)  # mape, rmse, direction_hit_rate
    previous_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    details = Column(JSON)

class PredictionQualityMetric(Base):
    """Aggregated prediction vs actual and vs baseline (for Quality Report)"""
    __tablename__ = "prediction_quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    horizon_days = Column(Integer, nullable=False)  # 1, 7, 30
    
    period_start = Column(DateTime, index=True, nullable=False)
    period_end = Column(DateTime, index=True, nullable=False)
    
    sample_count = Column(Integer, nullable=False)
    mape = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    direction_hit_rate = Column(Float, nullable=True)  # % correct direction
    vs_naive_improvement = Column(Float, nullable=True)  # % improvement over naive forecast
    vs_buy_hold_note = Column(Text, nullable=True)  # e.g. "outperformed in 60% of windows"
    
    abstention_count = Column(Integer, default=0)
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)

# ========== INDEXES ==========
# Create composite indexes for better query performance
from sqlalchemy import Index

Index('idx_predictions_user_date', Prediction.user_id, Prediction.prediction_date)
Index('idx_predictions_symbol_date', Prediction.symbol, Prediction.prediction_date)
Index('idx_market_data_symbol_date', MarketData.symbol, MarketData.date)
Index('idx_trades_portfolio_executed', Trade.portfolio_id, Trade.executed_at)
Index('idx_audit_logs_user_timestamp', AuditLog.user_id, AuditLog.timestamp)
Index('idx_data_quality_symbol_computed', DataQualityScore.symbol, DataQualityScore.computed_at)
Index('idx_regime_symbol_date', MarketRegimeSnapshot.symbol, MarketRegimeSnapshot.snapshot_date)
Index('idx_quality_metrics_symbol_horizon', PredictionQualityMetric.symbol, PredictionQualityMetric.horizon_days)
Index('idx_degradation_symbol_triggered', ModelDegradationAlert.symbol, ModelDegradationAlert.triggered_at)

# ========== DATABASE INITIALIZATION ==========
def init_db():
    """Initialize database and create all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========== MIGRATIONS SCRIPT ==========
def create_migration_script():
    """Generate Alembic migration script template"""
    return """
# Alembic migration template
# Run: alembic init alembic
# Then: alembic revision --autogenerate -m "Initial migration"
# Finally: alembic upgrade head

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Add new columns
    op.add_column('users', sa.Column('two_factor_enabled', sa.Boolean(), default=False))
    op.add_column('users', sa.Column('two_factor_secret', sa.String(64)))
    
    # Create indexes
    op.create_index('idx_user_email', 'users', ['email'])
    
def downgrade():
    op.drop_column('users', 'two_factor_enabled')
    op.drop_column('users', 'two_factor_secret')
    op.drop_index('idx_user_email')
"""

# ========== UTILITY FUNCTIONS ==========
def reset_api_calls_for_user(user_id: int, db: SessionLocal):
    """Reset API call counter for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.api_calls_count = 0
        user.api_calls_reset_date = datetime.utcnow()
        db.commit()

def calculate_portfolio_value(portfolio_id: int, db: SessionLocal) -> float:
    """Calculate current portfolio value"""
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        return 0.0
    
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    total_value = portfolio.cash_balance or 0.0
    
    for pos in positions:
        if pos.current_price and pos.shares:
            total_value += pos.current_price * pos.shares
    
    return total_value

def cleanup_old_audit_logs(days: int = 90, db: SessionLocal = None):
    """Clean up audit logs older than specified days"""
    if db is None:
        db = SessionLocal()
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    deleted = db.query(AuditLog).filter(AuditLog.timestamp < cutoff_date).delete()
    db.commit()
    return deleted

# ========== SEED DATA FOR DEVELOPMENT ==========
def seed_database():
    """Seed database with test data"""
    from passlib.context import CryptContext
    import secrets
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    
    # Create admin user
    admin = User(
        email="admin@stockpred.com",
        username="admin",
        hashed_password=pwd_context.hash("Admin@123"),
        first_name="System",
        last_name="Administrator",
        is_active=True,
        is_admin=True,
        is_verified=True,
        subscription_tier="enterprise",
        api_key=secrets.token_urlsafe(32),
        api_calls_limit=10000
    )
    db.add(admin)
    
    # Create test user
    test_user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=pwd_context.hash("Test@123"),
        first_name="Test",
        last_name="User",
        is_active=True,
        is_verified=True,
        subscription_tier="premium",
        api_key=secrets.token_urlsafe(32)
    )
    db.add(test_user)
    
    db.commit()
    print("Seed data created successfully!")
    print(f"Admin API Key: {admin.api_key}")
    print(f"Test User API Key: {test_user.api_key}")
    db.close()

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    
    print("\nSeeding database with test data...")
    seed_database()
    
    print("\nDatabase setup complete!")
    print("PostgreSQL connection string:", DATABASE_URL)