# backend/main.py
"""
Production-grade FastAPI backend with PostgreSQL, Redis, JWT auth, and security
"""
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, EmailStr, validator, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import redis
import logging
from typing import Optional, List, Dict
import os
import secrets
import hashlib
import asyncio
from contextlib import asynccontextmanager

# ========== CONFIGURATION ==========
class Settings:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/stockdb")
    
    # Redis Cache
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL = 300  # 5 minutes
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    # API Keys
    API_KEY_LENGTH = 32
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = "60/minute"
    RATE_LIMIT_PER_HOUR = "1000/hour"
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    
    # Model paths
    MODELS_DIR = os.getenv("MODELS_DIR", "./models")
    
settings = Settings()

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== DATABASE SETUP ==========
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ========== REDIS SETUP ==========
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# ========== SECURITY ==========
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ========== DATABASE MODELS ==========
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    subscription_tier = Column(String, default="free")  # free, premium, enterprise
    api_key = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    api_calls_count = Column(Integer, default=0)
    api_calls_limit = Column(Integer, default=100)  # per day

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    symbol = Column(String, index=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_version = Column(String)
    accuracy_score = Column(Float, nullable=True)
    metadata = Column(JSON)
    passport = Column(JSON, nullable=True)  # model_version, feature_set, data_start, data_end, regime
    abstained = Column(Boolean, default=False)
    abstention_reason = Column(String(100), nullable=True)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    model_version = Column(String)
    rmse = Column(Float)
    mape = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)
    num_epochs = Column(Integer)
    training_samples = Column(Integer)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    action = Column(String)
    resource = Column(String)
    ip_address = Column(String)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)


class DataQualityScore(Base):
    __tablename__ = "data_quality_scores"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    score = Column(Float, nullable=False)
    completeness_pct = Column(Float)
    staleness_hours = Column(Float)
    has_gaps = Column(Boolean, default=False)
    gap_count = Column(Integer, default=0)
    outlier_count = Column(Integer, default=0)
    details = Column(JSON)
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)


class MarketRegimeSnapshot(Base):
    __tablename__ = "market_regime_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=True)
    snapshot_date = Column(DateTime, index=True, nullable=False)
    regime = Column(String(30), nullable=False)
    vix_level = Column(Float, nullable=True)
    volatility_20d = Column(Float, nullable=True)
    trend_signal = Column(Float, nullable=True)
    metadata = Column(JSON)
    computed_at = Column(DateTime, default=datetime.utcnow)


class WebhookSubscription(Base):
    __tablename__ = "webhook_subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    url = Column(String(2000), nullable=False)
    secret = Column(String(64), nullable=True)
    events = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True)
    last_status_code = Column(Integer, nullable=True)
    failure_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ModelDegradationAlert(Base):
    __tablename__ = "model_degradation_alerts"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    model_type = Column(String(20), nullable=True)
    metric_name = Column(String(50), nullable=False)
    previous_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    details = Column(JSON)


class PredictionQualityMetric(Base):
    __tablename__ = "prediction_quality_metrics"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    horizon_days = Column(Integer, nullable=False)
    period_start = Column(DateTime, index=True, nullable=False)
    period_end = Column(DateTime, index=True, nullable=False)
    sample_count = Column(Integer, nullable=False)
    mape = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    direction_hit_rate = Column(Float, nullable=True)
    vs_naive_improvement = Column(Float, nullable=True)
    vs_buy_hold_note = Column(String(500), nullable=True)
    abstention_count = Column(Integer, default=0)
    computed_at = Column(DateTime, default=datetime.utcnow, index=True)


# Create tables
Base.metadata.create_all(bind=engine)

# ========== PYDANTIC MODELS ==========
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    start_date: str
    end_date: str
    include_uncertainty: bool = True
    model_type: str = "gru"  # gru, lstm, transformer

class PredictionResponse(BaseModel):
    symbol: str
    predicted_prices: List[float]
    actual_prices: List[float]
    confidence_lower: Optional[List[float]]
    confidence_upper: Optional[List[float]]
    metrics: Dict
    prediction_id: int

class TrainingRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=8, le=256)
    use_attention: bool = False

# ========== DEPENDENCY INJECTION ==========
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Validate JWT token and return current user"""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None or not user.is_active:
        raise credentials_exception
    
    return user

async def verify_api_key(api_key: str, db: Session = Depends(get_db)) -> User:
    """Verify API key for programmatic access"""
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check rate limits
    if user.api_calls_count >= user.api_calls_limit:
        raise HTTPException(status_code=429, detail="API rate limit exceeded")
    
    user.api_calls_count += 1
    db.commit()
    
    return user

def check_subscription(required_tier: str):
    """Decorator to check user subscription tier"""
    async def subscription_checker(current_user: User = Depends(get_current_user)):
        tiers = {"free": 0, "premium": 1, "enterprise": 2}
        if tiers.get(current_user.subscription_tier, 0) < tiers.get(required_tier, 0):
            raise HTTPException(
                status_code=403,
                detail=f"This feature requires {required_tier} subscription"
            )
        return current_user
    return subscription_checker

# ========== UTILITY FUNCTIONS ==========
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def generate_api_key() -> str:
    """Generate secure API key"""
    return secrets.token_urlsafe(settings.API_KEY_LENGTH)

async def cache_get(key: str) -> Optional[str]:
    """Get value from Redis cache"""
    try:
        return redis_client.get(key)
    except Exception as e:
        logger.error(f"Redis GET error: {e}")
        return None

async def cache_set(key: str, value: str, ttl: int = settings.CACHE_TTL):
    """Set value in Redis cache"""
    try:
        redis_client.setex(key, ttl, value)
    except Exception as e:
        logger.error(f"Redis SET error: {e}")

def log_audit(db: Session, user_id: Optional[int], action: str, resource: str, 
              ip: str, details: dict = None):
    """Log action to audit trail"""
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        ip_address=ip,
        details=details or {}
    )
    db.add(log)
    db.commit()

# ========== FASTAPI APP ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    yield
    # Shutdown
    logger.info("Shutting down application...")
    redis_client.close()

app = FastAPI(
    title="Stock Prediction API",
    description="Advanced GRU-based stock prediction with ML",
    version="2.0.0",
    lifespan=lifespan
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure in production

# ========== AUTHENTICATION ENDPOINTS ==========
@app.post("/api/v1/auth/register", response_model=Token, status_code=201)
@limiter.limit("5/minute")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    # Check if user exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create user
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password),
        api_key=generate_api_key()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Generate tokens
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    log_audit(db, db_user.id, "REGISTER", "User", "unknown", {"email": user.email})
    logger.info(f"New user registered: {user.email}")
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/api/v1/auth/login", response_model=Token)
@limiter.limit("10/minute")
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = db.query(User).filter(User.email == user_login.email).first()
    if not user or not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Generate tokens
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    log_audit(db, user.id, "LOGIN", "User", "unknown")
    logger.info(f"User logged in: {user.email}")
    
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "subscription_tier": current_user.subscription_tier,
        "api_key": current_user.api_key,
        "api_calls_remaining": current_user.api_calls_limit - current_user.api_calls_count
    }

@app.post("/api/v1/auth/regenerate-api-key")
async def regenerate_api_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Regenerate API key"""
    current_user.api_key = generate_api_key()
    db.commit()
    
    log_audit(db, current_user.id, "REGENERATE_API_KEY", "User", "unknown")
    
    return {"api_key": current_user.api_key}

# ========== PREDICTION ENDPOINTS ==========
def _predict_with_passport_and_abstention(
    symbol: str, start: str, end: str, include_uncertainty: bool, model_type: str
):
    """Run prediction and return actual, predicted, confidence, passport, abstained, reason."""
    from predictor import predict, compute_rmse, compute_mape
    import numpy as np
    regime_info = {}
    try:
        from regime_service import detect_regime
        regime_info = detect_regime(symbol=symbol)
    except Exception:
        pass
    actual, predicted = predict(
        symbol, start=start, end=end, use_attention=False
    )
    confidence_lower = (predicted * 0.95).tolist() if include_uncertainty else None
    confidence_upper = (predicted * 1.05).tolist() if include_uncertainty else None
    if include_uncertainty and len(predicted) > 0:
        try:
            import os
            model_path = os.path.join(settings.MODELS_DIR or "models", f"{symbol}_gru_model.pth")
            if os.path.exists(model_path):
                pred_lower = np.percentile(predicted, 2.5)
                pred_upper = np.percentile(predicted, 97.5)
                confidence_lower = (np.array(predicted) * 0.98 - (pred_upper - pred_lower) * 0.5).tolist()
                confidence_upper = (np.array(predicted) * 1.02 + (pred_upper - pred_lower) * 0.5).tolist()
        except Exception:
            pass
    pred_last = float(predicted[-1])
    conf_width = (confidence_upper[-1] - confidence_lower[-1]) if confidence_lower and confidence_upper else 0
    abstained = bool(conf_width > pred_last * 0.15)
    abstention_reason = "low_confidence" if abstained else None
    passport = {
        "model_version": "2.0",
        "model_type": model_type,
        "feature_set": ["Close", "MA_*", "EMA_*", "RSI", "MACD", "BB_*", "STD_*", "RET_*"],
        "data_start": start,
        "data_end": end,
        "regime": regime_info.get("regime", "unknown"),
        "vix_level": regime_info.get("vix_level"),
    }
    rmse = compute_rmse(actual, predicted)
    mape = compute_mape(actual, predicted)
    return {
        "actual": actual, "predicted": predicted,
        "confidence_lower": confidence_lower, "confidence_upper": confidence_upper,
        "passport": passport, "abstained": abstained, "abstention_reason": abstention_reason,
        "rmse": rmse, "mape": mape, "regime": regime_info.get("regime", "unknown"),
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def predict_stock(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get stock price prediction with uncertainty, data quality gating, and prediction passport."""
    import json
    cache_key = f"prediction:{request.symbol}:{request.start_date}:{request.end_date}"
    cached = await cache_get(cache_key)
    if cached:
        logger.info(f"Cache hit for {request.symbol}")
        return json.loads(cached)

    try:
        from data_quality_service import is_prediction_allowed
        allowed, quality_result = is_prediction_allowed(request.symbol)
        if not allowed:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Prediction not available: data quality below threshold",
                    "data_quality": quality_result,
                },
            )

        result = _predict_with_passport_and_abstention(
            request.symbol,
            request.start_date,
            request.end_date,
            request.include_uncertainty,
            request.model_type,
        )
        actual = result["actual"]
        predicted = result["predicted"]
        confidence_lower = result["confidence_lower"]
        confidence_upper = result["confidence_upper"]
        passport = result["passport"]
        abstained = result["abstained"]
        abstention_reason = result["abstention_reason"]
        rmse, mape = result["rmse"], result["mape"]

        db_prediction = Prediction(
            user_id=current_user.id,
            symbol=request.symbol,
            predicted_price=float(predicted[-1]),
            confidence_lower=float(confidence_lower[-1]) if confidence_lower else None,
            confidence_upper=float(confidence_upper[-1]) if confidence_upper else None,
            model_version="2.0",
            metadata={"rmse": rmse, "mape": mape, "regime": result.get("regime")},
            passport=passport,
            abstained=abstained,
            abstention_reason=abstention_reason,
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        response = {
            "symbol": request.symbol,
            "predicted_prices": predicted.tolist(),
            "actual_prices": actual.tolist(),
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "metrics": {"rmse": rmse, "mape": mape},
            "prediction_id": db_prediction.id,
            "passport": passport,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
        }
        await cache_set(cache_key, json.dumps(response))
        log_audit(db, current_user.id, "PREDICT", request.symbol, "unknown")

        try:
            from webhook_service import deliver_webhook
            for sub in db.query(WebhookSubscription).filter(
                WebhookSubscription.user_id == current_user.id,
                WebhookSubscription.is_active == True,
            ).all():
                if "prediction_created" in (sub.events or []):
                    payload = {"event": "prediction_created", "prediction_id": db_prediction.id, "symbol": request.symbol, "predicted_price": float(predicted[-1]), "abstained": abstained}
                    code, err = deliver_webhook(sub.url, payload, sub.secret)
                    sub.last_triggered_at = datetime.utcnow()
                    sub.last_status_code = code
                    if code >= 400 or code is None:
                        sub.failure_count = (sub.failure_count or 0) + 1
                    db.commit()
        except Exception as e:
            logger.debug(f"Webhook notify failed: {e}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/v1/train")
@limiter.limit("5/hour")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(check_subscription("premium")),
    db: Session = Depends(get_db)
):
    """Train model (Premium feature)"""
    try:
        from predictor import train as train_model_fn
        
        # Train in background
        def train_task():
            model_path, scaler_path = train_model_fn(
                request.symbol,
                start=request.start_date,
                end=request.end_date,
                epochs=request.epochs,
                batch_size=request.batch_size,
                use_attention=request.use_attention
            )
            logger.info(f"Model trained: {model_path}")
        
        background_tasks.add_task(train_task)
        
        log_audit(db, current_user.id, "TRAIN", request.symbol, "unknown")
        
        return {
            "message": f"Training started for {request.symbol}",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# ========== ANALYTICS ENDPOINTS ==========
@app.get("/api/v1/history")
async def get_prediction_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's prediction history"""
    predictions = db.query(Prediction)\
        .filter(Prediction.user_id == current_user.id)\
        .order_by(Prediction.prediction_date.desc())\
        .limit(limit)\
        .all()
    
    return [
        {
            "id": p.id,
            "symbol": p.symbol,
            "predicted_price": p.predicted_price,
            "date": p.prediction_date.isoformat()
        }
        for p in predictions
    ]

@app.get("/api/v1/metrics/{symbol}")
async def get_model_metrics(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get model performance metrics"""
    metrics = db.query(ModelMetrics)\
        .filter(ModelMetrics.symbol == symbol)\
        .order_by(ModelMetrics.trained_at.desc())\
        .first()
    
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    return {
        "symbol": metrics.symbol,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "trained_at": metrics.trained_at.isoformat()
    }

# ========== ADMIN ENDPOINTS ==========
@app.get("/api/v1/admin/users")
async def list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all users (Admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "subscription": u.subscription_tier,
            "is_active": u.is_active
        }
        for u in users
    ]

@app.get("/api/v1/admin/audit-logs")
async def get_audit_logs(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get audit logs (Admin only)"""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    logs = db.query(AuditLog)\
        .order_by(AuditLog.timestamp.desc())\
        .limit(limit)\
        .all()
    
    return [
        {
            "action": log.action,
            "resource": log.resource,
            "user_id": log.user_id,
            "timestamp": log.timestamp.isoformat()
        }
        for log in logs
    ]

# ========== SMS NOTIFICATION ENDPOINTS ==========
# Import SMS-related models and services
try:
    from database import SMSSubscription, SMSNotification, StockMovement
    from sms_service import sms_service
    from stock_explainer import stock_explainer
    from tasks import send_test_sms, monitor_single_stock
    SMS_ENABLED = True
except ImportError as e:
    logger.warning(f"SMS features not available: {e}")
    SMS_ENABLED = False

class SMSSubscriptionCreate(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)
    min_change_threshold: float = Field(default=1.0, ge=0.1, le=50.0)
    notification_frequency: str = Field(default="daily", pattern="^(daily|hourly|realtime)$")
    tracked_symbols: Optional[List[str]] = Field(default=None)

class SMSSubscriptionUpdate(BaseModel):
    phone_number: Optional[str] = Field(None, min_length=10, max_length=20)
    min_change_threshold: Optional[float] = Field(None, ge=0.1, le=50.0)
    notification_frequency: Optional[str] = Field(None, pattern="^(daily|hourly|realtime)$")
    tracked_symbols: Optional[List[str]] = None
    is_active: Optional[bool] = None

@app.post("/api/v1/sms/subscribe")
@limiter.limit("5/minute")
async def subscribe_sms(
    subscription: SMSSubscriptionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Subscribe to SMS stock movement notifications"""
    if not SMS_ENABLED:
        raise HTTPException(status_code=503, detail="SMS service not available")
    
    # Verify phone number format
    if not sms_service.verify_phone_number(subscription.phone_number):
        raise HTTPException(status_code=400, detail="Invalid phone number format")
    
    # Check if user already has a subscription
    existing = db.query(SMSSubscription).filter(SMSSubscription.user_id == current_user.id).first()
    
    if existing:
        # Update existing subscription
        existing.phone_number = subscription.phone_number
        existing.min_change_threshold = subscription.min_change_threshold
        existing.notification_frequency = subscription.notification_frequency
        existing.tracked_symbols = subscription.tracked_symbols or []
        existing.is_active = True
        db.commit()
        db.refresh(existing)
        
        return {
            "message": "SMS subscription updated",
            "subscription": {
                "id": existing.id,
                "phone_number": existing.phone_number,
                "min_change_threshold": existing.min_change_threshold,
                "notification_frequency": existing.notification_frequency,
                "tracked_symbols": existing.tracked_symbols,
                "is_active": existing.is_active
            }
        }
    else:
        # Create new subscription
        new_subscription = SMSSubscription(
            user_id=current_user.id,
            phone_number=subscription.phone_number,
            min_change_threshold=subscription.min_change_threshold,
            notification_frequency=subscription.notification_frequency,
            tracked_symbols=subscription.tracked_symbols or []
        )
        db.add(new_subscription)
        db.commit()
        db.refresh(new_subscription)
        
        # Send test SMS
        send_test_sms.delay(current_user.id, subscription.phone_number)
        
        return {
            "message": "SMS subscription created. Test message sent.",
            "subscription": {
                "id": new_subscription.id,
                "phone_number": new_subscription.phone_number,
                "min_change_threshold": new_subscription.min_change_threshold,
                "notification_frequency": new_subscription.notification_frequency,
                "tracked_symbols": new_subscription.tracked_symbols,
                "is_active": new_subscription.is_active
            }
        }

@app.get("/api/v1/sms/subscription")
async def get_sms_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's SMS subscription"""
    subscription = db.query(SMSSubscription).filter(SMSSubscription.user_id == current_user.id).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No SMS subscription found")
    
    return {
        "id": subscription.id,
        "phone_number": subscription.phone_number,
        "phone_verified": subscription.phone_verified,
        "min_change_threshold": subscription.min_change_threshold,
        "notification_frequency": subscription.notification_frequency,
        "tracked_symbols": subscription.tracked_symbols or [],
        "is_active": subscription.is_active,
        "notifications_sent_today": subscription.notifications_sent_today,
        "daily_notification_limit": subscription.daily_notification_limit
    }

@app.put("/api/v1/sms/subscription")
@limiter.limit("10/minute")
async def update_sms_subscription(
    subscription_update: SMSSubscriptionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update SMS subscription settings"""
    subscription = db.query(SMSSubscription).filter(SMSSubscription.user_id == current_user.id).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No SMS subscription found")
    
    if subscription_update.phone_number:
        if not sms_service.verify_phone_number(subscription_update.phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format")
        subscription.phone_number = subscription_update.phone_number
    
    if subscription_update.min_change_threshold is not None:
        subscription.min_change_threshold = subscription_update.min_change_threshold
    
    if subscription_update.notification_frequency:
        subscription.notification_frequency = subscription_update.notification_frequency
    
    if subscription_update.tracked_symbols is not None:
        subscription.tracked_symbols = subscription_update.tracked_symbols
    
    if subscription_update.is_active is not None:
        subscription.is_active = subscription_update.is_active
    
    db.commit()
    db.refresh(subscription)
    
    return {
        "message": "Subscription updated",
        "subscription": {
            "id": subscription.id,
            "phone_number": subscription.phone_number,
            "min_change_threshold": subscription.min_change_threshold,
            "notification_frequency": subscription.notification_frequency,
            "tracked_symbols": subscription.tracked_symbols,
            "is_active": subscription.is_active
        }
    }

@app.delete("/api/v1/sms/subscription")
async def delete_sms_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel SMS subscription"""
    subscription = db.query(SMSSubscription).filter(SMSSubscription.user_id == current_user.id).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No SMS subscription found")
    
    subscription.is_active = False
    db.commit()
    
    return {"message": "SMS subscription deactivated"}

@app.post("/api/v1/sms/test")
@limiter.limit("3/minute")
async def send_test_sms_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a test SMS to verify phone number"""
    subscription = db.query(SMSSubscription).filter(SMSSubscription.user_id == current_user.id).first()
    
    if not subscription:
        raise HTTPException(status_code=404, detail="No SMS subscription found")
    
    if not subscription.phone_number:
        raise HTTPException(status_code=400, detail="No phone number configured")
    
    # Send test SMS asynchronously
    send_test_sms.delay(current_user.id, subscription.phone_number)
    
    return {"message": "Test SMS queued for delivery"}

@app.get("/api/v1/sms/notifications")
async def get_sms_notifications(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get SMS notification history"""
    notifications = db.query(SMSNotification)\
        .filter(SMSNotification.user_id == current_user.id)\
        .order_by(SMSNotification.created_at.desc())\
        .limit(limit)\
        .all()
    
    return [
        {
            "id": n.id,
            "symbol": n.symbol,
            "message": n.message,
            "status": n.status,
            "change_pct": n.change_pct,
            "current_price": n.current_price,
            "explanation": n.explanation,
            "created_at": n.created_at.isoformat(),
            "sent_at": n.sent_at.isoformat() if n.sent_at else None
        }
        for n in notifications
    ]

@app.post("/api/v1/sms/explain/{symbol}")
@limiter.limit("10/minute")
async def explain_stock_movement(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Get explanation for a stock's recent movement"""
    if not SMS_ENABLED:
        raise HTTPException(status_code=503, detail="Stock explainer service not available")
    
    movement_data = stock_explainer.explain_movement(symbol.upper())
    
    if not movement_data:
        raise HTTPException(status_code=404, detail=f"Could not get movement data for {symbol}")
    
        return {
            "symbol": movement_data['symbol'],
            "explanation": movement_data['explanation'],
            "change_pct": movement_data['change_pct'],
            "current_price": movement_data['current_price'],
            "previous_price": movement_data.get('previous_price'),
            "volume_ratio": movement_data.get('volume_ratio', 1.0)
        }

# ========== INDUSTRY FEATURES: DATA QUALITY, REGIME, QUALITY REPORT, WEBHOOKS, SCENARIO, VaR ==========

@app.get("/api/v1/data-quality/{symbol}")
@limiter.limit("30/minute")
async def get_data_quality(
    symbol: str,
    lookback_days: int = 90,
    current_user: User = Depends(get_current_user),
):
    """Get data quality score for a symbol. Predictions are gated when score is below threshold."""
    try:
        from data_quality_service import compute_data_quality
        result = compute_data_quality(symbol.strip().upper(), lookback_days=lookback_days)
        return result
    except Exception as e:
        logger.error(f"Data quality error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/regime")
@limiter.limit("30/minute")
async def get_market_regime(
    symbol: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """Get current market regime (low_vol, high_vol, trending_up, trending_down, crisis)."""
    try:
        from regime_service import detect_regime
        result = detect_regime(symbol=symbol)
        return result
    except Exception as e:
        logger.error(f"Regime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/quality-report")
@limiter.limit("20/minute")
async def get_quality_report(
    symbol: Optional[str] = None,
    horizon_days: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get prediction quality report: accuracy vs baseline, direction hit rate, by symbol/horizon."""
    from datetime import timedelta
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=90)
    filters = {}
    if symbol:
        filters["symbol"] = symbol.upper()
    if horizon_days is not None:
        filters["horizon_days"] = horizon_days
    q = db.query(PredictionQualityMetric).filter(
        PredictionQualityMetric.period_start >= period_start,
        PredictionQualityMetric.period_end <= period_end,
    )
    for k, v in filters.items():
        q = q.filter(getattr(PredictionQualityMetric, k) == v)
    rows = q.order_by(PredictionQualityMetric.computed_at.desc()).limit(50).all()
    return [
        {
            "symbol": r.symbol,
            "horizon_days": r.horizon_days,
            "period_start": r.period_start.isoformat() if r.period_start else None,
            "period_end": r.period_end.isoformat() if r.period_end else None,
            "sample_count": r.sample_count,
            "mape": r.mape,
            "mae": r.mae,
            "direction_hit_rate": r.direction_hit_rate,
            "vs_naive_improvement": r.vs_naive_improvement,
            "vs_buy_hold_note": r.vs_buy_hold_note,
            "abstention_count": r.abstention_count or 0,
            "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        }
        for r in rows
    ]


@app.post("/api/v1/quality-report/compute")
@limiter.limit("5/minute")
async def compute_quality_report(
    symbol: str,
    horizon_days: int = 7,
    current_user: User = Depends(check_subscription("premium")),
    db: Session = Depends(get_db),
):
    """Compute and persist prediction quality metrics for a symbol (Premium)."""
    from datetime import timedelta
    from quality_report_service import compute_quality_metrics
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=90)
    predictions = db.query(Prediction).filter(
        Prediction.symbol == symbol.upper(),
        Prediction.prediction_date >= period_start,
        Prediction.abstained == False,
    ).order_by(Prediction.prediction_date).all()
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found for this symbol in period")
    actual_list = []
    predicted_list = []
    for p in predictions:
        if p.actual_price is not None and p.predicted_price is not None:
            actual_list.append(p.actual_price)
            predicted_list.append(p.predicted_price)
    abstention_count = db.query(Prediction).filter(
        Prediction.symbol == symbol.upper(),
        Prediction.prediction_date >= period_start,
        Prediction.abstained == True,
    ).count()
    metric = compute_quality_metrics(
        actual_list, predicted_list, symbol.upper(), horizon_days,
        period_start, period_end, abstention_count=abstention_count,
    )
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
    return {"status": "ok", "metric": metric}


class WebhookCreate(BaseModel):
    url: str = Field(..., min_length=10, max_length=2000)
    secret: Optional[str] = None
    events: List[str] = Field(default=["prediction_created"])


@app.post("/api/v1/webhooks")
@limiter.limit("10/minute")
async def create_webhook(
    body: WebhookCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Subscribe to webhook events (prediction_created, prediction_updated, alert_triggered)."""
    sub = WebhookSubscription(
        user_id=current_user.id,
        url=body.url,
        secret=body.secret or "",
        events=body.events,
        is_active=True,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return {"id": sub.id, "url": sub.url, "events": sub.events, "is_active": sub.is_active}


@app.get("/api/v1/webhooks")
async def list_webhooks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List current user's webhook subscriptions."""
    subs = db.query(WebhookSubscription).filter(WebhookSubscription.user_id == current_user.id).all()
    return [
        {"id": s.id, "url": s.url, "events": s.events, "is_active": s.is_active, "last_triggered_at": s.last_triggered_at.isoformat() if s.last_triggered_at else None}
        for s in subs
    ]


@app.delete("/api/v1/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a webhook subscription."""
    sub = db.query(WebhookSubscription).filter(
        WebhookSubscription.id == webhook_id,
        WebhookSubscription.user_id == current_user.id,
    ).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Webhook not found")
    sub.is_active = False
    db.commit()
    return {"status": "deactivated"}


class ScenarioRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    scenario: str = Field(default="base", pattern="^(base|high_vol|market_down_5|market_up_2)$")
    vol_multiplier: Optional[float] = Field(None, ge=0.5, le=3.0)
    market_shock_pct: Optional[float] = Field(None, ge=-20, le=20)


@app.post("/api/v1/predict/scenario")
@limiter.limit(settings.RATE_LIMIT_PER_MINUTE)
async def scenario_prediction(
    request: ScenarioRequest,
    current_user: User = Depends(get_current_user),
):
    """Get scenario-adjusted prediction (e.g. high_vol or market shock)."""
    try:
        result = _predict_with_passport_and_abstention(
            request.symbol, "2020-01-01", datetime.utcnow().strftime("%Y-%m-%d"),
            True, "gru",
        )
        base_return = (result["predicted"][-1] / result["actual"][-1] - 1.0) if result["actual"] and result["actual"][-1] else 0.0
        from scenario_service import scenario_adjust_prediction
        adj = scenario_adjust_prediction(
            base_return,
            request.scenario,
            vol_multiplier=request.vol_multiplier or 1.5,
            market_shock_pct=request.market_shock_pct or (-5.0 if request.scenario == "market_down_5" else 2.0),
        )
        return {
            "symbol": request.symbol,
            "scenario": request.scenario,
            "base_predicted_return": base_return,
            "adjusted_return": adj,
            "base_price": result["predicted"][-1] if result["predicted"] else None,
            "adjusted_price": result["predicted"][-1] * (1 + adj) if result["predicted"] else None,
        }
    except Exception as e:
        logger.error(f"Scenario prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class PortfolioVaRRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=20)
    weights: Optional[List[float]] = None
    confidence: float = Field(default=0.95, ge=0.9, le=0.99)
    volatility_scale: float = Field(default=1.5, ge=1.0, le=3.0)


@app.post("/api/v1/portfolio/var")
@limiter.limit("20/minute")
async def portfolio_var(
    request: PortfolioVaRRequest,
    current_user: User = Depends(get_current_user),
):
    """Estimate portfolio Value-at-Risk using predicted returns and volatility scale."""
    try:
        from scenario_service import portfolio_var_from_predictions
        predicted_returns = []
        for sym in request.symbols:
            try:
                r = _predict_with_passport_and_abstention(sym, "2020-01-01", datetime.utcnow().strftime("%Y-%m-%d"), False, "gru")
                if r["actual"] and r["predicted"]:
                    ret = (r["predicted"][-1] / r["actual"][-1] - 1.0)
                    predicted_returns.append(ret)
                else:
                    predicted_returns.append(0.0)
            except Exception:
                predicted_returns.append(0.0)
        if not predicted_returns:
            predicted_returns = [0.0] * len(request.symbols)
        out = portfolio_var_from_predictions(
            request.symbols, predicted_returns,
            volatility_scale=request.volatility_scale,
            confidence=request.confidence,
        )
        return out
    except Exception as e:
        logger.error(f"VaR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predict/{prediction_id}/explain")
@limiter.limit("20/minute")
async def explain_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get feature importance (explainability) for a prediction."""
    p = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id,
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Prediction not found")
    try:
        from explainability_service import feature_importance_permutation, explain_prediction as explain_pred
        import numpy as np
        feature_names = ["Close", "MA_5", "MA_10", "MA_20", "RSI", "MACD", "BB_Width", "STD_20", "RET_1", "RET_5"]
        imp = {}
        try:
            from predictor import predict
            actual, predicted = predict(p.symbol, "2020-01-01", datetime.utcnow().strftime("%Y-%m-%d"), use_attention=False)
            arr = np.array(predicted[-60:], dtype=np.float32) if len(predicted) >= 60 else np.array(predicted, dtype=np.float32)
            n_f = min(10, len(feature_names))
            if arr.size >= n_f:
                X = arr[-n_f:].reshape(1, n_f)
                def pred_fn(x):
                    return np.array([float(p.predicted_price)])
                imp = feature_importance_permutation(pred_fn, X, feature_names[:n_f], n_repeats=2, baseline_pred=float(p.predicted_price))
            else:
                imp = {f: 0.0 for f in feature_names[:n_f]}
        except Exception:
            imp = {f: 0.0 for f in feature_names[:5]}
        explanation = explain_pred(imp, top_n=5)
        return {"prediction_id": prediction_id, "symbol": p.symbol, "feature_importance": imp, "explanation": explanation}
    except Exception as e:
        logger.error(f"Explain error: {e}")
        return {"prediction_id": prediction_id, "symbol": p.symbol, "feature_importance": {}, "explanation": {"summary": "Explanation not available"}, "error": str(e)}


@app.get("/api/v1/admin/degradation-alerts")
async def list_degradation_alerts(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List model degradation alerts (Admin)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    alerts = db.query(ModelDegradationAlert).order_by(ModelDegradationAlert.triggered_at.desc()).limit(limit).all()
    return [
        {
            "id": a.id, "symbol": a.symbol, "metric_name": a.metric_name,
            "previous_value": a.previous_value, "current_value": a.current_value,
            "threshold": a.threshold, "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
            "acknowledged": a.acknowledged,
        }
        for a in alerts
    ]


# ========== HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")