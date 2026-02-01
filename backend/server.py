from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import os
import sys
from datetime import datetime
import uvicorn
import logging
from pathlib import Path

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from predictor import train, predict, compute_rmse, compute_mape, download_stock
except ImportError as e:
    logging.error(f"Failed to import predictor: {e}")
    raise

from motor.motor_asyncio import AsyncIOMotorClient
import yfinance as yf
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Prediction API", version="1.0.0")

# CORS Configuration - More secure defaults
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# MongoDB Configuration with connection pooling
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
try:
    client = AsyncIOMotorClient(
        MONGO_URL,
        maxPoolSize=10,
        minPoolSize=1,
        serverSelectionTimeoutMS=5000
    )
    db = client.stock_prediction
    logger.info("MongoDB connection initialized")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None  # Allow app to run without MongoDB

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()
        logger.info("MongoDB connection closed")

# Available stocks - Expanded list (30+ stocks)
AVAILABLE_STOCKS = [
    # Technology
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive/Tech"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Entertainment"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technology"},
    {"symbol": "IBM", "name": "IBM Corporation", "sector": "Technology"},
    
    # Finance
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Finance"},
    {"symbol": "GS", "name": "Goldman Sachs Group", "sector": "Finance"},
    {"symbol": "MS", "name": "Morgan Stanley", "sector": "Finance"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Finance"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Finance"},
    
    # Healthcare
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
    {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Healthcare"},
    
    # Retail & Consumer
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Retail"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Retail"},
    {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consumer"},
    {"symbol": "COST", "name": "Costco Wholesale", "sector": "Retail"},
    {"symbol": "DIS", "name": "Walt Disney Company", "sector": "Entertainment"},
    
    # Energy
    {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    
    # Others
    {"symbol": "BA", "name": "Boeing Company", "sector": "Aerospace"},
    {"symbol": "KO", "name": "Coca-Cola Company", "sector": "Beverages"},
    {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Beverages"},
]

# Validation functions
def validate_symbol(symbol: str) -> str:
    """Validate stock symbol format."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol

def validate_date(date_str: str) -> datetime:
    """Validate date string."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD")

# Pydantic Models with validation
class TrainRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5)
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    epochs: int = Field(10, ge=1, le=100)
    seq_len: int = Field(60, ge=10, le=200)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return validate_symbol(v)
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v, values):
        if v:
            date_obj = validate_date(v)
            if date_obj > datetime.now():
                raise ValueError("Date cannot be in the future")
            if 'start_date' in values and values.get('start_date'):
                start = validate_date(values['start_date'])
                if date_obj <= start:
                    raise ValueError("End date must be after start date")
        return v

class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5)
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    seq_len: int = Field(60, ge=10, le=200)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return validate_symbol(v)
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        if v:
            validate_date(v)
        return v

class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    current_price: Optional[float] = None
    change_percent: Optional[float] = None

# Routes
@app.get("/")
async def root():
    return {"message": "Stock Prediction API is running", "version": "1.0"}

@app.get("/api/stocks")
async def get_stocks():
    """Get list of all available stocks with current prices"""
    stocks_with_info = []
    for stock in AVAILABLE_STOCKS:
        try:
            ticker = yf.Ticker(stock["symbol"])
            info = ticker.info
            stock_info = stock.copy()
            stock_info["current_price"] = info.get('regularMarketPrice') or info.get('currentPrice')
            stock_info["change_percent"] = info.get('regularMarketChangePercent')
            stocks_with_info.append(stock_info)
        except Exception as e:
            logger.warning(f"Failed to fetch info for {stock['symbol']}: {e}")
            # Add stock without price info
            stock_info = stock.copy()
            stock_info["current_price"] = None
            stock_info["change_percent"] = None
            stocks_with_info.append(stock_info)
    
    return {"stocks": stocks_with_info, "total": len(stocks_with_info)}

@app.get("/api/stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get detailed information about a specific stock"""
    try:
        symbol = validate_symbol(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'symbol' not in info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        stock_data = {
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unknown'),
            "current_price": info.get('regularMarketPrice') or info.get('currentPrice'),
            "change_percent": info.get('regularMarketChangePercent'),
            "market_cap": info.get('marketCap'),
            "volume": info.get('volume'),
            "high_52week": info.get('fiftyTwoWeekHigh'),
            "low_52week": info.get('fiftyTwoWeekLow'),
        }
        return stock_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found: {str(e)}")

@app.post("/api/train/batch")
async def batch_train_models(symbols: List[str], start_date: str = "2018-01-01",
                            end_date: Optional[str] = None, epochs: int = 10,
                            background_tasks: BackgroundTasks = None):
    """Train multiple stock models in batch."""
    try:
        from batch_trainer import batch_train_stocks
        
        def train_background():
            try:
                results = batch_train_stocks(
                    symbols=symbols,
                    start=start_date,
                    end=end_date,
                    epochs=epochs,
                    parallel=True,
                    max_workers=3
                )
                logger.info(f"Batch training completed: {results}")
            except Exception as e:
                logger.error(f"Batch training error: {e}", exc_info=True)
        
        if background_tasks:
            background_tasks.add_task(train_background)
        
        return {
            "message": f"Batch training started for {len(symbols)} stocks",
            "symbols": symbols,
            "status": "training_in_progress"
        }
    except Exception as e:
        logger.error(f"Batch train endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a GRU model for the specified stock"""
    try:
        # Validate stock symbol
        valid_symbols = [s["symbol"] for s in AVAILABLE_STOCKS]
        if request.symbol not in valid_symbols:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stock symbol. Choose from: {valid_symbols}"
            )
        
        # Start training in background
        def train_background():
            try:
                logger.info(f"Starting training for {request.symbol}")
                model_path, scaler_path = train(
                    symbol=request.symbol,
                    start=request.start_date,
                    end=request.end_date,
                    seq_len=request.seq_len,
                    epochs=request.epochs,
                    save=True
                )
                # Save training log to database
                if db:
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            db.training_logs.insert_one({
                                "symbol": request.symbol,
                                "start_date": request.start_date,
                                "end_date": request.end_date or datetime.now().strftime("%Y-%m-%d"),
                                "epochs": request.epochs,
                                "model_path": model_path,
                                "trained_at": datetime.now()
                            })
                        )
                        loop.close()
                    except Exception as db_error:
                        logger.error(f"Failed to save training log: {db_error}")
                
                logger.info(f"Training completed for {request.symbol}")
            except Exception as e:
                logger.error(f"Training error for {request.symbol}: {e}", exc_info=True)
        
        background_tasks.add_task(train_background)
        
        return {
            "message": f"Training started for {request.symbol}",
            "symbol": request.symbol,
            "epochs": request.epochs,
            "status": "training_in_progress"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/predict")
async def predict_stock(request: PredictRequest):
    """Get predictions for the specified stock"""
    try:
        # Validate stock symbol
        valid_symbols = [s["symbol"] for s in AVAILABLE_STOCKS]
        if request.symbol not in valid_symbols:
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol")
        
        # Get predictions
        actual, predicted = predict(
            symbol=request.symbol,
            start=request.start_date,
            end=request.end_date,
            seq_len=request.seq_len,
            use_attention=False
        )
        
        # Calculate metrics
        rmse = float(compute_rmse(actual, predicted))
        mape = float(compute_mape(actual, predicted))
        
        # Save prediction to database
        await db.predictions.insert_one({
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date or datetime.now().strftime("%Y-%m-%d"),
            "rmse": rmse,
            "mape": mape,
            "predicted_at": datetime.now(),
            "num_predictions": len(predicted)
        })
        
        return {
            "symbol": request.symbol,
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
            "metrics": {
                "rmse": rmse,
                "mape": mape
            },
            "length": len(predicted)
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found for {request.symbol}. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{symbol}")
async def get_prediction_history(symbol: str, limit: int = 10):
    """Get prediction history for a stock"""
    try:
        predictions = await db.predictions.find(
            {"symbol": symbol}
        ).sort("predicted_at", -1).limit(limit).to_list(length=limit)
        
        # Convert ObjectId to string
        for pred in predictions:
            pred["_id"] = str(pred["_id"])
            pred["predicted_at"] = pred["predicted_at"].isoformat()
        
        return {"symbol": symbol, "history": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_trained_models():
    """Get list of all trained models from registry"""
    try:
        from batch_trainer import get_trained_models, get_model_info
        
        # Try to get from registry first
        trained_symbols = get_trained_models()
        models = []
        
        for symbol in trained_symbols:
            info = get_model_info(symbol)
            if info:
                models.append({
                    "symbol": symbol,
                    "path": info.get("model_path", ""),
                    "trained_at": info.get("trained_at", ""),
                    "metadata": info.get("metadata", {})
                })
        
        if models:
            return {"models": models, "total": len(models), "source": "registry"}
    except Exception as e:
        logger.warning(f"Failed to load from registry: {e}")
    
    # Fallback to file system scan
    import glob
    models_dir = BASE_DIR / "models"
    
    if not models_dir.exists():
        return {"models": [], "total": 0, "source": "filesystem"}
    
    model_files = glob.glob(str(models_dir / "*_gru.pth"))
    models = []
    
    for model_file in model_files:
        try:
            symbol = os.path.basename(model_file).replace("_gru.pth", "")
            models.append({
                "symbol": symbol,
                "path": model_file,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(model_file)).isoformat()
            })
        except Exception as e:
            logger.warning(f"Error processing model file {model_file}: {e}")
    
    return {"models": models, "total": len(models), "source": "filesystem"}

@app.get("/api/download/{symbol}")
async def download_stock_data(symbol: str, start: str = "2018-01-01", end: Optional[str] = None):
    """Download and return historical stock data"""
    try:
        df = download_stock(symbol, start=start, end=end, cache=True)
        
        data = {
            "symbol": symbol,
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "close": df["Close"].tolist(),
            "open": df["Open"].tolist() if "Open" in df.columns else [],
            "high": df["High"].tolist() if "High" in df.columns else [],
            "low": df["Low"].tolist() if "Low" in df.columns else [],
            "volume": df["Volume"].tolist() if "Volume" in df.columns else [],
        }
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8001))
    host = os.environ.get('HOST', '0.0.0.0')
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
