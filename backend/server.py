from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from datetime import datetime
import uvicorn

# Import existing predictor utilities
import sys
sys.path.append('/app')
from predictor import train, predict, compute_rmse, compute_mape, download_stock

from motor.motor_asyncio import AsyncIOMotorClient
import yfinance as yf
import numpy as np

app = FastAPI(title="Stock Prediction API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URL)
db = client.stock_prediction

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

# Pydantic Models
class TrainRequest(BaseModel):
    symbol: str
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    epochs: int = 10
    seq_len: int = 60

class PredictRequest(BaseModel):
    symbol: str
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    seq_len: int = 60

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
        except:
            stocks_with_info.append(stock)
    
    return {"stocks": stocks_with_info, "total": len(stocks_with_info)}

@app.get("/api/stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get detailed information about a specific stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found: {str(e)}")

@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a GRU model for the specified stock"""
    try:
        # Validate stock symbol
        valid_symbols = [s["symbol"] for s in AVAILABLE_STOCKS]
        if request.symbol not in valid_symbols:
            raise HTTPException(status_code=400, detail=f"Invalid stock symbol. Choose from: {valid_symbols}")
        
        # Start training in background
        def train_background():
            try:
                model_path, scaler_path = train(
                    symbol=request.symbol,
                    start=request.start_date,
                    end=request.end_date,
                    seq_len=request.seq_len,
                    epochs=request.epochs,
                    save=True
                )
                # Save training log to database
                db.training_logs.insert_one({
                    "symbol": request.symbol,
                    "start_date": request.start_date,
                    "end_date": request.end_date or datetime.now().strftime("%Y-%m-%d"),
                    "epochs": request.epochs,
                    "model_path": model_path,
                    "trained_at": datetime.now()
                })
            except Exception as e:
                print(f"Training error: {e}")
        
        background_tasks.add_task(train_background)
        
        return {
            "message": f"Training started for {request.symbol}",
            "symbol": request.symbol,
            "epochs": request.epochs,
            "status": "training_in_progress"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    """Get list of all trained models"""
    import glob
    models_dir = "/app/models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    model_files = glob.glob(os.path.join(models_dir, "*_gru.pth"))
    models = []
    
    for model_file in model_files:
        symbol = os.path.basename(model_file).replace("_gru.pth", "")
        models.append({
            "symbol": symbol,
            "path": model_file,
            "last_modified": datetime.fromtimestamp(os.path.getmtime(model_file)).isoformat()
        })
    
    return {"models": models, "total": len(models)}

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
    uvicorn.run(app, host="0.0.0.0", port=8001)
