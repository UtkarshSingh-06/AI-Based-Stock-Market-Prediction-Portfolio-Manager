# predictor.py
"""
Core prediction and training utilities for the GRU Stock Forecasting project.

Provides:
- Data download & caching (yfinance)
- Technical feature creation
- Sequence building for GRU input
- GRU model (with optional attention)
- Train / save / load model (model + scaler)
- Predict and evaluation helpers
"""

from typing import Tuple, List, Optional, Dict
import os
import math
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Config / paths / device
# ---------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RND_SEED = 42
torch.manual_seed(RND_SEED)
np.random.seed(RND_SEED)


# ---------------------------
# Data utilities
# ---------------------------
def download_stock(symbol: str, start: str = "2008-01-01", end: Optional[str] = None,
                   cache: bool = True, retries: int = 3) -> pd.DataFrame:
    """Download (or load cached) OHLCV data for symbol with retry logic."""
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Validate symbol
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    
    # Validate dates
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        if start_date > datetime.now():
            raise ValueError("Start date cannot be in the future")
    except ValueError as e:
        raise ValueError(f"Invalid start date format. Use YYYY-MM-DD: {e}")
    
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    else:
        try:
            end_date = datetime.strptime(end, "%Y-%m-%d")
            if end_date > datetime.now():
                raise ValueError("End date cannot be in the future")
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            raise ValueError(f"Invalid end date: {e}")
    
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    # Check cache
    if cache and os.path.exists(csv_path):
        try:
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(csv_path))
            if file_age.days < 1:  # Use cache if less than 1 day old
                logger.info(f"Loading cached data for {symbol}")
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if not df.empty and "Close" in df.columns:
                    return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # Download with retry
    for attempt in range(retries):
        try:
            logger.info(f"Downloading {symbol} data (attempt {attempt+1}/{retries})")
            df = yf.download(symbol, start=start, end=end, progress=False, threads=False)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            if "Close" not in df.columns:
                raise ValueError("Downloaded data has no 'Close' column.")
            
            # Save to cache
            if cache:
                df.to_csv(csv_path)
                logger.info(f"Cached data for {symbol}")
            
            return df
        except Exception as e:
            logger.error(f"Download attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                raise ValueError(f"Failed to download data for {symbol} after {retries} attempts: {e}")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators used as features."""
    df = df.copy()
    
    # Moving Averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # Volatility
    df["STD_10"] = df["Close"].rolling(window=10).std()
    df["STD_20"] = df["Close"].rolling(window=20).std()
    
    # Returns
    df["RET_1"] = df["Close"].pct_change().fillna(0)
    df["RET_5"] = df["Close"].pct_change(periods=5).fillna(0)
    
    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
    bb_std_val = df["Close"].rolling(window=bb_period).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * bb_std_val)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * bb_std_val)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Width"] + 1e-10)
    
    # Volume indicators (if Volume column exists)
    if "Volume" in df.columns:
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-10)
        df["Price_Volume"] = df["Close"] * df["Volume"]
    
    # High-Low indicators
    if "High" in df.columns and "Low" in df.columns:
        df["HL_Range"] = df["High"] - df["Low"]
        df["HL_Pct"] = df["HL_Range"] / (df["Close"] + 1e-10)
        df["Body"] = abs(df["Close"] - df["Open"]) if "Open" in df.columns else 0
    
    # Fill NaN values
    df = df.bfill().ffill()
    
    # Replace any remaining inf or NaN with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def build_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Turn feature DataFrame into sequences (X) and next-day target (y)."""
    if len(df) < seq_len + 1:
        raise ValueError(f"DataFrame too short. Need at least {seq_len + 1} rows, got {len(df)}")
    
    # Check all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    arr = df[feature_cols].values
    
    # Check for NaN or inf values
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Data contains NaN or Inf values. Please clean data first.")
    
    sequences, targets = [], []
    for i in range(len(arr) - seq_len):
        sequences.append(arr[i:i + seq_len])
        targets.append(arr[i + seq_len][feature_cols.index("Close")])  # predict Close
    
    if len(sequences) == 0:
        raise ValueError("No sequences generated from data")
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32).reshape(-1, 1)


# ---------------------------
# Model definitions
# ---------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H: (batch, seq_len, hidden)
        score = torch.tanh(self.W(H))                   # (batch, seq_len, hidden)
        e = self.v(score).squeeze(-1)                   # (batch, seq_len)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)   # (batch, seq_len, 1)
        context = (H * alpha).sum(dim=1)                # (batch, hidden)
        return context, alpha


class EnhancedGRUModel(nn.Module):
    """Enhanced GRU model with bidirectional option, layer normalization, and residual connections."""
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2,
                 dropout: float = 0.2, use_attention: bool = False, bidirectional: bool = False,
                 use_layer_norm: bool = True, use_residual: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization
        if use_layer_norm:
            gru_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.layer_norm = nn.LayerNorm(gru_output_size)
        else:
            self.layer_norm = None
        
        # Attention mechanism
        if use_attention:
            gru_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.att = Attention(gru_output_size)
        else:
            self.att = None
        
        # Fully connected layers with dropout
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        
        # Residual connection (if input and output sizes match)
        if use_residual and input_size == 1:
            self.residual_proj = nn.Linear(input_size, 1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)  # out: (batch, seq_len, hidden)
        
        # Apply layer normalization
        if self.layer_norm is not None:
            gru_out = self.layer_norm(gru_out)
        
        # Attention or last timestep
        if self.use_attention and self.att is not None:
            context, _ = self.att(gru_out)
            out = context
        else:
            out = gru_out[:, -1, :]  # last timestep
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        # Residual connection (if applicable)
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x[:, -1, :])  # last timestep of input
            out = out + residual
        
        return out


class GRUWithOptionalAttention(nn.Module):
    """Original GRU model for backward compatibility."""
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2,
                 dropout: float = 0.0, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.att = Attention(hidden_size) if use_attention else None
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)  # out: (batch, seq_len, hidden)
        if self.use_attention:
            context, _ = self.att(out)
            return self.fc(context)
        else:
            return self.fc(out[:, -1, :])  # last timestep


# ---------------------------
# Save / Load helpers
# ---------------------------
def save_model_and_scaler(symbol: str, model: nn.Module, scaler: MinMaxScaler, 
                         metadata: Optional[Dict] = None):
    """Save model and scaler with error handling and metadata."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    symbol = symbol.strip().upper()
    path = os.path.join(MODELS_DIR, f"{symbol}_gru.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{symbol}_metadata.json")
    
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Saved model to {path}")
    except Exception as e:
        raise IOError(f"Failed to save model: {e}")
    
    try:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
    except Exception as e:
        raise IOError(f"Failed to save scaler: {e}")
    
    # Save metadata if provided
    if metadata:
        try:
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    return path, scaler_path


def load_model_and_scaler(symbol: str, input_size: int, use_attention: bool = False, 
                          map_location: str = "cpu", use_enhanced: bool = True):
    """Load model and scaler with proper error handling. Tries enhanced model first."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    symbol = symbol.strip().upper()
    path = os.path.join(MODELS_DIR, f"{symbol}_gru.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_scaler.pkl")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found for symbol: {symbol}. Path: {path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found for symbol: {symbol}. Path: {scaler_path}")
    
    try:
        # Try enhanced model first
        if use_enhanced:
            try:
                model = EnhancedGRUModel(
                    input_size=input_size,
                    hidden_size=50,
                    num_layers=2,
                    dropout=0.2,
                    use_attention=use_attention,
                    bidirectional=False,
                    use_layer_norm=True,
                    use_residual=False
                ).to(device)
                state = torch.load(path, map_location=map_location)
                model.load_state_dict(state, strict=False)  # Allow partial loading
                logger.info(f"Loaded enhanced model for {symbol}")
            except Exception:
                # Fallback to original model
                logger.info(f"Enhanced model failed, using original model for {symbol}")
                model = GRUWithOptionalAttention(input_size=input_size, use_attention=use_attention).to(device)
                state = torch.load(path, map_location=map_location)
                model.load_state_dict(state)
        else:
            # Use original model
            model = GRUWithOptionalAttention(input_size=input_size, use_attention=use_attention).to(device)
            state = torch.load(path, map_location=map_location)
            model.load_state_dict(state)
            logger.info(f"Loaded original model for {symbol}")
    except Exception as e:
        raise IOError(f"Failed to load model for {symbol}: {e}")
    
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler for {symbol}")
    except Exception as e:
        raise IOError(f"Failed to load scaler for {symbol}: {e}")
    
    model.eval()
    return model, scaler


# ---------------------------
# Training & prediction
# ---------------------------
def train(symbol: str,
          start: str = "2008-01-01",
          end: Optional[str] = None,
          seq_len: int = 60,
          epochs: int = 10,
          batch_size: int = 32,
          lr: float = 1e-3,
          use_attention: bool = False,
          save: bool = True) -> Tuple[str, str]:
    """
    Train a GRU for a given symbol. Saves model and scaler to models/ by default.
    Returns paths (model_path, scaler_path).
    """
    df = download_stock(symbol, start=start, end=end, cache=True)
    df = add_technical_indicators(df)
    
    # Use enhanced feature set if available
    basic_features = ["Close", "MA_5", "MA_10", "EMA_10", "STD_10", "RET_1"]
    enhanced_features = ["Close", "MA_5", "MA_10", "MA_20", "EMA_10", "EMA_20", 
                        "STD_10", "STD_20", "RET_1", "RET_5", "RSI", "MACD", 
                        "MACD_Signal", "BB_Position", "BB_Width"]
    
    # Check which features are available
    available_features = [f for f in enhanced_features if f in df.columns]
    if len(available_features) < len(basic_features):
        feature_cols = [f for f in basic_features if f in df.columns]
    else:
        feature_cols = available_features
    
    # Ensure Close is always first
    if "Close" not in feature_cols:
        feature_cols.insert(0, "Close")
    elif feature_cols[0] != "Close":
        feature_cols.remove("Close")
        feature_cols.insert(0, "Close")
    
    X, y = build_sequences(df, feature_cols, seq_len=seq_len)
    if len(X) < 10:
        raise ValueError("Not enough sequence data to train.")

    # split train/val (simple) with validation
    if len(X) < 20:
        raise ValueError(f"Not enough data for train/val split. Need at least 20 sequences, got {len(X)}")
    
    split = int(0.9 * len(X))
    if split == 0:
        split = max(1, len(X) - 1)  # Ensure at least 1 validation sample
    
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train/validation split resulted in empty sets")

    scaler = MinMaxScaler()
    # fit scaler on training features (reshape)
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat_train)
    # transform
    X_train_scaled = scaler.transform(flat_train).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    # convert to tensors
    X_tr = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # model - use enhanced model if available, fallback to original
    input_size = X_tr.shape[-1]
    use_enhanced = True  # Set to False to use original model
    
    if use_enhanced:
        model = EnhancedGRUModel(
            input_size=input_size, 
            hidden_size=hidden_size if 'hidden_size' in locals() else 50,
            num_layers=num_layers if 'num_layers' in locals() else 2,
            dropout=0.2,
            use_attention=use_attention,
            bidirectional=False,  # Can be enabled for better performance
            use_layer_norm=True,
            use_residual=False
        ).to(device)
    else:
        model = GRUWithOptionalAttention(input_size=input_size, use_attention=use_attention).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Use Huber loss for robustness (less sensitive to outliers than MSE)
    criterion = nn.HuberLoss(delta=1.0)  # Can fallback to MSELoss() if needed

    # training loop with gradient clipping and early stopping
    import logging
    logger = logging.getLogger(__name__)
    
    model.train()
    num_batches = max(1, int(np.ceil(len(X_tr) / batch_size)))
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_tr))
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb = X_tr[idx]; yb = y_tr[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # validation performance
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t).cpu().numpy()
            val_rmse = math.sqrt(((y_val - pred_val) ** 2).mean())
            val_mape = (np.abs((y_val - pred_val) / (y_val + 1e-8)).mean()) * 100.0
            val_loss = val_rmse  # Use RMSE as validation loss
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        model.train()
        logger.info(f"[{symbol}] Epoch {epoch+1}/{epochs} train_loss={(epoch_loss / num_batches):.6f} val_rmse={val_rmse:.4f} val_mape={val_mape:.2f}%")

    # save model + scaler
    model_path, scaler_path = "", ""
    if save:
        model_path, scaler_path = save_model_and_scaler(symbol, model, scaler)
        print(f"Saved model: {model_path} scaler: {scaler_path}")

    return model_path, scaler_path


def predict_future(symbol: str,
                   days_ahead: int = 30,
                   seq_len: int = 60,
                   use_attention: bool = False,
                   use_enhanced: bool = True) -> np.ndarray:
    """
    Predict future stock prices for N days ahead.
    Returns array of predicted prices.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get recent data
    df = download_stock(symbol, start=None, end=None, cache=True)
    df = add_technical_indicators(df)
    feature_cols = ["Close", "MA_5", "MA_10", "EMA_10", "STD_10", "RET_1"]
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(symbol, input_size=len(feature_cols), 
                                          use_attention=use_attention, map_location=device)
    
    predictions = []
    current_data = df[feature_cols].tail(seq_len).values
    
    model.eval()
    with torch.no_grad():
        for day in range(days_ahead):
            # Scale current sequence
            current_scaled = scaler.transform(current_data)
            X_t = torch.tensor(current_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict next day
            pred_scaled = model(X_t).cpu().numpy()
            
            # Inverse transform only the Close price (first feature)
            # Create dummy array for inverse transform
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, 0] = pred_scaled[0, 0]  # Close price
            pred = scaler.inverse_transform(dummy)[0, 0]
            predictions.append(pred)
            
            # Update current_data with prediction (for next iteration)
            # Create new row with predicted close
            new_row = current_data[-1].copy()
            new_row[0] = pred  # Update Close
            # Update other features (simplified - could be more sophisticated)
            new_row[1] = np.mean([current_data[-4:, 0].mean(), pred])  # MA_5 approximation
            new_row[2] = np.mean([current_data[-9:, 0].mean(), pred])  # MA_10 approximation
            new_row[3] = 0.9 * new_row[3] + 0.1 * pred  # EMA_10 approximation
            new_row[4] = current_data[-10:, 0].std()  # STD_10
            new_row[5] = (pred - current_data[-1, 0]) / (current_data[-1, 0] + 1e-10)  # RET_1
            
            # Shift window
            current_data = np.vstack([current_data[1:], new_row.reshape(1, -1)])
    
    logger.info(f"Generated {days_ahead} future predictions for {symbol}")
    return np.array(predictions)


def predict(symbol: str,
            start: str = "2008-01-01",
            end: Optional[str] = None,
            seq_len: int = 60,
            use_attention: bool = False,
            return_series: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load model and scaler for symbol and return (actual_close_array, predicted_array).
    The shapes are (n,) arrays aligned so predicted[i] corresponds to actual[i + seq_len] in original data.
    """
    df = download_stock(symbol, start=start, end=end, cache=True)
    df = add_technical_indicators(df)
    feature_cols = ["Close", "MA_5", "MA_10", "EMA_10", "STD_10", "RET_1"]
    X, y = build_sequences(df, feature_cols, seq_len=seq_len)
    if len(X) == 0:
        raise ValueError("Not enough data to produce sequences for prediction.")

    # load model + scaler (try enhanced model first, fallback to original)
    try:
        # Try to load enhanced model
        model, scaler = load_model_and_scaler(symbol, input_size=X.shape[-1], 
                                              use_attention=use_attention, map_location=device)
    except Exception as e:
        # Fallback to original model loading
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load model, trying original format: {e}")
        model, scaler = load_model_and_scaler(symbol, input_size=X.shape[-1], 
                                              use_attention=False, map_location=device)
    # scale all sequences
    flat = X.reshape(-1, X.shape[-1])
    flat_scaled = scaler.transform(flat)
    X_scaled = flat_scaled.reshape(X.shape)

    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy().reshape(-1, 1)

    # preds correspond to scaled inverse transform only for close; since scaler used multivariate transform,
    # simplest approach here: we already predicted close in the original scale, because target was not scaled.
    # (We trained the model to predict the real close because scaler was applied to features;
    # the target used the original Close value, so preds are in original close scale.)
    preds = preds.flatten()
    actual = y.flatten()
    return actual, preds


# ---------------------------
# Ensemble prediction
# ---------------------------
def ensemble_predict_models(symbol: str, models: List[nn.Module], X: torch.Tensor,
                            method: str = 'mean') -> np.ndarray:
    """Make ensemble predictions from multiple models."""
    from model_utils import ensemble_predict
    return ensemble_predict(models, X, method)


# ---------------------------
# Metrics helpers
# ---------------------------
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(((y_true - y_pred) ** 2).mean())


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.abs((y_true - y_pred) / (y_true + 1e-8))).mean() * 100.0)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.abs(y_true - y_pred).mean())


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared score"""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


# ---------------------------
# Quick demo (if run directly)
# ---------------------------
if __name__ == "__main__":
    # simple demo: train a tiny model for MSFT (short epochs)
    sym = "MSFT"
    print("Training demo (short):", sym)
    train(sym, start="2018-01-01", end=None, epochs=3, batch_size=64, save=False)
    print("Demo predict (will fail if model not saved):")
    try:
        actual, pred = predict(sym, start="2018-01-01", end=None)
        print("Actual len:", len(actual), "Pred len:", len(pred))
        print("RMSE:", compute_rmse(actual, pred), "MAPE:", compute_mape(actual, pred))
    except Exception as e:
        print("Predict error (expected if model not saved):", e)























# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import yfinance as yf

# # Load stock data (Fetch from Yahoo Finance)
# def load_stock_data(stock_symbol):
#     stock = yf.download(stock_symbol, start="2020-01-01", end="2024-04-01")
#     stock.to_csv(f"{stock_symbol}.csv")  # Save for future use
#     return stock[['Close']]

# # Preprocess Data
# def preprocess_data(df):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df)
#     return scaled_data, scaler

# # Create sequences for GRU
# def create_sequences(data, seq_length=60):
#     sequences, targets = [], []
#     for i in range(len(data) - seq_length):
#         sequences.append(data[i:i+seq_length])
#         targets.append(data[i+seq_length])
#     return np.array(sequences), np.array(targets)

# # Define GRU Model
# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1, :])

# # Train model function
# def train_model(model, train_loader, criterion, optimizer, epochs=20):
#     for epoch in range(epochs):
#         for sequences, targets in train_loader:
#             sequences, targets = sequences.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(sequences)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# # Prediction function
# def predict(model, data, scaler, seq_length=60):
#     model.eval()
#     inputs = torch.tensor(data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
#         predicted = model(inputs).cpu().numpy()
#     return scaler.inverse_transform(predicted)

# # Combined visualization function for all stocks
# def plot_all_predictions(stock_data, predictions, stock_symbols):
#     plt.figure(figsize=(12, 6))

#     colors = ['blue', 'green', 'red']
    
#     for i, stock in enumerate(stock_symbols):
#         actual_prices = stock_data[stock]['Close'].values
#         predicted_prices = predictions[stock]

#         # Plot actual prices
#         plt.plot(actual_prices, label=f'Actual {stock}', color=colors[i], linestyle='dotted')
        
#         # Plot predicted prices (aligned at the end)
#         plt.plot(range(len(actual_prices) - len(predicted_prices), len(actual_prices)), 
#                  predicted_prices, label=f'Predicted {stock}', color=colors[i])

#     plt.title('Stock Price Prediction (AMZN, IBM, MSFT)')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.show()

# # Load and process data
# stock_symbols = ['AMZN', 'IBM', 'MSFT']
# seq_length = 60
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# stock_data = {}
# predictions = {}

# for stock in stock_symbols:
#     df = load_stock_data(stock)
#     stock_data[stock] = df  # Store actual stock data

#     data, scaler = preprocess_data(df)
#     sequences, targets = create_sequences(data, seq_length)
    
#     # Convert to PyTorch tensors
#     sequences = torch.tensor(sequences, dtype=torch.float32)
#     targets = torch.tensor(targets, dtype=torch.float32)
    
#     # Create DataLoader
#     train_dataset = torch.utils.data.TensorDataset(sequences, targets)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
#     # Model initialization
#     model = GRUModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Train model
#     train_model(model, train_loader, criterion, optimizer)
    
#     # Save model
#     torch.save(model.state_dict(), f'{stock}_gru_model.pth')
#     print(f"Trained model saved for {stock}")
    
#     # Predict next day's stock price
#     predicted_price = predict(model, data, scaler)
#     predictions[stock] = predicted_price.flatten()  # Store predicted prices

# # Plot all stocks together
# plot_all_predictions(stock_data, predictions, stock_symbols)



