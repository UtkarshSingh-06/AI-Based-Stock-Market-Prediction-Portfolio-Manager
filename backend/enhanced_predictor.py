# backend/enhanced_predictor.py
"""
Enhanced prediction module with:
- MC Dropout uncertainty
- Advanced technical indicators
- Sentiment analysis
- Backtesting
- Multiple model types (GRU, LSTM, Transformer)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Dict
import math
from datetime import datetime, timedelta
import pickle
import os
import requests
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# ========== ADVANCED TECHNICAL INDICATORS ==========
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators"""
    df = df.copy()
    ti = TechnicalIndicators()
    
    # Price-based
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    df['RSI'] = ti.calculate_rsi(df['Close'])
    df['RSI_SMA'] = df['RSI'].rolling(window=14).mean()
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ti.calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ti.calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volatility
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    df['Volatility_50'] = df['Returns'].rolling(window=50).std()
    
    if 'High' in df.columns and 'Low' in df.columns:
        # ATR
        df['ATR'] = ti.calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = ti.calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    if 'Volume' in df.columns:
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['OBV'] = ti.calculate_obv(df['Close'], df['Volume'])
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()
    
    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Price patterns
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    
    return df.fillna(method='bfill').fillna(method='ffill').fillna(0)

# ========== SENTIMENT ANALYSIS ==========
class SentimentAnalyzer:
    """Analyze sentiment from news and social media"""
    
    @staticmethod
    def get_news_sentiment(symbol: str, days: int = 7) -> float:
        """Get sentiment from news (simplified - use NewsAPI in production)"""
        try:
            # This is a placeholder - integrate with NewsAPI, Alpha Vantage, or similar
            # For demo purposes, return neutral sentiment
            return 0.0
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0
    
    @staticmethod
    def analyze_text_sentiment(text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

# ========== ADVANCED MODEL ARCHITECTURES ==========
class Attention(nn.Module):
    """Attention mechanism for sequence models"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = torch.tanh(self.W(H))
        e = self.v(score).squeeze(-1)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)
        context = (H * alpha).sum(dim=1)
        return context, alpha

class EnhancedGRUModel(nn.Module):
    """GRU with dropout and optional attention"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, 
                 dropout: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size) if use_attention else None
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        
        if self.use_attention:
            context, _ = self.attention(out)
            out = context
        else:
            out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LSTMModel(nn.Module):
    """LSTM model for comparison"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class TransformerModel(nn.Module):
    """Transformer-based model"""
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq, batch, features)
        out = self.transformer(x)
        out = out[-1, :, :]  # Last timestep
        out = self.dropout(out)
        return self.fc(out)

# ========== UNCERTAINTY ESTIMATION ==========
class MCDropoutPredictor:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model: nn.Module, n_iterations: int = 100):
        self.model = model
        self.n_iterations = n_iterations
    
    def enable_dropout(self):
        """Enable dropout during inference"""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict_with_uncertainty(self, X: torch.Tensor, scaler: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean, lower_bound, upper_bound)"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_iterations):
                self.enable_dropout()
                pred = self.model(X).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence interval
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        
        return mean_pred.flatten(), lower.flatten(), upper.flatten()

# ========== BACKTESTING ENGINE ==========
class AdvancedBacktester:
    """Comprehensive backtesting with multiple strategies"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def backtest_threshold_strategy(self, actual: np.ndarray, predicted: np.ndarray, 
                                    threshold: float = 0.02) -> Dict:
        """Buy if predicted return > threshold"""
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = [capital]
        
        for i in range(len(predicted) - 1):
            current_price = actual[i]
            predicted_return = (predicted[i+1] - current_price) / current_price
            
            if predicted_return > threshold and position == 0:
                # Buy
                shares = (capital * 0.95) / (current_price * (1 + self.slippage))
                cost = shares * current_price * (1 + self.commission + self.slippage)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    trades.append(('BUY', i, current_price, shares))
            
            elif predicted_return < -threshold and position > 0:
                # Sell
                proceeds = position * current_price * (1 - self.commission - self.slippage)
                capital += proceeds
                trades.append(('SELL', i, current_price, position))
                position = 0
            
            portfolio_values.append(capital + position * actual[i])
        
        # Close final position
        if position > 0:
            capital += position * actual[-1] * (1 - self.commission - self.slippage)
        
        return self._calculate_metrics(capital, portfolio_values, trades)
    
    def backtest_ml_confidence_strategy(self, actual: np.ndarray, predicted: np.ndarray,
                                       confidence_lower: np.ndarray, confidence_upper: np.ndarray) -> Dict:
        """Trade based on confidence intervals"""
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_values = [capital]
        
        for i in range(len(predicted) - 1):
            current_price = actual[i]
            predicted_return = (predicted[i+1] - current_price) / current_price
            confidence_width = confidence_upper[i+1] - confidence_lower[i+1]
            
            # Buy if predicted increase and narrow confidence
            if predicted_return > 0.01 and confidence_width < current_price * 0.1 and position == 0:
                shares = (capital * 0.95) / (current_price * (1 + self.slippage))
                cost = shares * current_price * (1 + self.commission + self.slippage)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    trades.append(('BUY', i, current_price, shares))
            
            # Sell if predicted decrease or wide confidence (high uncertainty)
            elif (predicted_return < -0.01 or confidence_width > current_price * 0.2) and position > 0:
                proceeds = position * current_price * (1 - self.commission - self.slippage)
                capital += proceeds
                trades.append(('SELL', i, current_price, position))
                position = 0
            
            portfolio_values.append(capital + position * actual[i])
        
        if position > 0:
            capital += position * actual[-1] * (1 - self.commission - self.slippage)
        
        return self._calculate_metrics(capital, portfolio_values, trades)
    
    def _calculate_metrics(self, final_capital: float, portfolio_values: List, trades: List) -> Dict:
        """Calculate performance metrics"""
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe Ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0
        
        # Maximum Drawdown
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Win Rate
        winning_trades = sum(1 for t in trades[1::2] if len(trades) > 1)  # Simplified
        win_rate = (winning_trades / (len(trades) / 2)) * 100 if len(trades) > 0 else 0
        
        # Calmar Ratio
        calmar = total_return / (max_dd * 100) if max_dd > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'calmar_ratio': calmar,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'final_capital': final_capital,
            'portfolio_values': portfolio_values.tolist(),
            'trades': trades
        }

# ========== ENSEMBLE MODEL ==========
class EnsemblePredictor:
    """Combine predictions from multiple models"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Weighted ensemble prediction"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred.flatten()

# ========== MAIN PREDICTION FUNCTIONS ==========
def train_advanced_model(symbol: str, start: str, end: str, 
                        model_type: str = 'gru', 
                        epochs: int = 50,
                        batch_size: int = 64,
                        learning_rate: float = 0.001,
                        seq_length: int = 60,
                        use_attention: bool = True,
                        device: str = 'cpu') -> Tuple[nn.Module, MinMaxScaler, Dict]:
    """Train advanced model with all features"""
    
    # Download data
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Add technical indicators
    df = add_all_indicators(df)
    
    # Add sentiment (placeholder)
    sentiment = SentimentAnalyzer.get_news_sentiment(symbol)
    df['Sentiment'] = sentiment
    
    # Select features
    feature_cols = ['Close', 'Returns', 'SMA_5', 'SMA_20', 'EMA_10', 'EMA_50',
                   'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'BB_Position',
                   'Volatility_20', 'Momentum_10', 'ROC_10', 'Sentiment']
    
    # Filter available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare sequences
    data = df[feature_cols].values
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, 0])  # Predict Close price
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    # Train/validation split
    split_idx = int(0.9 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # Convert to tensors
    device = torch.device(device)
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Initialize model
    input_size = X_train.shape[-1]
    if model_type == 'gru':
        model = EnhancedGRUModel(input_size, use_attention=use_attention).to(device)
    elif model_type == 'lstm':
        model = LSTMModel(input_size).to(device)
    elif model_type == 'transformer':
        model = TransformerModel(input_size).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mape': []}
    
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(X_train_t))
        epoch_loss = 0
        num_batches = int(np.ceil(len(X_train_t) / batch_size))
        
        for i in range(num_batches):
            idx = perm[i*batch_size:(i+1)*batch_size]
            batch_X = X_train_t[idx]
            batch_y = y_train_t[idx]
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_loss = criterion(model(X_val_t), y_val_t).item()
            val_rmse = np.sqrt(((y_val - val_pred) ** 2).mean())
            val_mape = (np.abs((y_val - val_pred) / (y_val + 1e-8)).mean()) * 100
        
        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mape'].append(val_mape)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), f'models/{symbol}_{model_type}_best.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, '
                  f'Val RMSE: {val_rmse:.4f}, Val MAPE: {val_mape:.2f}%')
    
    return model, scaler, history

def predict_advanced(symbol: str, start: str, end: str,
                    model_type: str = 'gru',
                    with_uncertainty: bool = True,
                    device: str = 'cpu') -> Dict:
    """Advanced prediction with all features"""
    
    # Load model and scaler
    model_path = f'models/{symbol}_{model_type}_best.pth'
    scaler_path = f'models/{symbol}_scaler.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Download and prepare data
    df = yf.download(symbol, start=start, end=end, progress=False)
    df = add_all_indicators(df)
    
    sentiment = SentimentAnalyzer.get_news_sentiment(symbol)
    df['Sentiment'] = sentiment
    
    feature_cols = ['Close', 'Returns', 'SMA_5', 'SMA_20', 'EMA_10', 'EMA_50',
                   'RSI', 'MACD', 'MACD_Hist', 'BB_Width', 'BB_Position',
                   'Volatility_20', 'Momentum_10', 'ROC_10', 'Sentiment']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare sequences
    seq_length = 60
    data = df[feature_cols].values
    sequences = [data[i:i+seq_length] for i in range(len(data) - seq_length)]
    X = np.array(sequences, dtype=np.float32)
    
    # Load scaler and scale
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(X.shape)
    
    # Load model
    device = torch.device(device)
    input_size = X.shape[-1]
    
    if model_type == 'gru':
        model = EnhancedGRUModel(input_size).to(device)
    elif model_type == 'lstm':
        model = LSTMModel(input_size).to(device)
    elif model_type == 'transformer':
        model = TransformerModel(input_size).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # Predict with uncertainty
    if with_uncertainty:
        mc_predictor = MCDropoutPredictor(model, n_iterations=100)
        mean_pred, lower, upper = mc_predictor.predict_with_uncertainty(X_t, scaler)
    else:
        with torch.no_grad():
            mean_pred = model(X_t).cpu().numpy().flatten()
        lower = upper = None
    
    actual = df['Close'].values[seq_length:]
    
    return {
        'actual': actual,
        'predicted': mean_pred,
        'confidence_lower': lower,
        'confidence_upper': upper,
        'dates': df.index[seq_length:].tolist()
    }

# Example usage
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    print("Training enhanced model...")
    model, scaler, history = train_advanced_model(
        'AAPL', '2020-01-01', '2024-01-01',
        model_type='gru', epochs=20, use_attention=True
    )
    
    # Save scaler
    with open('models/AAPL_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nGenerating predictions...")
    results = predict_advanced('AAPL', '2020-01-01', '2024-01-01', with_uncertainty=True)
    
    print(f"Predictions generated for {len(results['predicted'])} days")
    print(f"Final predicted price: ${results['predicted'][-1]:.2f}")
    print(f"Actual price: ${results['actual'][-1]:.2f}")