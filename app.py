"""
GRU Stock Price Prediction Application
Enhanced with proper error handling, validation, and best practices
"""








import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import time
import os
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize device early
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Directory setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------- GRU Model ----------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ---------- Validation Functions ----------
def validate_symbol(symbol):
    """Validate stock symbol format."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol

def validate_date(date_str):
    """Validate and parse date string."""
    if not date_str:
        return None
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj > datetime.now():
            raise ValueError("Date cannot be in the future")
        return date_obj
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

def validate_date_range(start, end):
    """Validate date range."""
    if start and end:
        start_date = validate_date(start)
        end_date = validate_date(end)
        if start_date and end_date and start_date >= end_date:
            raise ValueError("Start date must be before end date")
        if start_date and (datetime.now() - start_date).days > 3650:  # 10 years max
            raise ValueError("Date range too large. Maximum 10 years of data.")

# ---------- Safe Download with Retry + Cache ----------
def safe_download(symbol, start, end, retries=5, delay=5):
    """Download stock data safely with retry and local caching."""
    symbol = validate_symbol(symbol)
    start_date = validate_date(start) if start else None
    end_date = validate_date(end) if end else None
    
    if start_date and end_date:
        validate_date_range(start, end)
    
    cache_file = DATA_DIR / f"{symbol}_{start}_{end}.csv"

    # If cached data exists and is recent (less than 1 day old), use it
    if cache_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age < timedelta(days=1):
            logger.info(f"Using cached data for {symbol}")
            try:
                import pandas as pd
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    for attempt in range(retries):
        try:
            logger.info(f"Downloading {symbol} data... (Attempt {attempt+1}/{retries})")
            df = yf.download(symbol, start=start, end=end, progress=False, threads=False, auto_adjust=False)
            if not df.empty and 'Close' in df.columns:
                df.to_csv(cache_file)
                logger.info(f"✅ Data for {symbol} saved to cache.")
                return df
            else:
                logger.warning(f"No data returned for {symbol}, attempt {attempt+1}")
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff

    raise ValueError(f"⚠️ No data found for {symbol}. Check symbol, date range, or internet connection.")


# ---------- Train Model ----------
def train_model(symbol, start, end, epochs=10, seq_len=60, hidden_size=50, num_layers=2):
    """Train GRU model with proper error handling and validation."""
    try:
        symbol = validate_symbol(symbol)
        if not start or not end:
            raise ValueError("Start and end dates are required for training")
        
        logger.info(f"Starting training for {symbol}")
        df = safe_download(symbol, start, end)
        
        if 'Close' not in df.columns:
            raise ValueError("Close price column not found in data")
        
        df = df[['Close']].dropna()
        
        if len(df) < seq_len + 10:
            raise ValueError(f"Not enough data. Need at least {seq_len + 10} data points, got {len(df)}")

        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])

        if len(X) == 0:
            raise ValueError("No sequences generated from data")

        X = torch.tensor(np.array(X), dtype=torch.float32).view(-1, seq_len, 1).to(device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(device)

        model = GRUModel(1, hidden_size, num_layers, 1, dropout=0.2).to(device)
        
        # Use Huber loss for better robustness (falls back to MSE if not available)
        try:
            criterion = nn.HuberLoss(delta=1.0)
        except:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        model.train()
        best_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            current_loss = loss.item()
            logger.info(f"{symbol} Train Epoch {epoch+1}/{epochs}: Loss={current_loss:.6f}")
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        model_path = MODELS_DIR / f"{symbol}_gru_model.pth"
        scaler_path = MODELS_DIR / f"{symbol}_scaler.pkl"
        
        torch.save(model.state_dict(), model_path)
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Model saved to {model_path}")
        messagebox.showinfo("Training Complete", f"Model trained and saved for {symbol}")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        messagebox.showerror("Training Error", error_msg)
        raise


# ---------- Predict Data ----------
def get_predictions(symbol, start, end, seq_len=60, hidden_size=50, num_layers=2):
    """Get predictions with proper error handling and validation."""
    try:
        symbol = validate_symbol(symbol)
        
        logger.info(f"Generating predictions for {symbol}")
        df = safe_download(symbol, start, end)
        
        if 'Close' not in df.columns:
            raise ValueError("Close price column not found in data")
        
        df = df[['Close']].dropna()

        if len(df) < seq_len + 10:
            raise ValueError(f"Not enough data. Need at least {seq_len + 10} data points, got {len(df)}")

        model_path = MODELS_DIR / f"{symbol}_gru_model.pth"
        scaler_path = MODELS_DIR / f"{symbol}_scaler.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {symbol}. Please train the model first.")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found for {symbol}. Please train the model first.")

        # Load scaler
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        data = scaler.transform(df)

        seqs = [data[i:i+seq_len] for i in range(len(data)-seq_len)]
        if len(seqs) == 0:
            raise ValueError("No sequences generated from data")
        
        X = torch.tensor(np.array(seqs), dtype=torch.float32).view(-1, seq_len, 1).to(device)

        model = GRUModel(1, hidden_size, num_layers, 1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds = model(X).cpu().numpy()

        preds = scaler.inverse_transform(preds)
        actual = df.values[seq_len:]
        
        logger.info(f"Generated {len(preds)} predictions for {symbol}")
        return actual.flatten(), preds.flatten()
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


# ---------- GUI with Dark Mode & Animation ----------
def run_app():
    root = tk.Tk()
    root.title("📈 GRU Stock Predictor")
    root.geometry("980x800")
    dark = tk.BooleanVar(value=False)

    # ---------- Style ----------
    style = ttk.Style()
    style.theme_use('default')

    def apply_theme():
        bg = '#263238' if dark.get() else '#e1f5fe'
        fg = 'white' if dark.get() else 'black'
        root.configure(bg=bg)
        title.config(bg=bg, fg=fg)
        for lbl in labels:
            lbl.config(bg=bg, fg=fg)
        stock_label.config(bg=bg, fg='#80CBC4' if dark.get() else '#1B5E20')
        graph_frame.config(bg=bg)

    # ---------- UI Components ----------
    toggle = ttk.Checkbutton(root, text="Dark Mode", variable=dark, command=apply_theme)
    toggle.pack(anchor='ne', padx=10, pady=5)

    title = tk.Label(root, text="Stock Price Forecasting with GRU", font=("Segoe UI", 18, 'bold'))
    title.pack(pady=10)

    labels = []
    lbl1 = tk.Label(root, text="Symbol (e.g., AAPL):"); lbl1.pack(); labels.append(lbl1)
    combo = ttk.Combobox(root, values=["AMZN","IBM","MSFT","GOOGL","AAPL","TSLA","META","NFLX","NVDA","ADBE"])
    combo.pack(pady=5)

    stock_label = tk.Label(root, text="", font=("Segoe UI", 12)); stock_label.pack(pady=5)

    lbl2 = tk.Label(root, text="Start Date (YYYY-MM-DD):"); lbl2.pack(); labels.append(lbl2)
    start_entry = tk.Entry(root); start_entry.pack(pady=5)

    lbl3 = tk.Label(root, text="End Date (YYYY-MM-DD):"); lbl3.pack(); labels.append(lbl3)
    end_entry = tk.Entry(root); end_entry.pack(pady=5)

    def on_train():
        symbol = combo.get().strip()
        s, e = start_entry.get().strip(), end_entry.get().strip()
        
        if not symbol:
            messagebox.showwarning("Missing Input", "Please select a stock symbol.")
            return
        if not s or not e:
            messagebox.showwarning("Missing Input", "Please enter both start and end dates.")
            return
        
        try:
            validate_symbol(symbol)
            validate_date(s)
            validate_date(e)
            validate_date_range(s, e)
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            return
        
        # Run training in a separate thread to avoid blocking UI
        import threading
        def train_thread():
            try:
                train_model(symbol, s, e)
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
        messagebox.showinfo("Training Started", f"Training {symbol} model in background. This may take a few minutes...")
    
    ttk.Button(root, text="Train Model", command=on_train).pack(pady=5)
    ttk.Button(root, text="Predict & Animate", command=lambda: on_predict()).pack(pady=5)

    graph_frame = tk.Frame(root); graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
    apply_theme()

    # ---------- Prediction Logic ----------
    def on_predict():
        symbol = combo.get().strip()
        s, e = start_entry.get().strip(), end_entry.get().strip()
        
        # Validation
        if not symbol:
            messagebox.showwarning("Missing Input", "Please select a stock symbol.")
            return
        if not s:
            messagebox.showwarning("Missing Input", "Please enter a start date.")
            return
        if not e:
            messagebox.showwarning("Missing Input", "Please enter an end date.")
            return
        
        try:
            validate_symbol(symbol)
            validate_date(s)
            validate_date(e)
            validate_date_range(s, e)
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            return

        # Fetch stock info
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            change = info.get('regularMarketChangePercent')
            if price:
                change_str = f" | Change: {change:.2f}%" if change is not None else ""
                stock_label.config(text=f"Price: ${price:.2f}{change_str}")
            else:
                stock_label.config(text="Price information unavailable")
        except Exception as e:
            logger.warning(f"Failed to fetch stock info: {e}")
            stock_label.config(text="Unable to fetch stock info.")

        # Get predictions
        try:
            actual, pred = get_predictions(symbol, s, e)
        except Exception as ex:
            messagebox.showerror("Prediction Error", str(ex))
            return

        # Clear previous graphs
        for w in graph_frame.winfo_children():
            w.destroy()

        # Create new graph
        fig = plt.Figure(figsize=(8,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(f"{symbol} Stock Price Prediction", color='white' if dark.get() else 'black')
        ax.set_facecolor('#37474F' if dark.get() else 'white')
        ax.set_xlabel("Time", color='white' if dark.get() else 'black')
        ax.set_ylabel("Price ($)", color='white' if dark.get() else 'black')
        
        line1, = ax.plot([], [], color='#80CBC4' if dark.get() else '#1976D2', label='Actual', linewidth=2)
        line2, = ax.plot([], [], color='#FF8A65' if dark.get() else '#E53935', label='Predicted', linewidth=2)
        
        ax.legend(facecolor=('#263238' if dark.get() else 'white'),
                  labelcolor=('white' if dark.get() else 'black'))
        ax.tick_params(colors='white' if dark.get() else 'black')
        
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        n = len(actual)

        # Smooth animation that stops at the end
        def animate(i=0):
            if i <= n:
                line1.set_data(range(i), actual[:i])
                line2.set_data(range(i), pred[:i])
                ax.relim()
                ax.autoscale_view()
                canvas.draw()
                root.after(15, lambda: animate(i+1))
            else:
                # Calculate and display metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
                rmse = np.sqrt(mean_squared_error(actual, pred))
                mape = mean_absolute_percentage_error(actual, pred) * 100
                ax.text(0.02, 0.98, f'RMSE: ${rmse:.2f}\nMAPE: {mape:.2f}%', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat' if not dark.get() else '#263238', alpha=0.8),
                       color='black' if not dark.get() else 'white')
                canvas.draw()

        animate()

    root.mainloop()


# ---------- Entry Point ----------
if __name__ == "__main__":
    try:
        logger.info("Starting Stock Prediction Application")
        run_app()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
