"""
GRU Stock Prediction GUI Application
Enhanced with error handling and validation
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
from datetime import datetime
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Directory setup
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------- GRU Model ----------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Validation functions
def validate_symbol(symbol):
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    symbol = symbol.strip().upper()
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol

def validate_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD")

# ---------- Train Model ----------
def train_model(symbol, start, end):
    """Train model with proper error handling."""
    try:
        symbol = validate_symbol(symbol)
        start_date = validate_date(start)
        end_date = validate_date(end)
        
        if not start_date or not end_date:
            raise ValueError("Both start and end dates are required")
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        logger.info(f"Training model for {symbol}")
        
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
        except Exception as e:
            raise ValueError(f"Failed to download data: {e}")
        
        if df.empty or 'Close' not in df.columns:
            raise ValueError(f"No data available for {symbol}")
        
        df = df[['Close']].dropna()
        
        if len(df) < 100:
            raise ValueError(f"Not enough data. Need at least 100 data points, got {len(df)}")
        
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        seq_len = 60
        
        if len(data) < seq_len + 1:
            raise ValueError(f"Not enough data for sequences. Need at least {seq_len + 1} points")
        
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        
        if len(X) == 0:
            raise ValueError("No sequences generated")
        
        X = torch.tensor(np.array(X), dtype=torch.float32).view(-1, seq_len, 1).to(device)
        y = torch.tensor(np.array(y), dtype=torch.float32).to(device)
        
        model = GRUModel(1, 50, 2, 1).to(device)
        criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            logger.info(f"{symbol} Train Epoch {epoch+1}: {loss.item():.6f}")
        
        model_path = MODELS_DIR / f"{symbol}_gru_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        messagebox.showinfo("Training", f"Model saved for {symbol}")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        messagebox.showerror("Training Error", error_msg)

# ---------- Predict Data ----------
def get_predictions(symbol, start, end):
    """Get predictions with proper error handling."""
    try:
        symbol = validate_symbol(symbol)
        validate_date(start)
        validate_date(end)
        
        logger.info(f"Generating predictions for {symbol}")
        
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
        except Exception as e:
            raise ValueError(f"Failed to download data: {e}")
        
        if df.empty or 'Close' not in df.columns:
            raise ValueError(f"No data available for {symbol}")
        
        df = df[['Close']].dropna()
        
        if len(df) < 100:
            raise ValueError(f"Not enough data. Need at least 100 data points, got {len(df)}")
        
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
        seq_len = 60
        
        if len(data) < seq_len + 1:
            raise ValueError(f"Not enough data for sequences")
        
        seqs = [data[i:i+seq_len] for i in range(len(data)-seq_len)]
        
        if len(seqs) == 0:
            raise ValueError("No sequences generated")
        
        X = torch.tensor(np.array(seqs), dtype=torch.float32).view(-1, seq_len, 1).to(device)
        
        model = GRUModel(1, 50, 2, 1).to(device)
        model_path = MODELS_DIR / f"{symbol}_gru_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model for {symbol} not found. Please train the model first.")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")
        
        model.eval()
        with torch.no_grad():
            preds = model(X).cpu().numpy()
        
        preds = scaler.inverse_transform(preds)
        actual = df.values[seq_len:]
        
        logger.info(f"Successfully generated predictions for {symbol}")
        return actual.flatten(), preds.flatten()
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)

# ---------- GUI with Dark Mode & Animation ----------
def run_app():
    root = tk.Tk()
    root.title("📈 GRU Stock Predictor")
    root.geometry("960x780")
    dark = tk.BooleanVar(value=False)

    # Styles
    style = ttk.Style()
    style.theme_use('default')

    def apply_theme():
        bg = '#263238' if dark.get() else '#e1f5fe'
        fg = 'white' if dark.get() else 'black'
        root.configure(bg=bg)
        title.config(bg=bg, fg=fg)
        for lbl in labels:
            lbl.config(bg=bg, fg=fg)
        stock_label.config(bg=bg, fg= '#80CBC4' if dark.get() else '#1B5E20')
        graph_frame.config(bg=bg)

    # Toggle
    toggle = ttk.Checkbutton(root, text="Dark Mode", variable=dark, command=apply_theme)
    toggle.pack(anchor='ne', padx=10, pady=5)

    title = tk.Label(root, text="Stock Price Forecasting with GRU", font=("Segoe UI", 18, 'bold'))
    title.pack(pady=10)

    # Inputs
    labels=[]
    lbl1 = tk.Label(root, text="Symbol (e.g., AAPL):"); lbl1.pack(); labels.append(lbl1)
    combo = ttk.Combobox(root, values=["AMZN","IBM","MSFT","GOOGL","AAPL","TSLA","META","NFLX","NVDA","ADBE"]) ; combo.pack(pady=5)
    stock_label = tk.Label(root, text="", font=("Segoe UI", 12)) ; stock_label.pack(pady=5)

    lbl2 = tk.Label(root, text="Start Date (YYYY-MM-DD):"); lbl2.pack(); labels.append(lbl2)
    start_entry = tk.Entry(root); start_entry.pack(pady=5)
    lbl3 = tk.Label(root, text="End Date (YYYY-MM-DD):"); lbl3.pack(); labels.append(lbl3)
    end_entry = tk.Entry(root); end_entry.pack(pady=5)

    btn_train = ttk.Button(root, text="Train Model", command=lambda:train_model(combo.get(), start_entry.get(), end_entry.get()))
    btn_train.pack(pady=5)
    btn_predict = ttk.Button(root, text="Predict & Animate", command=lambda:on_predict())
    btn_predict.pack(pady=5)

    graph_frame = tk.Frame(root); graph_frame.pack(fill='both', expand=True, padx=10, pady=10)
    apply_theme()

    def on_predict():
        symbol = combo.get().strip()
        s = start_entry.get().strip()
        e = end_entry.get().strip()
        
        if not symbol or not s or not e:
            messagebox.showwarning("Missing Input", "Please complete all inputs")
            return
        
        try:
            validate_symbol(symbol)
            validate_date(s)
            validate_date(e)
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            return
        
        # Stock Info
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            p = info.get('regularMarketPrice') or info.get('currentPrice')
            c = info.get('regularMarketChangePercent') or 0
            if p:
                stock_label.config(text=f"Price: ${p:.2f} | Change: {c:.2f}%")
            else:
                stock_label.config(text="Price information unavailable")
        except Exception as e:
            logger.warning(f"Failed to fetch stock info: {e}")
            stock_label.config(text="Unable to fetch stock info")
        
        # Get Data
        try:
            actual, pred = get_predictions(symbol, s, e)
        except Exception as ex:
            messagebox.showerror("Prediction Error", str(ex))
            return
        # Clear
        for w in graph_frame.winfo_children(): w.destroy()
        fig=plt.Figure(figsize=(8,5), dpi=100)
        ax=fig.add_subplot(111)
        ax.set_title(f"{symbol} Prediction", color='white' if dark.get() else 'black')
        ax.set_facecolor('#37474F' if dark.get() else 'white')
        line1, = ax.plot([],[], color='#80CBC4' if dark.get() else '#1976D2', label='Actual')
        line2, = ax.plot([],[], color='#FF8A65' if dark.get() else '#E53935', label='Predicted')
        ax.legend(facecolor=('#263238' if dark.get() else 'white'), labelcolor=('white' if dark.get() else 'black'))
        canvas=FigureCanvasTkAgg(fig, master=graph_frame); canvas.get_tk_widget().pack(fill='both', expand=True)
        # Animate
        n=len(actual)
        def animate(i=0):
            if i<=n:
                line1.set_data(range(i), actual[:i])
                line2.set_data(range(i), pred[:i])
                ax.relim(); ax.autoscale_view()
                canvas.draw()
                root.after(20, lambda: animate(i+1))
        animate()

    root.mainloop()

# Entry
if __name__ == "__main__":
    try:
        logger.info("Starting Stock Prediction Application")
        run_app()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
