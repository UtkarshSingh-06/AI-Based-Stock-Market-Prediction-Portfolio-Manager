"""
Simple GRU Stock Prediction GUI Application
Enhanced with error handling and validation
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Directory setup
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# GRU model definition
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Prediction and visualization logic
def predict_and_plot(symbol):
    """Predict stock prices with proper error handling."""
    try:
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        logger.info(f"Predicting for {symbol}")
        
        # Download data with error handling
        try:
            df = yf.download(symbol, start="2020-01-01", end="2024-04-01", progress=False)
        except Exception as e:
            raise ValueError(f"Failed to download data: {e}")
        
        if df.empty or 'Close' not in df.columns:
            raise ValueError(f"No data available for {symbol}")
        
        close_prices = df[['Close']].dropna()
        
        if len(close_prices) < 100:
            raise ValueError(f"Not enough data for {symbol}. Need at least 100 data points.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        seq_length = 60
        if len(scaled_data) < seq_length:
            raise ValueError(f"Not enough data. Need at least {seq_length} data points.")
        
        sequences = []
        for i in range(len(scaled_data) - seq_length):
            sequences.append(scaled_data[i:i+seq_length])

        if len(sequences) == 0:
            raise ValueError("No sequences generated from data")

        sequences = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

        model = GRUModel(1, 50, 2, 1).to(device)
        model_path = MODELS_DIR / f"{symbol}_gru_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model for {symbol} not found at {model_path}")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")

        model.eval()
        with torch.no_grad():
            predictions = model(sequences.unsqueeze(-1)).cpu().numpy()

        predictions = scaler.inverse_transform(predictions)
        actual_prices = close_prices.values[seq_length:]

        logger.info(f"Successfully generated predictions for {symbol}")
        return actual_prices, predictions
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        messagebox.showerror("Prediction Error", error_msg)
        return None, None

# GUI App
def run_app():
    def on_predict():
        symbol = combo.get()
        if not symbol:
            messagebox.showwarning("Input Error", "Please select a stock symbol")
            return

        actual, predicted = predict_and_plot(symbol)
        if actual is None:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(actual, label="Actual Prices", color="blue")
        ax.plot(predicted, label="Predicted Prices", color="red")
        ax.set_title(f"{symbol} Stock Price Prediction")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    root = tk.Tk()
    root.title("Stock Price Prediction App (GRU)")
    root.geometry("700x600")

    tk.Label(root, text="Select Stock Symbol", font=("Arial", 12)).pack(pady=10)
    combo = ttk.Combobox(root, values=["AMZN", "IBM", "MSFT"], font=("Arial", 12))
    combo.pack(pady=5)

    predict_button = tk.Button(root, text="Predict & Show Graph", command=on_predict,
                               font=("Arial", 12), bg="#4CAF50", fg="white")
    predict_button.pack(pady=10)

    global frame
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    root.mainloop()

# Run the app
if __name__ == "__main__":
    try:
        logger.info("Starting Stock Prediction Application")
        run_app()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
