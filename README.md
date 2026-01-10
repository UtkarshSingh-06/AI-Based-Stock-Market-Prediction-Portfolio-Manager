# GRU Stock Price Prediction

A machine learning project for predicting stock prices using Gated Recurrent Units (GRU) neural networks. This project includes data collection, model training, prediction, and a graphical user interface for visualization.

## Features

- **Data Collection**: Downloads historical stock data using Yahoo Finance API
- **GRU Model**: Implements a custom GRU neural network for time series forecasting
- **Training**: Trains models on historical stock data with configurable parameters
- **Prediction**: Generates future price predictions based on trained models
- **Visualization**: GUI application for plotting actual vs predicted prices
- **Pre-trained Models**: Includes trained models for major stocks (AAPL, AMZN, GOOGL, IBM, META, MSFT, NFLX, NVDA, TSLA)

## Installation

1. Clone or download the repository.
2. Create a virtual environment:
   ```bash
   python -m venv gru_env
   ```
3. Activate the virtual environment:
   - Windows: `gru_env\Scripts\activate`
   - Linux/Mac: `source gru_env/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- yfinance
- scikit-learn
- matplotlib
- tkinter

## Usage

### Training a Model

Run the training script for a specific stock:

```python
from predictor import train_and_save_model

# Train model for Apple stock
train_and_save_model('AAPL', start_date='2020-01-01', end_date='2023-12-31')
```

### Making Predictions

Use the predictor module to generate forecasts:

```python
from predictor import load_model_and_predict

# Load model and predict next 30 days
predictions = load_model_and_predict('AAPL', days_ahead=30)
```

### GUI Application

Run the graphical interface:

```bash
python final.py
```

Or use the alternative app:

```bash
python gru_stock_app.py
```

Select a stock symbol from the dropdown and click "Predict" to view the forecast.

## Data

- Historical stock data is downloaded automatically using yfinance
- Data includes OHLCV (Open, High, Low, Close, Volume) information
- Default training period: 2020-03-14 to 2023-10-02
- Sequence length: 60 days for model input

## Models

Pre-trained GRU models are available for the following stocks:
- AAPL (Apple)
- ADBE (Adobe)
- AMZN (Amazon)
- GOOGL (Google)
- IBM (IBM)
- META (Meta Platforms)
- MSFT (Microsoft)
- NFLX (Netflix)
- NVDA (NVIDIA)
- TSLA (Tesla)

Models are saved in `.pth` format and can be loaded for inference.

## Project Structure

```
├── app.py                 # Main application (commented out)
├── final.py               # GUI application for predictions
├── gru_stock_app.py       # Alternative GUI app with training
├── predictor.py           # Core prediction and training utilities
├── requirements.txt       # Python dependencies
├── data/                  # Directory for cached stock data
├── models/                # Directory for saved models
├── *.csv                  # Historical stock data files
├── *_gru_model.pth        # Pre-trained model files
└── README.md              # This file
```

## Configuration

Key parameters can be adjusted in the code:
- `seq_length`: Number of days used for prediction input (default: 60)
- `hidden_size`: GRU hidden layer size (default: 50)
- `num_layers`: Number of GRU layers (default: 2)
- `epochs`: Training epochs (default: 10)

## License

This project is licensed under the MIT License.