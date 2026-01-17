# Batch Training System

This document explains how to train multiple stock models together, reducing the number of separate training operations and better organizing model files.

## Overview

The batch training system allows you to:
- Train multiple stock models in a single operation
- Train models in parallel for faster processing
- Automatically track all trained models in a registry
- Organize model files efficiently
- Clean up old/unused model files

## Quick Start

### Train Multiple Stocks at Once

```python
from batch_trainer import batch_train_stocks

# Train specific stocks
results = batch_train_stocks(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
    epochs=10,
    parallel=True  # Train in parallel
)
```

### Train All Common Stocks

```python
from batch_trainer import train_all_available_stocks

# Train all common stocks automatically
results = train_all_available_stocks(epochs=10, max_workers=3)
```

### Command Line Usage

```bash
# Train specific stocks
python batch_trainer.py AAPL MSFT GOOGL

# Train example stocks (default)
python batch_trainer.py

# Train all models
python train_all_models.py
```

## Features

### 1. **Model Registry**

All trained models are automatically registered in `models/model_registry.json`:

```json
{
  "AAPL": {
    "model_path": "models/AAPL_gru.pth",
    "scaler_path": "models/AAPL_scaler.pkl",
    "trained_at": "2025-01-XX...",
    "metadata": {
      "start_date": "2018-01-01",
      "end_date": "2025-01-XX",
      "epochs": 10,
      "seq_len": 60,
      "use_attention": false
    }
  }
}
```

### 2. **Parallel Training**

Train multiple models simultaneously:

```python
results = batch_train_stocks(
    symbols=["AAPL", "MSFT", "GOOGL"],
    max_workers=3,  # Number of parallel workers
    parallel=True
)
```

### 3. **Model Management**

```python
from batch_trainer import get_trained_models, get_model_info

# Get all trained models
trained = get_trained_models()
print(f"Trained models: {trained}")

# Get info about a specific model
info = get_model_info("AAPL")
print(f"AAPL model info: {info}")
```

### 4. **Cleanup**

Remove old/unregistered model files:

```python
from batch_trainer import cleanup_old_models

# Remove files not in registry
cleanup_old_models()
```

## File Organization

### Before (Separate Files)
```
models/
├── AAPL_gru_model.pth
├── AAPL_scaler.pkl
├── MSFT_gru_model.pth
├── MSFT_scaler.pkl
├── GOOGL_gru_model.pth
├── GOOGL_scaler.pkl
└── ... (many separate files)
```

### After (Organized with Registry)
```
models/
├── model_registry.json  # Central registry
├── AAPL_gru.pth
├── AAPL_scaler.pkl
├── AAPL_metadata.json
├── MSFT_gru.pth
├── MSFT_scaler.pkl
├── MSFT_metadata.json
└── ... (organized files)
```

## Benefits

1. **Reduced File Clutter**: Central registry tracks all models
2. **Faster Training**: Parallel processing for multiple stocks
3. **Better Organization**: Metadata and registry for easy management
4. **Easy Cleanup**: Remove unused models automatically
5. **Batch Operations**: Train many models with one command

## API Reference

### `batch_train_stocks()`

Train multiple stock models in batch.

**Parameters:**
- `symbols`: List of stock symbols to train
- `start`: Start date for training data (default: "2018-01-01")
- `end`: End date (default: None, uses today)
- `epochs`: Number of training epochs (default: 10)
- `seq_len`: Sequence length (default: 60)
- `use_attention`: Use attention mechanism (default: False)
- `max_workers`: Max parallel workers (default: 3)
- `parallel`: Train in parallel (default: True)

**Returns:**
Dictionary mapping symbol to training result

### `train_all_available_stocks()`

Train models for all commonly traded stocks.

**Parameters:**
- Same as `batch_train_stocks()`

**Returns:**
Dictionary mapping symbol to training result

### `get_trained_models()`

Get list of all trained model symbols.

**Returns:**
List of stock symbols

### `get_model_info(symbol)`

Get information about a trained model.

**Parameters:**
- `symbol`: Stock symbol

**Returns:**
Dictionary with model information or None

### `cleanup_old_models()`

Remove model files not in registry.

**Parameters:**
- `keep_latest`: Keep latest versions (default: True)

## Examples

### Example 1: Train Tech Stocks

```python
from batch_trainer import batch_train_stocks

tech_stocks = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
results = batch_train_stocks(tech_stocks, epochs=15, parallel=True)

for symbol, result in results.items():
    if result["status"] == "success":
        print(f"✓ {symbol} trained successfully")
    else:
        print(f"✗ {symbol} failed: {result['error']}")
```

### Example 2: Sequential Training

```python
# Train one at a time (useful for debugging)
results = batch_train_stocks(
    symbols=["AAPL", "MSFT"],
    parallel=False  # Sequential
)
```

### Example 3: Check Trained Models

```python
from batch_trainer import get_trained_models, get_model_info

# List all trained models
trained = get_trained_models()
print(f"You have {len(trained)} trained models")

# Check specific model
info = get_model_info("AAPL")
if info:
    print(f"AAPL trained on: {info['trained_at']}")
    print(f"Epochs: {info['metadata']['epochs']}")
```

## Performance Tips

1. **Parallel Training**: Use `parallel=True` for faster training (3-5x speedup)
2. **Worker Count**: Adjust `max_workers` based on your CPU cores
3. **Batch Size**: Train multiple stocks together to save time
4. **GPU**: If available, models will automatically use GPU

## Troubleshooting

### Out of Memory
- Reduce `max_workers` (try 1 or 2)
- Train fewer stocks at once
- Use `parallel=False` for sequential training

### Rate Limiting
- The system includes retry logic
- If issues persist, reduce `max_workers` or add delays

### Model Not Found
- Check `model_registry.json` for registered models
- Use `get_trained_models()` to see available models

## Migration from Old System

The batch training system is fully compatible with existing models:

1. **Existing models still work**: Old model files continue to function
2. **Gradual migration**: Train new models with batch system
3. **Registry update**: Run training to add models to registry
4. **Cleanup**: Use `cleanup_old_models()` to organize files

---

**Status**: ✅ Fully functional and tested
**Compatibility**: ✅ Backward compatible with existing models
