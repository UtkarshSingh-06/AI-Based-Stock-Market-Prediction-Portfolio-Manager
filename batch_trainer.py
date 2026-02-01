"""
Batch Training System for Multiple Stock Models
Trains multiple models together and organizes them efficiently.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import concurrent.futures

# Try to import tqdm, fallback to simple progress if not available
try:
    from tqdm import tqdm
except ImportError:
    # Simple progress bar fallback
    def tqdm(iterable, desc="", **kwargs):
        print(f"{desc}...")
        return iterable

from predictor import train, download_stock, add_technical_indicators

# Import config with fallback
try:
    from config import MODELS_DIR, DATA_DIR
except ImportError:
    # Fallback if config not available
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# Model registry file to track all trained models
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"


def load_model_registry() -> Dict:
    """Load the model registry from JSON file."""
    if MODEL_REGISTRY_FILE.exists():
        try:
            with open(MODEL_REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
            return {}
    return {}


def save_model_registry(registry: Dict):
    """Save the model registry to JSON file."""
    try:
        with open(MODEL_REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
        logger.info(f"Model registry saved to {MODEL_REGISTRY_FILE}")
    except Exception as e:
        logger.error(f"Failed to save model registry: {e}")


def update_model_registry(symbol: str, model_path: str, scaler_path: str, 
                         metadata: Dict):
    """Update the model registry with a new model entry."""
    registry = load_model_registry()
    
    registry[symbol] = {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "trained_at": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    save_model_registry(registry)
    logger.info(f"Updated registry for {symbol}")


def train_single_stock(symbol: str, start: str = "2018-01-01", 
                      end: Optional[str] = None, epochs: int = 10,
                      seq_len: int = 60, use_attention: bool = False) -> Dict:
    """
    Train a single stock model and return training results.
    
    Returns:
        Dict with training results and status
    """
    try:
        logger.info(f"Starting training for {symbol}")
        
        model_path, scaler_path = train(
            symbol=symbol,
            start=start,
            end=end,
            seq_len=seq_len,
            epochs=epochs,
            use_attention=use_attention,
            save=True
        )
        
        # Update registry
        metadata = {
            "start_date": start,
            "end_date": end or datetime.now().strftime("%Y-%m-%d"),
            "epochs": epochs,
            "seq_len": seq_len,
            "use_attention": use_attention
        }
        
        update_model_registry(symbol, model_path, scaler_path, metadata)
        
        return {
            "symbol": symbol,
            "status": "success",
            "model_path": model_path,
            "scaler_path": scaler_path
        }
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "status": "failed",
            "error": str(e)
        }


def batch_train_stocks(symbols: List[str], start: str = "2018-01-01",
                      end: Optional[str] = None, epochs: int = 10,
                      seq_len: int = 60, use_attention: bool = False,
                      max_workers: int = 3, parallel: bool = True) -> Dict[str, Dict]:
    """
    Train multiple stock models in batch.
    
    Args:
        symbols: List of stock symbols to train
        start: Start date for training data
        end: End date for training data
        epochs: Number of training epochs
        seq_len: Sequence length
        use_attention: Whether to use attention mechanism
        max_workers: Maximum number of parallel workers
        parallel: Whether to train in parallel (True) or sequential (False)
    
    Returns:
        Dictionary mapping symbol to training result
    """
    logger.info(f"Starting batch training for {len(symbols)} stocks: {symbols}")
    
    results = {}
    
    if parallel and len(symbols) > 1:
        # Parallel training
        logger.info(f"Training {len(symbols)} models in parallel (max_workers={max_workers})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training tasks
            future_to_symbol = {
                executor.submit(
                    train_single_stock, 
                    symbol, start, end, epochs, seq_len, use_attention
                ): symbol 
                for symbol in symbols
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(symbols), desc="Training models") as pbar:
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        status = "✓" if result["status"] == "success" else "✗"
                        pbar.set_postfix_str(f"{symbol}: {status}")
                    except Exception as e:
                        logger.error(f"Exception for {symbol}: {e}")
                        results[symbol] = {
                            "symbol": symbol,
                            "status": "failed",
                            "error": str(e)
                        }
                    pbar.update(1)
    else:
        # Sequential training
        logger.info(f"Training {len(symbols)} models sequentially")
        
        for symbol in tqdm(symbols, desc="Training models"):
            result = train_single_stock(symbol, start, end, epochs, seq_len, use_attention)
            results[symbol] = result
    
    # Summary
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = len(results) - successful
    
    logger.info(f"Batch training complete: {successful} successful, {failed} failed")
    
    return results


def get_trained_models() -> List[str]:
    """Get list of all trained model symbols from registry."""
    registry = load_model_registry()
    return list(registry.keys())


def get_model_info(symbol: str) -> Optional[Dict]:
    """Get information about a trained model."""
    registry = load_model_registry()
    return registry.get(symbol.upper())


def train_all_available_stocks(start: str = "2018-01-01", 
                               end: Optional[str] = None,
                               epochs: int = 10,
                               seq_len: int = 60,
                               use_attention: bool = False,
                               max_workers: int = 3) -> Dict[str, Dict]:
    """
    Train models for all commonly traded stocks.
    
    Returns:
        Dictionary mapping symbol to training result
    """
    # Common stock symbols
    common_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
        "ADBE", "AMD", "INTC", "CRM", "ORCL", "IBM", "JPM", "BAC", "GS",
        "V", "MA", "JNJ", "UNH", "PFE", "WMT", "HD", "NKE", "COST", "DIS"
    ]
    
    logger.info(f"Training models for {len(common_stocks)} common stocks")
    
    return batch_train_stocks(
        symbols=common_stocks,
        start=start,
        end=end,
        epochs=epochs,
        seq_len=seq_len,
        use_attention=use_attention,
        max_workers=max_workers,
        parallel=True
    )


def cleanup_old_models(keep_latest: bool = True):
    """Clean up old model files, keeping only the latest versions."""
    registry = load_model_registry()
    
    # Get all model files in models directory
    model_files = list(MODELS_DIR.glob("*_gru.pth"))
    scaler_files = list(MODELS_DIR.glob("*_scaler.pkl"))
    
    # Find files not in registry
    registered_models = set()
    for info in registry.values():
        if "model_path" in info:
            registered_models.add(Path(info["model_path"]).name)
        if "scaler_path" in info:
            registered_models.add(Path(info["scaler_path"]).name)
    
    # Remove unregistered files
    removed = 0
    for file in model_files + scaler_files:
        if file.name not in registered_models:
            try:
                file.unlink()
                removed += 1
                logger.info(f"Removed unregistered file: {file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file.name}: {e}")
    
    logger.info(f"Cleanup complete: removed {removed} unregistered files")


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        # Train specific stocks from command line
        symbols = [s.upper() for s in sys.argv[1:]]
        results = batch_train_stocks(symbols, epochs=10, parallel=True)
    else:
        # Train a few example stocks
        example_stocks = ["AAPL", "MSFT", "GOOGL"]
        print(f"Training models for: {example_stocks}")
        results = batch_train_stocks(example_stocks, epochs=10, parallel=True)
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for symbol, result in results.items():
        status = "SUCCESS" if result["status"] == "success" else "FAILED"
        print(f"{symbol}: {status}")
        if result["status"] == "failed":
            print(f"  Error: {result.get('error', 'Unknown error')}")
