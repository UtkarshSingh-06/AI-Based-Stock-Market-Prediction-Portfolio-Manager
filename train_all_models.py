"""
Simple script to train all models at once.
Usage: python train_all_models.py
"""
import logging
from batch_trainer import train_all_available_stocks, batch_train_stocks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("="*60)
    print("Batch Model Training")
    print("="*60)
    print("\nThis will train models for multiple stocks.")
    print("You can customize the stocks list in batch_trainer.py\n")
    
    # Option 1: Train all common stocks
    # results = train_all_available_stocks(epochs=10, max_workers=3)
    
    # Option 2: Train specific stocks
    custom_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "NFLX", "ADBE", "IBM"
    ]
    
    print(f"Training models for: {', '.join(custom_stocks)}")
    print("This may take a while...\n")
    
    results = batch_train_stocks(
        symbols=custom_stocks,
        start="2018-01-01",
        end=None,
        epochs=10,
        seq_len=60,
        use_attention=False,
        max_workers=3,
        parallel=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    successful = []
    failed = []
    
    for symbol, result in results.items():
        if result["status"] == "success":
            successful.append(symbol)
            print(f"✓ {symbol}: SUCCESS")
        else:
            failed.append(symbol)
            print(f"✗ {symbol}: FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\nTotal: {len(results)} models")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully trained models: {', '.join(successful)}")
    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
