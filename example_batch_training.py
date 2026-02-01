"""
Simple example of batch training multiple stock models.
"""
from batch_trainer import batch_train_stocks, get_trained_models, get_model_info

# Example 1: Train a few stocks
print("Example 1: Training a few stocks")
print("-" * 60)

results = batch_train_stocks(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start="2020-01-01",
    epochs=5,  # Short training for demo
    parallel=True,
    max_workers=2
)

print("\nResults:")
for symbol, result in results.items():
    status = "✓ SUCCESS" if result["status"] == "success" else "✗ FAILED"
    print(f"  {symbol}: {status}")

# Example 2: Check what models are trained
print("\n" + "=" * 60)
print("Example 2: Checking trained models")
print("-" * 60)

trained = get_trained_models()
print(f"\nYou have {len(trained)} trained models:")
for symbol in trained[:10]:  # Show first 10
    info = get_model_info(symbol)
    if info:
        print(f"  - {symbol}: trained on {info['trained_at'][:10]}")

# Example 3: Get detailed info about a model
print("\n" + "=" * 60)
print("Example 3: Detailed model information")
print("-" * 60)

if trained:
    symbol = trained[0]
    info = get_model_info(symbol)
    if info:
        print(f"\nModel: {symbol}")
        print(f"  Model path: {info['model_path']}")
        print(f"  Scaler path: {info['scaler_path']}")
        print(f"  Trained at: {info['trained_at']}")
        print(f"  Metadata: {info.get('metadata', {})}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
