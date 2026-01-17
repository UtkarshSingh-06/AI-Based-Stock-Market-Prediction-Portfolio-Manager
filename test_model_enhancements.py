"""
Test script to verify model enhancements work correctly without breaking existing functionality.
"""
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        from predictor import (
            EnhancedGRUModel, GRUWithOptionalAttention, 
            add_technical_indicators, predict_future,
            download_stock, build_sequences
        )
        from model_utils import ensemble_predict
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_technical_indicators():
    """Test enhanced technical indicators."""
    print("\nTesting technical indicators...")
    try:
        import pandas as pd
        import numpy as np
        from predictor import add_technical_indicators
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        df_enhanced = add_technical_indicators(df)
        
        # Check that new indicators are present
        expected_indicators = ['RSI', 'MACD', 'BB_Position', 'BB_Width', 'MA_20', 'MA_50']
        found_indicators = [ind for ind in expected_indicators if ind in df_enhanced.columns]
        
        if len(found_indicators) >= 4:  # At least 4 new indicators
            print(f"[OK] Technical indicators added successfully. Found: {found_indicators}")
            return True
        else:
            print(f"[WARN] Some indicators missing. Found: {found_indicators}")
            return False
    except Exception as e:
        print(f"[FAIL] Technical indicators test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_model():
    """Test enhanced GRU model creation."""
    print("\nTesting enhanced GRU model...")
    try:
        import torch
        from predictor import EnhancedGRUModel
        
        # Test model creation
        model = EnhancedGRUModel(
            input_size=10,
            hidden_size=50,
            num_layers=2,
            dropout=0.2,
            use_attention=False,
            bidirectional=False,
            use_layer_norm=True,
            use_residual=False
        )
        
        # Test forward pass
        batch_size = 32
        seq_len = 60
        input_size = 10
        x = torch.randn(batch_size, seq_len, input_size)
        
        with torch.no_grad():
            output = model(x)
        
        if output.shape == (batch_size, 1):
            print("[OK] Enhanced model works correctly")
            return True
        else:
            print(f"[FAIL] Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        print(f"[FAIL] Enhanced model test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that original model still works."""
    print("\nTesting backward compatibility...")
    try:
        import torch
        from predictor import GRUWithOptionalAttention
        
        # Test original model
        model = GRUWithOptionalAttention(
            input_size=6,
            hidden_size=50,
            num_layers=2,
            dropout=0.0,
            use_attention=False
        )
        
        batch_size = 32
        seq_len = 60
        input_size = 6
        x = torch.randn(batch_size, seq_len, input_size)
        
        with torch.no_grad():
            output = model(x)
        
        if output.shape == (batch_size, 1):
            print("[OK] Original model still works (backward compatible)")
            return True
        else:
            print(f"[FAIL] Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        print(f"[FAIL] Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_ensemble_utils():
    """Test ensemble prediction utilities."""
    print("\nTesting ensemble utilities...")
    try:
        import torch
        import numpy as np
        from model_utils import ensemble_predict, calculate_prediction_confidence
        from predictor import GRUWithOptionalAttention
        
        # Create dummy models
        models = []
        for i in range(3):
            model = GRUWithOptionalAttention(input_size=6, hidden_size=50, num_layers=2)
            models.append(model)
        
        # Test ensemble prediction
        x = torch.randn(10, 60, 6)
        predictions = ensemble_predict(models, x, method='mean')
        
        if predictions.shape == (10, 1):
            print("[OK] Ensemble prediction works")
            
            # Test confidence calculation
            conf = calculate_prediction_confidence(predictions, method='std')
            print("[OK] Confidence calculation works")
            return True
        else:
            print(f"[FAIL] Unexpected ensemble output shape: {predictions.shape}")
            return False
    except Exception as e:
        print(f"[FAIL] Ensemble utilities test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Model Enhancement Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_technical_indicators,
        test_enhanced_model,
        test_backward_compatibility,
        test_ensemble_utils
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Model enhancements are working correctly.")
        return 0
    else:
        print("[WARNING] Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
