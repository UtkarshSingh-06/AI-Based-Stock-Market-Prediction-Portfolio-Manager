"""
Model utility functions for ensemble predictions and model management.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def ensemble_predict(models: List[nn.Module], X: torch.Tensor, 
                    method: str = 'mean') -> np.ndarray:
    """
    Make ensemble predictions from multiple models.
    
    Args:
        models: List of trained models
        X: Input tensor (batch, seq_len, features)
        method: 'mean', 'median', or 'weighted'
    
    Returns:
        Ensemble predictions
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.array(predictions)  # (n_models, batch_size, 1)
    
    if method == 'mean':
        ensemble_pred = np.mean(predictions, axis=0)
    elif method == 'median':
        ensemble_pred = np.median(predictions, axis=0)
    elif method == 'weighted':
        # Simple weighted average (can be improved with validation performance)
        weights = np.ones(len(models)) / len(models)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_pred


def calculate_prediction_confidence(predictions: np.ndarray, 
                                  method: str = 'std') -> np.ndarray:
    """
    Calculate confidence scores for predictions.
    
    Args:
        predictions: Array of predictions (can be from ensemble)
        method: 'std' for standard deviation, 'range' for min-max range
    
    Returns:
        Confidence scores (lower is more confident)
    """
    if method == 'std':
        if predictions.ndim > 1:
            # For ensemble predictions
            return np.std(predictions, axis=0)
        else:
            return np.zeros_like(predictions)
    elif method == 'range':
        if predictions.ndim > 1:
            return np.ptp(predictions, axis=0)  # Peak-to-peak (range)
        else:
            return np.zeros_like(predictions)
    else:
        raise ValueError(f"Unknown confidence method: {method}")


def adaptive_learning_rate_scheduler(optimizer, epoch: int, 
                                    initial_lr: float = 0.001,
                                    decay_factor: float = 0.95,
                                    min_lr: float = 1e-6) -> float:
    """
    Adaptive learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch
        initial_lr: Initial learning rate
        decay_factor: Factor to decay LR by
        min_lr: Minimum learning rate
    
    Returns:
        Current learning rate
    """
    new_lr = initial_lr * (decay_factor ** epoch)
    new_lr = max(new_lr, min_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    return new_lr


def model_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float,
                    filepath: str, save_best: bool = True, 
                    best_loss: Optional[float] = None) -> bool:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        save_best: Whether to save only if best
        best_loss: Best loss so far
    
    Returns:
        True if checkpoint was saved, False otherwise
    """
    if save_best and best_loss is not None:
        if loss >= best_loss:
            return False
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath} (epoch {epoch}, loss {loss:.6f})")
    return True
