"""Training module for scratch detection."""
from .trainer import train_model, get_optimizer_for_model
from .early_stopping import EarlyStopping

__all__ = [
    'train_model',
    'get_optimizer_for_model',
    'EarlyStopping'
]
