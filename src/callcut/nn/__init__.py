"""Neural network module for call detection models."""

from callcut.nn._models import BaseDetector, TinySegCNN, load_model, save_model
from callcut.nn._registry import (
    get_model,
    list_models,
    register_model,
    unregister_model,
)
