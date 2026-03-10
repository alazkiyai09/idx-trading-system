import functools
import logging
import random
import os
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)

def safe_calculation(default_return=None):
    """Decorator to catch exceptions and return a default value."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if tf is not None:
        tf.random.set_seed(seed)
