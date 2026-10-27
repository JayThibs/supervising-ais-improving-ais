import random
import torch
import numpy as np
from typing import Optional

from src.soft_prompting.utils.device_utils import get_device

def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False
) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Integer seed for random number generation
        deterministic: If True, use deterministic algorithms for CUDA operations
        benchmark: If True, use benchmark mode for CUDA operations
            Note: benchmark=True may improve performance but reduces reproducibility
    
    Example:
        >>> set_seed(42)  # Basic usage
        >>> set_seed(42, deterministic=True, benchmark=False)  # Strict reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = get_device()
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    elif device == 'mps':
        # MPS-specific settings if needed
        pass
