import random
import torch
import numpy as np
from typing import Optional

def set_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False
) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
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
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        # Set CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
