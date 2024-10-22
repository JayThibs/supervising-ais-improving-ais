import os
import random
import torch
import numpy as np
from typing import Dict, Optional

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    soft_prompt: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: object,
    metrics: Dict[str, float],
    path: str
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "soft_prompt": soft_prompt.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "metrics": metrics
    }, path)

def load_checkpoint(path: str) -> Dict:
    """Load training checkpoint."""
    return torch.load(path)