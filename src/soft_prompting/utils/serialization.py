from typing import Any, Dict, List, Union
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def serialize_for_json(obj: Any, max_prob_length: int = 100) -> Any:
    """
    Convert objects to JSON-serializable format with optimizations.
    
    Args:
        obj: Object to serialize
        max_prob_length: Maximum length of probability arrays to store
    """
    if isinstance(obj, (np.ndarray, np.generic)):
        # Truncate and downsample large probability arrays
        if obj.size > max_prob_length:
            # Take evenly spaced samples
            indices = np.linspace(0, obj.size - 1, max_prob_length, dtype=int)
            obj = obj[indices]
        return obj.tolist()
    
    elif isinstance(obj, torch.Tensor):
        # Detach, move to CPU, and convert to numpy first
        obj = obj.detach().cpu().numpy()
        return serialize_for_json(obj, max_prob_length)
    
    elif isinstance(obj, dict):
        # Optimize large dictionaries
        return {
            k: serialize_for_json(v, max_prob_length) 
            for k, v in obj.items()
            if not (k.endswith('_probs') and isinstance(v, (np.ndarray, torch.Tensor)) and v.size > max_prob_length * 10)
        }
    
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item, max_prob_length) for item in obj]
    
    elif isinstance(obj, Path):
        return str(obj)
    
    return obj

def safe_json_dump(obj: Dict, path: Path) -> None:
    """Safely dump object to JSON file."""
    serialized = serialize_for_json(obj)
    with open(path, 'w') as f:
        json.dump(serialized, f, indent=2)
