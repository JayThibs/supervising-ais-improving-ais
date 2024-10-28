from typing import Dict, Optional
import torch
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def save_checkpoint(
    path: Path,
    soft_prompt: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: object,
    global_step: int,
    best_divergence: float,
    additional_data: Optional[Dict] = None
):
    """Save training checkpoint with metadata."""
    checkpoint = {
        "soft_prompt": soft_prompt.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best_divergence": best_divergence,
        "config": vars(config),
        "timestamp": datetime.now().isoformat()
    }
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
        
    if additional_data:
        checkpoint.update(additional_data)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    # Save human-readable metadata
    meta_path = path.with_suffix('.meta.json')
    meta = {
        "global_step": global_step,
        "best_divergence": best_divergence,
        "timestamp": checkpoint["timestamp"],
        "config": vars(config)
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
        
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(path: Path) -> Dict:
    """Load training checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path, map_location='cpu')
    logger.info(f"Loaded checkpoint from {path}")
    
    return checkpoint

def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Get path to latest checkpoint in directory."""
    checkpoints = list(checkpoint_dir.glob("checkpoint-*.pt"))
    if not checkpoints:
        return None
        
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.stem.split('-')[1]))
    return checkpoints[-1]
