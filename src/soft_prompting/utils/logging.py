import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import wandb
import json
from datetime import datetime

def setup_experiment_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logging configuration for experiments."""
    # Create logger
    logger = logging.getLogger("soft_prompting")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

class WandbLogger:
    """Wrapper for wandb logging with additional features."""
    
    def __init__(
        self,
        project: str,
        name: str,
        config: Dict,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            reinit=True
        )
        
        self.step = 0
        self.metrics_history = []
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """Log metrics to wandb with history tracking."""
        if step is None:
            step = self.step
            self.step += 1
            
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Log to wandb
        wandb.log(metrics, step=step, commit=commit)
        
        # Save to history
        self.metrics_history.append({
            "step": step,
            **metrics
        })
        
    def log_artifact(
        self,
        artifact_path: Path,
        name: str,
        type: str,
        metadata: Optional[Dict] = None
    ):
        """Log artifact to wandb."""
        artifact = wandb.Artifact(
            name=name,
            type=type,
            metadata=metadata
        )
        artifact.add_file(str(artifact_path))
        self.run.log_artifact(artifact)
        
    def log_model(
        self,
        model_path: Path,
        name: str,
        metadata: Optional[Dict] = None
    ):
        """Log model checkpoint as artifact."""
        self.log_artifact(
            artifact_path=model_path,
            name=name,
            type="model",
            metadata=metadata
        )
        
    def log_dataset(
        self,
        dataset_path: Path,
        name: str,
        metadata: Optional[Dict] = None
    ):
        """Log dataset as artifact."""
        self.log_artifact(
            artifact_path=dataset_path,
            name=name,
            type="dataset",
            metadata=metadata
        )
        
    def save_metrics_history(self, output_path: Path):
        """Save metrics history to file."""
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def finish(self):
        """Finish the wandb run."""
        wandb.finish()
