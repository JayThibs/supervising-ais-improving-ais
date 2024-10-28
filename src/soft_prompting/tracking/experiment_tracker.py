from typing import Dict, Optional, Any
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import torch
import wandb

from ..config.configs import ExperimentConfig

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Track and log experiment progress and results."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        use_wandb: bool = False,
        project_name: Optional[str] = None
    ):
        self.config = config
        self.experiment_dir = config.output_dir
        self.use_wandb = use_wandb
        
        # Setup directories
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        
        for dir_path in [self.logs_dir, self.metrics_dir, self.artifacts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project=project_name or "soft-prompting",
                name=config.name,
                config=self._get_wandb_config()
            )
            
        # Initialize metrics history
        self.metrics_history = []
        
    def _get_wandb_config(self) -> Dict:
        """Convert experiment config to wandb format."""
        return {
            "model_1": self.config.model_1_name,
            "model_2": self.config.model_2_name,
            "training": vars(self.config.training),
            "generation": vars(self.config.generation),
            "data": vars(self.config.data)
        }
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """Log training/evaluation metrics."""
        # Add timestamp and step
        logged_metrics = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **{f"{prefix}{k}": v for k, v in metrics.items()}
        }
        
        # Save to history
        self.metrics_history.append(logged_metrics)
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(logged_metrics, step=step)
            
        # Save metrics file
        metrics_file = self.metrics_dir / f"metrics_{step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(logged_metrics, f, indent=2)
            
    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        data: Any,
        metadata: Optional[Dict] = None
    ):
        """Log experiment artifact (model checkpoint, dataset, etc)."""
        artifact_dir = self.artifacts_dir / artifact_type
        artifact_dir.mkdir(exist_ok=True)
        
        # Save artifact
        artifact_path = artifact_dir / name
        if isinstance(data, torch.nn.Module):
            torch.save(data.state_dict(), artifact_path)
        elif isinstance(data, (dict, list)):
            with open(artifact_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            torch.save(data, artifact_path)
            
        # Save metadata if provided
        if metadata:
            meta_path = artifact_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Log to wandb if enabled
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata
            )
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
            
    def save_experiment_summary(self):
        """Save experiment summary with all metrics and metadata."""
        summary = {
            "config": vars(self.config),
            "metrics_history": self.metrics_history,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
