from typing import Dict, Optional, List
import wandb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .logging import WandbLogger

class DivergenceLogger(WandbLogger):
    """Specialized logger for divergence experiments."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.divergence_history = []
        
    def log_divergence_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        examples: Optional[List[Dict]] = None
    ):
        """Log divergence metrics with optional examples."""
        # Add to history
        self.divergence_history.append({
            "step": step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create visualizations
        if examples:
            fig = self._create_divergence_plot(examples)
            metrics["divergence_plot"] = wandb.Image(fig)
            plt.close(fig)
            
        # Log to wandb
        self.log_metrics(metrics, step)
        
    def _create_divergence_plot(self, examples: List[Dict]) -> plt.Figure:
        """Create visualization of divergence patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # KL divergence distribution
        divergences = [ex["metrics"]["kl_divergence"] for ex in examples]
        sns.histplot(divergences, ax=ax1)
        ax1.set_title("KL Divergence Distribution")
        
        # Token disagreement heatmap
        disagreements = np.array([
            list(ex["metrics"]["disagreement_positions"].values())
            for ex in examples
        ])
        sns.heatmap(disagreements[:10], ax=ax2)
        ax2.set_title("Token Disagreement Patterns")
        
        return fig
    
    def log_generation_examples(
        self,
        examples: List[Dict],
        step: int,
        num_examples: int = 5
    ):
        """Log interesting generation examples."""
        # Sort by divergence
        sorted_examples = sorted(
            examples,
            key=lambda x: x["metrics"]["kl_divergence"],
            reverse=True
        )
        
        # Create table
        table = wandb.Table(columns=[
            "Prompt", "Generation 1", "Generation 2", "KL Divergence"
        ])
        
        for ex in sorted_examples[:num_examples]:
            table.add_data(
                ex["prompt"],
                ex["generation_1"],
                ex["generation_2"],
                ex["metrics"]["kl_divergence"]
            )
            
        self.run.log({f"generation_examples_{step}": table})
    
    def save_divergence_history(self, output_path: Path):
        """Save complete divergence history."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.divergence_history, f, indent=2)
