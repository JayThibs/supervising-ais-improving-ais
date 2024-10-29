from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
import numpy as np
from collections import defaultdict

class DivergenceMetrics:
    """Compute divergence metrics between model outputs."""
    
    def __init__(self):
        # Initialize metrics dictionary
        self.metrics = {
            "kl_divergence": self.kl_divergence,
            # Add other metrics here as needed
        }
    
    def kl_divergence(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence between two distributions."""
        # Ensure inputs require grad
        if not logits_1.requires_grad:
            logits_1.requires_grad_(True)
        if not logits_2.requires_grad:
            logits_2.requires_grad_(True)
        
        # Convert logits to probabilities while maintaining gradients
        probs_1 = F.softmax(logits_1, dim=-1)
        probs_2 = F.softmax(logits_2, dim=-1)
        
        print(f"Probs requires_grad: {probs_1.requires_grad}")
        
        # Compute KL divergence
        kl_div = torch.sum(probs_1 * (torch.log(probs_1 + 1e-10) - torch.log(probs_2 + 1e-10)), dim=-1)
        
        print(f"KL div requires_grad: {kl_div.requires_grad}")
        
        if attention_mask is not None:
            # Apply attention mask
            kl_div = kl_div * attention_mask
            # Average over non-padded tokens
            kl_div = kl_div.sum() / (attention_mask.sum() + 1e-10)
        else:
            kl_div = kl_div.mean()
        
        print(f"Final KL div requires_grad: {kl_div.requires_grad}")
        print(f"Final KL div grad_fn: {kl_div.grad_fn}")
        
        return kl_div
    
    def compute_all_metrics(
        self,
        outputs_1: Dict[str, torch.Tensor],
        outputs_2: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all configured metrics between model outputs."""
        
        # Get logits and mask
        logits_1 = outputs_1["logits"]
        logits_2 = outputs_2["logits"]
        attention_mask = inputs["attention_mask"]
        
        # Ensure logits match the attention mask size
        if logits_1.size(1) != attention_mask.size(1):
            # Trim logits to match attention mask if needed
            logits_1 = logits_1[:, :attention_mask.size(1), :]
            logits_2 = logits_2[:, :attention_mask.size(1), :]
        
        metrics = {}
        
        # Compute each configured metric
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(
                logits_1,
                logits_2,
                attention_mask=attention_mask
            )  # Don't call .item() here
        
        return metrics
