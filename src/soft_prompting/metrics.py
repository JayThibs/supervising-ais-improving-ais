import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_kl_divergence(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute KL divergence between two sets of logits.
    
    Args:
        logits_1: Logits from first model [batch_size, seq_len, vocab_size]
        logits_2: Logits from second model [batch_size, seq_len, vocab_size] 
        temperature: Temperature for softmax
        
    Returns:
        torch.Tensor: KL divergence loss
    """
    probs_1 = F.softmax(logits_1 / temperature, dim=-1)
    probs_2 = F.softmax(logits_2 / temperature, dim=-1)
    
    kl_div = F.kl_div(
        F.log_softmax(logits_1 / temperature, dim=-1),
        probs_2,
        reduction='batchmean',
        log_target=False
    )
    return kl_div

def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from loss."""
    return torch.exp(loss)

def compute_metrics(
    model_outputs_1: Dict[str, torch.Tensor],
    model_outputs_2: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute metrics between two model outputs.
    
    Returns dict with:
        - kl_divergence
        - perplexity_1  
        - perplexity_2
    """
    metrics = {}
    
    # KL divergence
    kl_div = compute_kl_divergence(
        model_outputs_1["logits"],
        model_outputs_2["logits"]
    )
    metrics["kl_divergence"] = kl_div.item()
    
    # Perplexities
    metrics["perplexity_1"] = compute_perplexity(
        model_outputs_1["loss"]
    ).item()
    metrics["perplexity_2"] = compute_perplexity(
        model_outputs_2["loss"]
    ).item()
    
    return metrics