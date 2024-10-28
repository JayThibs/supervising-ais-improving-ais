from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
import numpy as np
from collections import defaultdict

class DivergenceMetrics:
    """Compute various divergence metrics between model outputs."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def compute_kl_divergence(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        temperature: float = 1.0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between model outputs.
        
        Args:
            logits_1: Logits from first model [batch, seq_len, vocab]
            logits_2: Logits from second model [batch, seq_len, vocab]
            temperature: Softmax temperature
            mask: Optional attention mask [batch, seq_len]
        """
        probs_1 = F.softmax(logits_1 / temperature, dim=-1)
        probs_2 = F.softmax(logits_2 / temperature, dim=-1)
        
        kl_div = F.kl_div(
            F.log_softmax(logits_1 / temperature, dim=-1),
            probs_2,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # [batch, seq_len]
        
        if mask is not None:
            kl_div = kl_div * mask
            kl_div = kl_div.sum(dim=1) / mask.sum(dim=1)  # Average over sequence
        else:
            kl_div = kl_div.mean(dim=1)
            
        return kl_div.mean()  # Average over batch
        
    def compute_token_divergence(
        self,
        logits_1: torch.Tensor,
        logits_2: torch.Tensor,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute token-level divergence statistics."""
        probs_1 = F.softmax(logits_1, dim=-1)
        probs_2 = F.softmax(logits_2, dim=-1)
        
        # Get top token predictions
        top_tokens_1 = torch.argmax(probs_1, dim=-1)  # [batch, seq_len]
        top_tokens_2 = torch.argmax(probs_2, dim=-1)  # [batch, seq_len]
        
        # Calculate token disagreement
        disagreement = (top_tokens_1 != top_tokens_2)
        if mask is not None:
            disagreement = disagreement & mask.bool()
        
        # Get positions of disagreements
        batch_size, seq_len = disagreement.shape
        disagreement_positions = defaultdict(int)
        
        for b in range(batch_size):
            for pos in range(seq_len):
                if disagreement[b, pos]:
                    token_1 = self.tokenizer.decode([top_tokens_1[b, pos]])
                    token_2 = self.tokenizer.decode([top_tokens_2[b, pos]])
                    orig_token = self.tokenizer.decode([input_ids[b, pos]])
                    key = f"{orig_token} -> {token_1}/{token_2}"
                    disagreement_positions[key] += 1
        
        # Calculate metrics
        total_tokens = mask.sum().item() if mask is not None else (batch_size * seq_len)
        disagreement_rate = disagreement.sum().item() / total_tokens
        
        return {
            "token_disagreement_rate": disagreement_rate,
            "disagreement_positions": dict(disagreement_positions)
        }
    
    def compute_semantic_divergence(
        self,
        texts_1: List[str],
        texts_2: List[str]
    ) -> Dict[str, float]:
        """Compute semantic-level divergence metrics."""
        # This could be expanded with more sophisticated semantic analysis
        # For now, implementing basic length and vocabulary divergence
        
        def get_vocab(texts: List[str]) -> set:
            return set(word for text in texts for word in text.split())
        
        vocab_1 = get_vocab(texts_1)
        vocab_2 = get_vocab(texts_2)
        
        jaccard_sim = len(vocab_1 & vocab_2) / len(vocab_1 | vocab_2)
        
        lengths_1 = [len(text.split()) for text in texts_1]
        lengths_2 = [len(text.split()) for text in texts_2]
        
        length_diff = abs(np.mean(lengths_1) - np.mean(lengths_2))
        
        return {
            "vocab_jaccard_similarity": jaccard_sim,
            "mean_length_difference": length_diff
        }
    
    def compute_all_metrics(
        self,
        model_outputs_1: Dict[str, torch.Tensor],
        model_outputs_2: Dict[str, torch.Tensor],
        input_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute all available divergence metrics.
        Optimized for detecting intervention effects by comparing model_1 (intervention) 
        against model_2 (original base model).
        """
        metrics = {}
        
        # KL divergence from intervention to base model
        kl_div = self.compute_kl_divergence(
            model_outputs_1["logits"],  # intervention model
            model_outputs_2["logits"],  # base model
            mask=input_data.get("attention_mask")
        )
        metrics["kl_divergence"] = kl_div.item()
        
        # Compute asymmetric KL to better detect intervention effects
        reverse_kl = self.compute_kl_divergence(
            model_outputs_2["logits"],  # base model
            model_outputs_1["logits"],  # intervention model
            mask=input_data.get("attention_mask")
        )
        metrics["intervention_divergence"] = kl_div.item() - reverse_kl.item()
        
        # Token-level metrics
        token_metrics = self.compute_token_divergence(
            model_outputs_1["logits"],
            model_outputs_2["logits"],
            input_data["input_ids"],
            input_data.get("attention_mask")
        )
        metrics.update(token_metrics)
        
        # If texts are available, compute semantic metrics
        if "text" in input_data:
            texts_1 = self.tokenizer.batch_decode(
                model_outputs_1["logits"].argmax(dim=-1),
                skip_special_tokens=True
            )
            texts_2 = self.tokenizer.batch_decode(
                model_outputs_2["logits"].argmax(dim=-1),
                skip_special_tokens=True
            )
            semantic_metrics = self.compute_semantic_divergence(texts_1, texts_2)
            metrics.update(semantic_metrics)
        
        return metrics
