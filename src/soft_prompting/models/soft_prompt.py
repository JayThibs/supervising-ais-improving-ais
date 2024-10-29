# src/soft_prompting/models/soft_prompt.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

class DivergenceSoftPrompt(nn.Module):
    """Trainable soft prompt for maximizing divergence between models."""
    
    def __init__(
        self, 
        num_tokens: int,
        embedding_dim: int,
        init_std: float = 0.02
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        
        # Initialize trainable embeddings
        self.embeddings = nn.Parameter(
            torch.randn(num_tokens, embedding_dim) * init_std,
            requires_grad=True
        )
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompt embeddings to input embeddings.
        
        Args:
            input_embeddings: Input token embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embeddings [batch_size, seq_len + num_tokens, embedding_dim]
        """
        batch_size = input_embeddings.shape[0]
        
        print(f"\nSoft Prompt Forward Pass:")
        print(f"Input embeddings shape: {input_embeddings.shape}")
        print(f"Soft prompt embeddings requires_grad: {self.embeddings.requires_grad}")
        
        # Expand soft prompt embeddings to batch size
        soft_prompt_expanded = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        print(f"Expanded soft prompt requires_grad: {soft_prompt_expanded.requires_grad}")
        
        # Concatenate along sequence length dimension
        combined_embeddings = torch.cat([soft_prompt_expanded, input_embeddings], dim=1)
        print(f"Combined embeddings requires_grad: {combined_embeddings.requires_grad}")
        print(f"Combined embeddings grad_fn: {combined_embeddings.grad_fn}")
        
        return combined_embeddings

    def get_soft_prompt_length(self) -> int:
        """Return number of soft prompt tokens."""
        return self.num_tokens