from typing import Dict, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper for consistent interface across HuggingFace models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate text with consistent output format."""
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs
        )
        
        return {
            "sequences": outputs.sequences,
            "scores": outputs.scores,
            "logits": outputs.scores  # Alias for consistency
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
    def get_input_embeddings(self) -> torch.nn.Module:
        """Get input embeddings layer."""
        return self.model.get_input_embeddings()
    
    def to(self, device: str) -> "ModelWrapper":
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        return self
    
    def eval(self) -> "ModelWrapper":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
