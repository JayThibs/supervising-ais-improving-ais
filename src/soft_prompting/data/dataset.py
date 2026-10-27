from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TextDataset(Dataset):
    """Dataset for text samples with optional soft prompts."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text with consistent padding
        encoded = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        
        # Remove batch dimension and ensure proper shape
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": text
        }
