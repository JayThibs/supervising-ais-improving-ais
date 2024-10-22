from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class TextDataset(Dataset):
    """Simple dataset for text samples."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze()
        }

def create_dataloader(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    config: object
) -> DataLoader:
    """Create DataLoader from texts."""
    dataset = TextDataset(texts, tokenizer, config.max_length)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )