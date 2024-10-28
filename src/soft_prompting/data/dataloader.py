from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizer
from pathlib import Path
import logging

from ..config.configs import ExperimentConfig
from .dataset import TextDataset
from .processors import get_dataset_processor

logger = logging.getLogger(__name__)

def create_experiment_dataloaders(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for experiment."""
    
    # Load and process data
    processor = get_dataset_processor(config.data.categories)
    texts = processor.load_texts(
        data_path=config.data.train_path,
        max_texts=config.data.max_texts_per_category,
        min_length=config.data.min_text_length,
        max_length=config.data.max_text_length
    )
    
    # Create dataset
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=config.training.max_length
    )
    
    # Split into train/val
    train_size = int(len(dataset) * config.data.train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Created dataloaders with {len(train_dataset)} train and {len(val_dataset)} val samples")
    
    return train_loader, val_loader

def create_eval_dataloader(
    data_dir: Path,
    datasets: List[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: Optional[int] = None,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for evaluation datasets."""
    
    processor = get_dataset_processor(datasets)
    texts = processor.load_texts(
        data_path=data_dir,
        max_length=max_length
    )
    
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
