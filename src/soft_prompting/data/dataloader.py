from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizer
from pathlib import Path
import logging

from ..config.configs import ExperimentConfig
from .dataset import TextDataset
from .processors import get_dataset_processor
from ..utils.device_utils import get_device

logger = logging.getLogger(__name__)

def create_experiment_dataloaders(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for experiment."""
    
    print(f"Creating dataloaders with config: {config.data}")
    
    base_path = Path(__file__).parents[3] / "data" / "evals"
    print(f"Using base path: {base_path}")
    
    if not base_path.exists():
        raise ValueError(f"Data directory does not exist: {base_path}")
    
    # Handle categories properly - if string convert to list
    categories = (
        ["all"] if config.data.categories == "all"
        else [config.data.categories] if isinstance(config.data.categories, str)
        else config.data.categories
    )
    
    # Load and process data for each category
    all_texts = []
    for category in categories:
        print(f"Processing category: {category}")
        processor = get_dataset_processor([category])
        
        try:
            category_texts = processor.load_texts(
                data_path=base_path,
                max_texts=config.data.max_texts_per_category,
                min_length=config.data.min_text_length,
                max_length=config.data.max_text_length
            )
            print(f"Loaded {len(category_texts)} texts from {category}")
            all_texts.extend(category_texts)
        except Exception as e:
            print(f"Error loading category {category}: {e}")
            continue
    
    if not all_texts:
        raise ValueError("No texts loaded from any of the specified categories")
    
    print(f"Loaded {len(all_texts)} total texts across all categories")
    
    # Calculate effective max length
    max_input_length = config.training.max_length - config.training.num_soft_prompt_tokens
    print(f"Using max input length of {max_input_length} tokens")
    print(f"Total length after soft prompt will be: {config.training.max_length}")
    
    # Create dataset with adjusted length
    dataset = TextDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_length=max_input_length,  # Use adjusted length
        padding="max_length",  # Ensure consistent padding
        truncation=True  # Enable truncation
    )
    
    # Split into train/val
    train_size = int(len(dataset) * config.data.train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    
    # Get device from config
    device = get_device(config.device)
    
    # Configure device-specific DataLoader settings
    pin_memory = device != 'cpu'  # Only use pin_memory for GPU devices
    num_workers = 0 if device == 'mps' else 2  # MPS doesn't work well with multiple workers
    
    # Create dataloaders with proper batch handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,  # Use full batch size
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Changed to False to use all data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,  # Use full batch size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    logger.info(f"Created dataloaders with {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    
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
