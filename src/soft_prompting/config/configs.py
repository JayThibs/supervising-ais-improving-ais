# src/soft_prompting/config/configs.py

from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for training soft prompts."""
    num_soft_prompt_tokens: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    seed: int = 42
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 3
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    max_length: int = 512
    device: str = "cuda"

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 128
    num_generations_per_prompt: int = 10
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1

@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_path: Optional[Path] = None
    eval_path: Optional[Path] = None
    categories: List[str] = None
    max_texts_per_category: int = 1000
    min_text_length: int = 10
    max_text_length: int = 150
    train_split: float = 0.9

@dataclass
class ExperimentConfig:
    """Master configuration for experiments."""
    name: str
    output_dir: Path
    model_1_name: str
    model_2_name: str
    training: TrainingConfig
    generation: GenerationConfig
    data: DataConfig
    metrics: Dict[str, bool] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            name=config_dict["name"],
            output_dir=Path(config_dict["output_dir"]),
            model_1_name=config_dict["model_1_name"],
            model_2_name=config_dict["model_2_name"],
            training=TrainingConfig(**config_dict["training"]),
            generation=GenerationConfig(**config_dict["generation"]),
            data=DataConfig(**config_dict["data"]),
            metrics=config_dict.get("metrics", {})
        )
