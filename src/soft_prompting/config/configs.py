# src/soft_prompting/config/configs.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import yaml

@dataclass
class TrainingConfig:
    """Configuration for training soft prompts."""
    num_soft_prompt_tokens: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_epochs: int = 10
    test_mode_epochs: int = 5  # Added for test mode
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    seed: int = 42
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 50  # Reduced for more frequent test mode evaluation
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
    test_mode_texts: int = 12  # Added for test mode
    min_text_length: int = 10
    max_text_length: int = 150
    train_split: float = 0.9
    test_mode: bool = False

@dataclass
class ExperimentConfig:
    """Master configuration for experiments."""
    name: str
    output_dir: Path
    model_pairs: List[Dict[str, str]]
    training: TrainingConfig
    generation: GenerationConfig
    data: DataConfig
    metrics: Dict = field(default_factory=dict)
    torch_dtype: str = "auto"
    load_in_8bit: bool = False
    device: str = "auto"
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Create nested configs first
        training_config = TrainingConfig(**config_dict.get("training", {}))
        generation_config = GenerationConfig(**config_dict.get("generation", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Create main config
        return cls(
            name=config_dict.get("name"),
            output_dir=Path(config_dict.get("output_dir", "outputs")),
            model_pairs=config_dict.get("model_pairs", []),
            training=training_config,
            generation=generation_config,
            data=data_config,
            metrics=config_dict.get("metrics", {}),
            torch_dtype=config_dict.get("torch_dtype", "auto"),
            load_in_8bit=config_dict.get("load_in_8bit", False),
            device=config_dict.get("device", "auto")
        )

    def __init__(self, **kwargs):
        # Convert any Path objects to strings during initialization
        for key, value in kwargs.items():
            if isinstance(value, Path):
                setattr(self, key, str(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert config to dictionary with serializable values."""
        return {
            "name": self.name,
            "output_dir": str(self.output_dir) if isinstance(self.output_dir, Path) else self.output_dir,
            "model_pairs": self.model_pairs,
            "training": vars(self.training),  # Convert TrainingConfig to dict
            "generation": vars(self.generation),  # Convert GenerationConfig to dict
            "data": vars(self.data),  # Convert DataConfig to dict
            "metrics": self.metrics,
            "torch_dtype": self.torch_dtype,
            "load_in_8bit": self.load_in_8bit,
            "device": self.device
        }

    def __getstate__(self):
        """Custom serialization method."""
        return self.to_dict()
