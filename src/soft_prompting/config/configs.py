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
        # Load base config defaults
        base_config_path = Path(__file__).parent / "base_config.yaml"
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)["defaults"]
        
        # Merge base config with experiment config
        training_config = {**base_config["training"], **config_dict.get("training", {})}
        generation_config = {**base_config["generation"], **config_dict.get("generation", {})}
        data_config = {**base_config["data"], **config_dict.get("data", {})}
        
        # Ensure numeric types, but skip certain keys
        training_config = {k: int(v) if isinstance(v, str) and k not in ["learning_rate", "max_grad_norm"] else v 
                          for k, v in training_config.items()}
        generation_config = {k: int(v) if isinstance(v, str) and k not in ["temperature", "top_p"] else v 
                           for k, v in generation_config.items()}
        
        # Modified data config conversion to handle non-numeric values
        numeric_data_keys = ["max_texts_per_category", "min_text_length", "max_text_length"]
        data_config = {
            k: (int(v) if isinstance(v, str) and k in numeric_data_keys else v)
            for k, v in data_config.items()
        }
        
        return cls(
            name=config_dict["name"],
            output_dir=Path(config_dict["output_dir"]),
            model_pairs=config_dict["model_pairs"],
            training=TrainingConfig(**training_config),
            generation=GenerationConfig(**generation_config),
            data=DataConfig(**data_config),
            metrics=config_dict.get("metrics", {}),
            torch_dtype=config_dict.get("torch_dtype", base_config["model"]["torch_dtype"]),
            load_in_8bit=config_dict.get("load_in_8bit", base_config["model"]["load_in_8bit"]),
            device=config_dict.get("device", base_config["model"]["device"])
        )
