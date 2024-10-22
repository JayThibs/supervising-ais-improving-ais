from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for divergence training."""
    num_soft_prompt_tokens: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_length: int = 128
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    generate_length: int = 64
    num_generations_per_prompt: int = 10
    generation_temperature: float = 0.7
    device: str = "cuda"
    save_dir: str = "outputs"
    seed: int = 42
    logging_steps: int = 10