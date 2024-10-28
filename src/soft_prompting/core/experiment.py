# src/soft_prompting/core/experiment.py

from pathlib import Path
from typing import Dict, Optional
import logging

from ..config.configs import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs behavioral comparison experiments."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[Path] = None
    ):
        self.config = config
        self.output_dir = output_dir or config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment config
        config_path = self.output_dir / "config.yaml"
        self.config.save(config_path)
        
        # Initialize components
        self.model_manager = ModelPairManager(
            device=config.training.device,
            torch_dtype=config.training.torch_dtype,
            load_in_8bit=config.training.load_in_8bit
        )
        
    def run(self) -> Dict:
        """Run the experiment end-to-end."""
        # Load models
        logger.info("Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair(
            self.config.model_1_name,
            self.config.model_2_name
        )
        
        # Initialize trainer
        trainer = DivergenceTrainer(
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=self.config
        )
        
        # Run training
        results = trainer.train()
        
        # Save results
        results_path = self.output_dir / "results.json"
        trainer.save_results(results_path)
        
        return results
