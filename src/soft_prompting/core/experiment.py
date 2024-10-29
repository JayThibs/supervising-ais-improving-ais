# src/soft_prompting/core/experiment.py

from pathlib import Path
from typing import Dict, Optional
import logging
import json

from ..config.configs import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs behavioral comparison experiments."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[Path] = None,
        test_mode: bool = False
    ):
        self.config = config
        self.output_dir = output_dir or config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager with config
        self.model_manager = ModelPairManager(
            config=config,
            test_mode=test_mode
        )
        
    def run(self, model_pair_index: int = 0) -> Dict:
        """Run the experiment end-to-end."""
        # Load models
        logger.info("Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair(
            pair_index=model_pair_index
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
        model_pair = self.model_manager.get_model_pair(model_pair_index)
        results_path = self.output_dir / f"results_pair_{model_pair_index}.json"
        
        # Add model pair info to results
        results["model_pair"] = {
            "model_1": model_pair[0],
            "model_2": model_pair[1]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
