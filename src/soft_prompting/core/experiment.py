# src/soft_prompting/core/experiment.py

from pathlib import Path
from typing import Dict, Optional
import logging
import json
import yaml

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
        
    @classmethod
    def setup(
        cls,
        experiment_name: str = "intervention_comparison",
        output_dir: Optional[str] = None,
        test_mode: bool = False,
    ) -> "ExperimentRunner":
        """
        Setup an experiment with configuration.
        
        Args:
            experiment_name: Name of experiment config file (without .yaml)
            output_dir: Directory to save outputs. If None, uses default from config
            test_mode: If True, uses small test models
        
        Returns:
            ExperimentRunner instance ready to run
        """
        # Load experiment config
        config_path = Path(__file__).parents[1] / "config" / "experiments" / f"{experiment_name}.yaml"
        
        if not config_path.exists():
            raise ValueError(f"Experiment config not found: {config_path}")
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            config_dict["output_dir"] = str(output_path)
        
        # Create experiment config
        config = ExperimentConfig.from_dict({
            "name": f"{experiment_name}_{'test' if test_mode else 'full'}",
            **config_dict,
        })
        
        # Create and return runner instance
        return cls(config=config, test_mode=test_mode)
        
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
        results_path = self.output_dir / f"results_pair_{model_pair_index}.json"
        
        # Add model pair info to results
        results["model_pair"] = {
            "model_1": model_1.name,
            "model_2": model_2.name
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
