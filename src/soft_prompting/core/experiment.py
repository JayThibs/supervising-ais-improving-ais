# src/soft_prompting/core/experiment.py

from pathlib import Path
from typing import Dict, Optional
import logging
import json
import yaml

from ..config.configs import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer
from ..data.dataloader import create_experiment_dataloaders

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runs behavioral comparison experiments."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[Path] = None,
        test_mode: bool = False
    ):
        print(f"Initializing ExperimentRunner with output_dir: {output_dir}")
        self.config = config
        # Debug print config
        print(f"Config output_dir: {getattr(config, 'output_dir', None)}")
        
        # Ensure output_dir is a Path object and has a default
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        print(f"Final output_dir: {self.output_dir}")
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
        print(f"Setting up experiment with name: {experiment_name}, output_dir: {output_dir}")
        
        # Load experiment config
        config_path = Path(__file__).parents[1] / "config" / "experiments" / f"{experiment_name}.yaml"
        print(f"Looking for config at: {config_path}")
        
        if not config_path.exists():
            raise ValueError(f"Experiment config not found: {config_path}")
        
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded config: {config_dict}")
        
        # Ensure output directory is set
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path("outputs") / experiment_name
        
        output_path.mkdir(parents=True, exist_ok=True)
        config_dict["output_dir"] = str(output_path)
        print(f"Set output_dir in config to: {config_dict['output_dir']}")
        
        # Create experiment config
        config = ExperimentConfig.from_dict({
            "name": f"{experiment_name}_{'test' if test_mode else 'full'}",
            **config_dict,
        })
        print(f"Created config with output_dir: {getattr(config, 'output_dir', None)}")
        
        # Create and return runner instance
        return cls(config=config, output_dir=output_path, test_mode=test_mode)
        
    def run(self, model_pair_index: int = 0) -> Dict:
        """Run the experiment end-to-end."""
        # Load models
        logger.info("Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair(
            pair_index=model_pair_index
        )
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_experiment_dataloaders(
            config=self.config,
            tokenizer=tokenizer
        )
        
        # Initialize trainer
        trainer = DivergenceTrainer(
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=self.config
        )
        
        # Run training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )
        
        # Save results
        results_path = self.output_dir / f"results_pair_{model_pair_index}.json"
        
        # Add model pair info to results
        results["model_pair"] = {
            "model_1": model_1.name_or_path,
            "model_2": model_2.name_or_path
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
