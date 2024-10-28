# src/soft_prompting/core/pipeline.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from .experiment import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer
from ..data.dataloader import create_experiment_dataloaders
from ..analysis.divergence_analyzer import DivergenceAnalyzer
from ..tracking.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class DivergencePipeline:
    """Pipeline for discovering behavioral differences between HuggingFace models."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        use_wandb: bool = True
    ):
        self.config = config
        self.output_dir = config.output_dir
        
        # Initialize components
        self.model_manager = ModelPairManager(
            device="cuda",
            torch_dtype=config.torch_dtype,
            load_in_8bit=config.load_in_8bit
        )
        
        # Setup experiment tracking
        self.tracker = ExperimentTracker(
            config=config,
            use_wandb=use_wandb,
            project_name="soft-prompting"
        )
        
    def run(self, checkpoint_paths: Optional[List[Path]] = None):
        """
        Run full pipeline.
        
        Args:
            checkpoint_paths: Optional list of soft prompt checkpoint paths to use
                            If None, will train new soft prompts
        """
        # Load models
        logger.info("Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair(
            self.config.model_1_name,
            self.config.model_2_name
        )
        
        results = {}
        
        if checkpoint_paths:
            # Use existing soft prompts to generate hard prompts
            from .hard_prompt_generator import HardPromptGenerator
            
            generator = HardPromptGenerator(
                model_1=model_1,
                model_2=model_2,
                tokenizer=tokenizer,
                metrics=self.trainer.metrics,
                device=self.config.training.device
            )
            
            # Generate hard prompts using each checkpoint
            results["hard_prompts"] = generator.batch_generate(
                checkpoint_paths=checkpoint_paths,
                input_texts=self.trainer.get_eval_texts(),
                config=self.config.generation,
                output_dir=self.output_dir / "hard_prompts"
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
            config=self.config,
            tracker=self.tracker
        )
        
        # Train soft prompts
        logger.info("Training soft prompts...")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )
        
        # Generate divergent dataset
        logger.info("Generating divergent dataset...")
        dataset = trainer.generate_divergent_dataset(
            output_file=self.output_dir / "divergent_dataset.pt"
        )
        
        # Analyze results
        logger.info("Analyzing results...")
        analyzer = DivergenceAnalyzer(
            metrics=trainer.metrics,
            output_dir=self.output_dir
        )
        analysis = analyzer.generate_report(dataset)
        
        # Save results
        self.tracker.save_experiment_summary()
        
        return {
            "dataset": dataset,
            "analysis": analysis,
            "trainer": trainer
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.model_manager.clear_cache()
        if hasattr(self, 'tracker'):
            self.tracker.finish()
