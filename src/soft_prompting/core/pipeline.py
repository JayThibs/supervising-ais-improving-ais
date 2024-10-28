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
from ..utils.device_utils import get_device

logger = logging.getLogger(__name__)

class DivergencePipeline:
    """Pipeline for discovering behavioral differences between HuggingFace models."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        use_wandb: bool = True
    ):
        print("Initializing DivergencePipeline...")
        self.config = config
        self.output_dir = config.output_dir
        
        # Set device using utility function
        self.device = get_device(config.device)
        config.device = self.device  # Update config to match
            
        # Initialize components
        print("Setting up ModelPairManager...")
        self.model_manager = ModelPairManager(
            model_1_name=config.model_1_name,
            model_2_name=config.model_2_name,
            device=self.device,  # Pass consistent device
            torch_dtype=torch.float16 if config.torch_dtype == "float16" else torch.float32,
            load_in_8bit=config.load_in_8bit
        )
        
        # Setup experiment tracking
        print("Setting up experiment tracking...")
        self.tracker = ExperimentTracker(
            config=config,
            use_wandb=use_wandb,
            project_name="soft-prompting"
        )
        print("Pipeline initialization complete.")
        
    def run(self, checkpoint_paths: Optional[List[Path]] = None):
        """
        Run full pipeline.
        
        Args:
            checkpoint_paths: Optional list of soft prompt checkpoint paths to use
                            If None, will train new soft prompts
        """
        print("\n=== Starting pipeline run ===")
        # Load models
        print("\nStep 1: Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair()
        print(f"Successfully loaded models: {self.config.model_1_name} and {self.config.model_2_name}")
        
        results = {}
        
        if checkpoint_paths:
            print("\nUsing existing checkpoints for hard prompt generation...")
            # Use existing soft prompts to generate hard prompts
            from .hard_prompt_generator import HardPromptGenerator
            
            print("Initializing HardPromptGenerator...")
            generator = HardPromptGenerator(
                model_1=model_1,
                model_2=model_2,
                tokenizer=tokenizer,
                metrics=self.trainer.metrics,
                device=self.config.training.device
            )
            
            # Generate hard prompts using each checkpoint
            print("Generating hard prompts...")
            results["hard_prompts"] = generator.batch_generate(
                checkpoint_paths=checkpoint_paths,
                input_texts=self.trainer.get_eval_texts(),
                config=self.config.generation,
                output_dir=self.output_dir / "hard_prompts"
            )
            print("Hard prompt generation complete.")
        
        # Create dataloaders
        print("\nStep 2: Creating dataloaders...")
        train_loader, val_loader = create_experiment_dataloaders(
            config=self.config,
            tokenizer=tokenizer
        )
        print("Dataloaders created successfully.")
        
        # Initialize trainer
        print("\nStep 3: Initializing trainer...")
        trainer = DivergenceTrainer(
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=self.config,
            tracker=self.tracker
        )
        print("Trainer initialized successfully.")
        
        # Train soft prompts
        print("\nStep 4: Training soft prompts...")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )
        print("Soft prompt training complete.")
        
        # Generate divergent dataset
        print("\nStep 5: Generating divergent dataset...")
        dataset = trainer.generate_divergent_dataset(
            output_file=self.output_dir / "divergent_dataset.pt"
        )
        print("Divergent dataset generation complete.")
        
        # Analyze results
        print("\nStep 6: Analyzing results...")
        analyzer = DivergenceAnalyzer(
            metrics=trainer.metrics,
            output_dir=self.output_dir
        )
        analysis = analyzer.generate_report(dataset)
        print("Analysis complete.")
        
        # Save results
        print("\nStep 7: Saving experiment summary...")
        self.tracker.save_experiment_summary()
        print("\n=== Pipeline run complete ===\n")
        
        return {
            "dataset": dataset,
            "analysis": analysis,
            "trainer": trainer
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            print("Cleaning up pipeline resources...")
            if hasattr(self, 'model_manager'):
                self.model_manager.clear_cache()
            if hasattr(self, 'tracker'):
                self.tracker.finish()
            print("Cleanup complete.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
