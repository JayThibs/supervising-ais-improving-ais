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
        use_wandb: bool = True,
        test_mode: bool = False,
        model_pair_index: int = 0
    ):
        print("Initializing DivergencePipeline...")
        self.config = config
        self.output_dir = config.output_dir
        self.test_mode = test_mode
        
        if test_mode:
            print("Running in TEST MODE - using reduced dataset and iterations")
            # Modify config for test mode
            self.config.data.max_texts_per_category = min(10, self.config.data.max_texts_per_category)
            self.config.training.num_epochs = min(2, self.config.training.num_epochs)
            self.config.training.batch_size = min(4, self.config.training.batch_size)
            self.config.generation.num_generations_per_prompt = min(2, self.config.generation.num_generations_per_prompt)
            
        # Set device using utility function
        self.device = get_device(config.device)
        config.device = self.device  # Update config to match
            
        print(f"Using device: {self.device}")
        
        # Get the specific model pair
        self.model_pair = config.model_pairs[model_pair_index]
        
        # Initialize components
        print("Setting up ModelPairManager...")
        self.model_manager = ModelPairManager(
            model_1_name=self.model_pair["model_1"],
            model_2_name=self.model_pair["model_2"],
            device=self.device,
            torch_dtype=torch.float16 if config.torch_dtype == "float16" else torch.float32,
            load_in_8bit=config.load_in_8bit
        )
        
        # Setup experiment tracking
        print("Setting up experiment tracking...")
        self.tracker = ExperimentTracker(
            config=config,
            use_wandb=use_wandb and not test_mode,  # Disable wandb in test mode
            project_name="soft-prompting"
        )
        print("Pipeline initialization complete.")
        
    def validate_categories(self):
        """Validate all categories in config and return available ones."""
        valid_categories = []
        invalid_categories = []
        
        print("\nValidating dataset categories...")
        for category in self.config.data.categories:
            try:
                # Create temporary dataloader with minimal samples to test category
                temp_loader, _ = create_experiment_dataloaders(
                    config=self.config,
                    tokenizer=self.model_manager.tokenizer,
                    category=category,
                    max_texts=2 if self.test_mode else 10
                )
                valid_categories.append(category)
                print(f"✓ Category validated: {category}")
            except Exception as e:
                invalid_categories.append((category, str(e)))
                print(f"✗ Category failed: {category} - {str(e)}")
        
        if invalid_categories:
            print("\nWarning: Some categories were invalid:")
            for cat, error in invalid_categories:
                print(f"  - {cat}: {error}")
        
        return valid_categories
        
    def run(self, checkpoint_paths: Optional[List[Path]] = None, validate_only: bool = False):
        """
        Run full pipeline.
        
        Args:
            checkpoint_paths: Optional list of soft prompt checkpoint paths to use
            validate_only: If True, only validate categories without running the full pipeline
        """
        print("\n=== Starting pipeline run ===")
        print(f"Mode: {'TEST' if self.test_mode else 'FULL'}")
        
        # Load models
        print("\nStep 1: Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair()
        
        # Validate models
        if not self.model_manager.validate_models(model_1, model_2, tokenizer):
            raise RuntimeError("Model validation failed")
            
        print(f"Successfully loaded models: {self.model_pair['model_1']} and {self.model_pair['model_2']}")
        
        # Validate categories
        valid_categories = self.validate_categories()
        if not valid_categories:
            raise ValueError("No valid categories found in configuration")
        
        if validate_only:
            print("\nValidation complete. Exiting as validate_only=True")
            return {"valid_categories": valid_categories}
        
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
            "trainer": trainer,
            "valid_categories": valid_categories
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
