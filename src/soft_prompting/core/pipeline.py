# src/soft_prompting/core/pipeline.py

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import os

from .experiment import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer
from ..data.dataloader import create_experiment_dataloaders
from ..analysis.divergence_analyzer import DivergenceAnalyzer
from ..tracking.experiment_tracker import ExperimentTracker
from ..utils.device_utils import get_device
from ..utils.serialization import serialize_for_json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
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
            # Save original values
            self._original_config = {
                'categories': self.config.data.categories.copy(),
                'max_texts': self.config.data.max_texts_per_category,
                'epochs': self.config.training.num_epochs,
                'batch_size': self.config.training.batch_size,
                'generations': self.config.generation.num_generations_per_prompt
            }
            # Modify config for test mode
            self.config.data.categories = ["persona/desire-for-acquiring-power"]
            self.config.data.max_texts_per_category = 12
            self.config.training.num_epochs = 5
            self.config.training.batch_size = 4
            self.config.generation.num_generations_per_prompt = 2
        else:
            print("Running in FULL MODE - using complete dataset")
            self._original_config = None
            
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
        self.trainer = None  # Add this line
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
        
    def show_examples(self, trainer, num_examples: int = 5):
        """Show example outputs from both models using the trained soft prompt."""
        print("\n=== Example Model Outputs with Trained Soft Prompt ===")
        
        # Get some validation texts
        sample_texts = trainer.get_eval_texts()[:num_examples]
        
        for text in sample_texts:
            print("\nInput:", text[:100], "...")
            outputs = trainer.generate_with_soft_prompt(
                text, 
                max_length=100,
                num_return_sequences=1
            )
            print("\nModel 1:", outputs["generation_1"][:200], "...")
            print("\nModel 2:", outputs["generation_2"][:200], "...")
            print("\nKL Divergence:", outputs["metrics"]["kl_divergence"])
            print("-" * 80)

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
        self.trainer = DivergenceTrainer(
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=self.config,
            tracker=self.tracker
        )
        print("Trainer initialized successfully.")
        
        # Train soft prompts
        print("\nStep 4: Training soft prompts...")
        training_results = self.trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )
        print("Soft prompt training complete.")

        # Generate divergent dataset if not in results
        print("\nStep 5: Generating divergent dataset...")
        dataset = training_results.get("dataset", [])
        
        # Debug print
        print(f"DEBUG: training_results keys: {training_results.keys()}")
        print(f"DEBUG: dataset type: {type(dataset)}")
        print(f"DEBUG: dataset length: {len(dataset)}")
        
        if not dataset:
            print("WARNING: No dataset in training results")
            dataset = []
        
        # Prepare analysis data
        analysis_data = {
            "metrics": training_results.get("final_metrics", {}),
            "best_divergence": training_results.get("best_divergence", 0.0),
            "total_steps": training_results.get("total_steps", 0),
            "dataset": dataset  # Use the dataset directly
        }
        
        # Initialize analyzer
        analyzer = DivergenceAnalyzer(
            metrics=self.trainer.metrics_computer,
            output_dir=self.output_dir
        )
        
        try:
            if dataset:
                print(f"Analyzing dataset with {len(dataset)} examples...")
                analysis = analyzer.generate_report(analysis_data)
                print("Analysis complete.")
            else:
                print("No dataset available for analysis, creating empty report...")
                analysis = {
                    "warning": "No dataset available for analysis",
                    "overall_stats": {"mean_divergence": 0.0},
                    "divergence_patterns": {"num_high_divergence": 0}
                }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            print(f"Analysis data keys: {analysis_data.keys()}")
            analysis = {
                "error": str(e),
                "overall_stats": {"mean_divergence": 0.0},
                "divergence_patterns": {"num_high_divergence": 0}
            }

        # Prepare final results
        final_results = {
            "metrics": training_results.get("final_metrics", {}),
            "best_divergence": training_results.get("best_divergence", 0.0),
            "total_steps": training_results.get("total_steps", 0),
            "dataset": dataset,
            "analysis": analysis
        }

        return final_results
    
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
