# src/soft_prompting/core/experiment.py

from pathlib import Path
from typing import Dict, Optional, Union
import logging
import json
import yaml
import uuid
from datetime import datetime

from ..config.configs import ExperimentConfig
from ..models.model_manager import ModelPairManager
from ..training.trainer import DivergenceTrainer
from ..data.dataloader import create_experiment_dataloaders
from ..analysis.divergence_analyzer import DivergenceAnalyzer

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
        self.test_mode = test_mode
        
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
        
        # Add run metadata tracking
        self.run_id = None
        self.run_metadata = None
        self.results = None
        
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
        
        # Set test mode configuration
        if test_mode:
            config_dict["data"] = {
                **config_dict.get("data", {}),
                "categories": "anthropic-model-written-evals/advanced-ai-risk/human_generated_evals/power-seeking-inclination",
                "max_texts_per_category": 25,
                "test_mode": True
            }
        
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
        
    @classmethod
    def load_from_run(
        cls,
        run_dir: Union[str, Path],
        run_id: Optional[str] = None
    ) -> "ExperimentRunner":
        """
        Load an experiment runner from a previous run.
        
        Args:
            run_dir: Directory containing experiment runs
            run_id: Specific run ID to load. If None, loads most recent run.
            
        Returns:
            Loaded ExperimentRunner instance
        """
        run_dir = Path(run_dir)
        
        # Find available runs
        runs = list(run_dir.glob("run_*"))
        if not runs:
            raise ValueError(f"No runs found in {run_dir}")
            
        if run_id is None:
            # Get most recent run
            runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            run_path = runs[0]
        else:
            run_path = run_dir / f"run_{run_id}"
            if not run_path.exists():
                raise ValueError(f"Run {run_id} not found in {run_dir}")
        
        # Load run metadata
        with open(run_path / "metadata.json") as f:
            metadata = json.load(f)
            
        # Load config
        with open(run_path / "config.yaml") as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig.from_dict(config_dict)
        
        # Create runner instance
        runner = cls(config=config, output_dir=run_path)
        runner.run_id = metadata["run_id"]
        runner.run_metadata = metadata
        
        # Load results if available
        results_path = run_path / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                runner.results = json.load(f)
                
        # Load trained soft prompt if available
        soft_prompt_path = run_path / "soft_prompt.pt"
        if soft_prompt_path.exists():
            runner.load_soft_prompt(soft_prompt_path)
            
        return runner
    
    def save_run(self) -> str:
        """
        Save the current run state.
        
        Returns:
            run_id: Unique identifier for this run
        """
        # Generate run ID if not exists
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            
        # Create run directory
        run_dir = self.output_dir / f"run_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "test_mode": getattr(self, 'test_mode', False),  # Use getattr with default
            "model_pair": {
                "model_1": self.model_manager.model_1_name,
                "model_2": self.model_manager.model_2_name
            }
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Convert config to dict and handle Path objects
        config_dict = self.config.to_dict()
        def convert_paths(d):
            return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
        
        config_dict = convert_paths(config_dict)
        
        # Save config
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f)
            
        # Convert results for JSON serialization
        if hasattr(self, "results"):
            serializable_results = json.loads(
                json.dumps(self.results, default=lambda x: str(x) if isinstance(x, Path) else x)
            )
            with open(run_dir / "results.json", "w") as f:
                json.dump(serializable_results, f, indent=2)
                
        return self.run_id
    
    def list_runs(self, output_format: str = "text") -> Union[str, Dict]:
        """
        List all available runs in the output directory.
        
        Args:
            output_format: "text" or "dict"
            
        Returns:
            Formatted run information
        """
        runs = []
        for run_path in self.output_dir.glob("run_*"):
            try:
                with open(run_path / "metadata.json") as f:
                    metadata = json.load(f)
                runs.append({
                    "run_id": metadata["run_id"],
                    "timestamp": metadata["timestamp"],
                    "test_mode": metadata["test_mode"],
                    "model_pair": metadata["model_pair"],
                    "path": str(run_path)
                })
            except Exception as e:
                logger.warning(f"Could not load metadata for run in {run_path}: {e}")
                
        if output_format == "dict":
            return runs
            
        # Format as text
        output = ["Available Runs:"]
        for run in sorted(runs, key=lambda x: x["timestamp"], reverse=True):
            output.append(f"\nRun ID: {run['run_id']}")
            output.append(f"Timestamp: {run['timestamp']}")
            output.append(f"Test Mode: {run['test_mode']}")
            output.append(f"Models: {run['model_pair']['model_1']} vs {run['model_pair']['model_2']}")
            output.append(f"Path: {run['path']}")
            output.append("-" * 50)
            
        return "\n".join(output)
    
    def _make_serializable(self, obj):
        """
        Recursively convert Path objects in a data structure to strings.
        
        Args:
            obj: The data structure to convert.
        
        Returns:
            A new data structure with Path objects converted to strings.
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def run(self, model_pair_index: int = 0) -> Dict:
        """Run the experiment end-to-end."""
        print("\n=== Starting Experiment Run ===")
        
        # Load models
        print("Loading models...")
        model_1, model_2, tokenizer = self.model_manager.load_model_pair(
            pair_index=model_pair_index
        )
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader = create_experiment_dataloaders(
            config=self.config,
            tokenizer=tokenizer
        )
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = DivergenceTrainer(
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=self.config
        )
        
        # Run training
        print("Starting training...")
        training_results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader
        )
        
        print("\nPreparing analysis...")
        # Initialize analyzer with string path
        analyzer = DivergenceAnalyzer(
            metrics=trainer.metrics_computer,
            output_dir=str(self.output_dir / f"analysis_pair_{model_pair_index}")
        )
        
        # Generate analysis report
        analysis_report = analyzer.generate_report(
            dataset=training_results.get("dataset", []),
            output_file="analysis_report.json"
        )
        
        # Prepare final results with explicit serialization
        final_results = {
            "metrics": training_results.get("final_metrics", {}),
            "best_divergence": float(training_results.get("best_divergence", 0.0)),
            "total_steps": int(training_results.get("total_steps", 0)),
            "dataset": training_results.get("dataset", []),
            "analysis": analysis_report,
            "model_pair": {
                "model_1": str(model_1.name_or_path),
                "model_2": str(model_2.name_or_path)
            }
        }
        
        # Save results
        results_path = self.output_dir / f"results_pair_{model_pair_index}.json"
        print(f"\nSaving results to {results_path}")
        
        # Convert any remaining Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, (int, float, bool, str)):
                return obj
            else:
                return str(obj)
        
        serializable_results = convert_paths(final_results)
        
        # Test JSON serialization before saving
        try:
            print("\nTesting JSON serialization...")
            json_str = json.dumps(serializable_results, indent=2)
            with open(results_path, 'w') as f:
                f.write(json_str)
            print("Results successfully saved")
        except TypeError as e:
            print(f"JSON serialization error: {str(e)}")
            print("\nResults structure:")
            for k, v in serializable_results.items():
                print(f"{k}: {type(v)}")
                if isinstance(v, dict):
                    print(f"  Nested keys in {k}: {list(v.keys())}")
            raise
        
        # Save run state
        self.results = serializable_results
        self.save_run()
        
        return serializable_results

    def get_run_info(self, detailed: bool = False) -> Dict[str, Dict]:
        """
        Get information about all available runs.
        
        Args:
            detailed: If True, includes additional metrics and analysis results
            
        Returns:
            Dictionary mapping run_ids to their information
        """
        runs_info = {}
        
        for run_path in sorted(self.output_dir.glob("run_*"), 
                              key=lambda x: x.stat().st_mtime, 
                              reverse=True):
            try:
                # Load basic metadata
                with open(run_path / "metadata.json") as f:
                    metadata = json.load(f)
                
                run_info = {
                    "id": metadata["run_id"],
                    "timestamp": metadata["timestamp"],
                    "models": {
                        "model_1": metadata["model_pair"]["model_1"],
                        "model_2": metadata["model_pair"]["model_2"]
                    },
                    "test_mode": metadata["test_mode"],
                    "path": str(run_path)
                }
                
                # Add status information
                run_info["status"] = {
                    "has_results": (run_path / "results.json").exists(),
                    "has_soft_prompt": (run_path / "soft_prompt.pt").exists(),
                    "has_analysis": (run_path / "analysis_report.json").exists()
                }
                
                if detailed:
                    # Add results summary if available
                    results_path = run_path / "results.json"
                    if results_path.exists():
                        with open(results_path) as f:
                            results = json.load(f)
                        run_info["results_summary"] = {
                            "best_divergence": results.get("best_divergence", None),
                            "training_steps": results.get("total_steps", None),
                            "early_stopped": results.get("early_stopped", None)
                        }
                    
                    # Add analysis summary if available
                    analysis_path = run_path / "analysis_report.json"
                    if analysis_path.exists():
                        with open(analysis_path) as f:
                            analysis = json.load(f)
                        run_info["analysis_summary"] = {
                            "mean_divergence": analysis.get("overall_stats", {}).get("mean_divergence"),
                            "num_high_divergence": analysis.get("divergence_patterns", {}).get("num_high_divergence")
                        }
                
                runs_info[metadata["run_id"]] = run_info
                
            except Exception as e:
                logger.warning(f"Error loading run info from {run_path}: {e}")
                continue
        
        return runs_info

    def summarize_runs(self, format: str = "text", detailed: bool = False) -> Union[str, Dict]:
        """
        Summarize all available runs in a readable format.
        
        Args:
            format: Output format - "text" or "dict"
            detailed: Include detailed metrics and analysis
            
        Returns:
            Formatted run summary
        """
        runs_info = self.get_run_info(detailed=detailed)
        
        if format == "dict":
            return runs_info
        
        # Format as text
        lines = ["=== Available Experiment Runs ===\n"]
        
        for run_id, info in runs_info.items():
            timestamp = datetime.fromisoformat(info["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            
            lines.extend([
                f"Run ID: {run_id}",
                f"Timestamp: {timestamp}",
                f"Models:",
                f"  - Model 1: {info['models']['model_1']}",
                f"  - Model 2: {info['models']['model_2']}",
                f"Test Mode: {info['test_mode']}",
                f"Status:",
                f"  - Results: {'✓' if info['status']['has_results'] else '✗'}",
                f"  - Soft Prompt: {'✓' if info['status']['has_soft_prompt'] else '✗'}",
                f"  - Analysis: {'✓' if info['status']['has_analysis'] else '✗'}"
            ])
            
            if detailed and "results_summary" in info:
                lines.extend([
                    "Results:",
                    f"  - Best Divergence: {info['results_summary']['best_divergence']:.4f}",
                    f"  - Training Steps: {info['results_summary']['training_steps']}",
                    f"  - Early Stopped: {info['results_summary']['early_stopped']}"
                ])
                
            if detailed and "analysis_summary" in info:
                lines.extend([
                    "Analysis:",
                    f"  - Mean Divergence: {info['analysis_summary']['mean_divergence']:.4f}",
                    f"  - High Divergence Examples: {info['analysis_summary']['num_high_divergence']}"
                ])
                
            lines.extend(["", "-" * 50, ""])
        
        return "\n".join(lines)

    @classmethod
    def list_available_runs(cls, output_dir: Path, detailed: bool = False) -> str:
        """
        Static method to list available runs without creating an ExperimentRunner instance.
        
        Args:
            output_dir: Directory containing experiment runs
            detailed: Include detailed metrics and analysis
            
        Returns:
            Formatted string describing available runs
        """
        temp_runner = cls(
            config=ExperimentConfig(output_dir=output_dir),
            output_dir=output_dir
        )
        return temp_runner.summarize_runs(detailed=detailed)
