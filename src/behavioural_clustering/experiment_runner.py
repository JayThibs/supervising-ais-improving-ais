"""
Experiment Runner for Behavioral Clustering

This module provides a comprehensive experiment runner for behavioral clustering
that makes it easy to run multiple experiments with different configurations.
It supports batch processing, parallel execution, and result aggregation.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import datetime
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.behavioural_clustering.main import main as run_main
from src.behavioural_clustering.config.run_settings import RunSettings
from src.behavioural_clustering.config.run_configuration_manager import RunConfigurationManager
from src.behavioural_clustering.utils.hardware_detection import get_hardware_info, configure_models_for_hardware

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Comprehensive experiment runner for behavioral clustering.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None,
                output_dir: Optional[Union[str, Path]] = None,
                log_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config_dir: Directory containing experiment configurations
            output_dir: Directory to save experiment results
            log_dir: Directory to save experiment logs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_config_manager = RunConfigurationManager()
        
        self.hardware_info = get_hardware_info()
        self.model_configs = configure_models_for_hardware()
        
        self._setup_logging()
        
    def _setup_logging(self):
        """
        Set up logging for the experiment runner.
        """
        log_file = self.log_dir / f"experiment_runner_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def load_experiment_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an experiment configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing the experiment configuration
        """
        config_path = Path(config_path)
        
        logger.info(f"Loading experiment configuration from {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        return config
        
    def run_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Run a single experiment with the given configuration.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            
        Returns:
            Run ID of the experiment
        """
        logger.info(f"Running experiment: {experiment_name}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_{timestamp}"
        
        log_file = self.log_dir / f"{run_id}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        
        logging.getLogger().addHandler(file_handler)
        
        try:
            args = argparse.Namespace()
            args.run = experiment_name
            args.run_only = config.get("run_only", [])
            args.skip = config.get("skip_sections", [])
            args.max_iterations = config.get("max_iterations", None)
            args.prompts_per_iteration = config.get("prompts_per_iteration", None)
            args.min_difference = config.get("min_difference", None)
            args.output_dir = str(self.output_dir / run_id)
            args.config_path = config.get("config_path", None)
            args.list_sections = False
            args.test_mode = config.get("test_mode", False)
            
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
            config_file = Path(args.output_dir) / "config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Starting experiment {run_id} with configuration: {config}")
            
            run_main(args)
            
            logger.info(f"Experiment {run_id} completed successfully")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Error running experiment {run_id}: {e}", exc_info=True)
            raise
            
        finally:
            logging.getLogger().removeHandler(file_handler)
            
    def run_batch_experiments(self, config_path: Union[str, Path],
                             parallel: bool = False,
                             max_workers: Optional[int] = None) -> List[str]:
        """
        Run a batch of experiments from a configuration file.
        
        Args:
            config_path: Path to the batch configuration file
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of run IDs for the experiments
        """
        batch_config = self.load_experiment_config(config_path)
        
        logger.info(f"Running batch of {len(batch_config)} experiments")
        
        run_ids = []
        
        if parallel:
            if max_workers is None:
                max_workers = min(len(batch_config), self.hardware_info.cpu_info.get("cores", 1))
                
            logger.info(f"Running experiments in parallel with {max_workers} workers")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_experiment = {
                    executor.submit(self.run_experiment, experiment_name, config): experiment_name
                    for experiment_name, config in batch_config.items()
                }
                
                for future in concurrent.futures.as_completed(future_to_experiment):
                    experiment_name = future_to_experiment[future]
                    
                    try:
                        run_id = future.result()
                        run_ids.append(run_id)
                        
                    except Exception as e:
                        logger.error(f"Experiment {experiment_name} failed: {e}")
                        
        else:
            for experiment_name, config in batch_config.items():
                try:
                    run_id = self.run_experiment(experiment_name, config)
                    run_ids.append(run_id)
                    
                except Exception as e:
                    logger.error(f"Experiment {experiment_name} failed: {e}")
                    
        logger.info(f"Completed batch of {len(batch_config)} experiments")
        
        return run_ids
        
    def run_grid_search(self, base_config_path: Union[str, Path],
                       param_grid: Dict[str, List[Any]],
                       output_prefix: str = "grid_search",
                       parallel: bool = False,
                       max_workers: Optional[int] = None) -> List[str]:
        """
        Run a grid search over parameter combinations.
        
        Args:
            base_config_path: Path to the base configuration file
            param_grid: Dictionary mapping parameter names to lists of values
            output_prefix: Prefix for the output directory
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of run IDs for the experiments
        """
        base_config = self.load_experiment_config(base_config_path)
        
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
        
        logger.info(f"Running grid search with {len(param_values)} parameter combinations")
        
        configs = []
        
        for i, values in enumerate(param_values):
            config = base_config.copy()
            
            for name, value in zip(param_names, values):
                if "." in name:
                    parts = name.split(".")
                    current = config
                    
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                            
                        current = current[part]
                        
                    current[parts[-1]] = value
                else:
                    config[name] = value
                    
            param_str = "_".join([f"{name.split('.')[-1]}_{value}" for name, value in zip(param_names, values)])
            experiment_name = f"{output_prefix}_{param_str}"
            
            configs.append((experiment_name, config))
            
        run_ids = []
        
        if parallel:
            if max_workers is None:
                max_workers = min(len(configs), self.hardware_info.cpu_info.get("cores", 1))
                
            logger.info(f"Running grid search in parallel with {max_workers} workers")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_experiment = {
                    executor.submit(self.run_experiment, experiment_name, config): experiment_name
                    for experiment_name, config in configs
                }
                
                for future in concurrent.futures.as_completed(future_to_experiment):
                    experiment_name = future_to_experiment[future]
                    
                    try:
                        run_id = future.result()
                        run_ids.append(run_id)
                        
                    except Exception as e:
                        logger.error(f"Experiment {experiment_name} failed: {e}")
                        
        else:
            for experiment_name, config in configs:
                try:
                    run_id = self.run_experiment(experiment_name, config)
                    run_ids.append(run_id)
                    
                except Exception as e:
                    logger.error(f"Experiment {experiment_name} failed: {e}")
                    
        logger.info(f"Completed grid search with {len(configs)} parameter combinations")
        
        return run_ids
        
    def aggregate_results(self, run_ids: List[str],
                         output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Aggregate results from multiple experiments.
        
        Args:
            run_ids: List of run IDs to aggregate
            output_path: Path to save the aggregated results
            
        Returns:
            Dictionary containing the aggregated results
        """
        logger.info(f"Aggregating results from {len(run_ids)} experiments")
        
        results = {}
        
        for run_id in run_ids:
            try:
                run_metadata = self.run_config_manager.get_run_metadata(run_id)
                
                if run_metadata:
                    results[run_id] = run_metadata
                    
            except Exception as e:
                logger.error(f"Error loading results for run {run_id}: {e}")
                
        summary = {
            "run_ids": run_ids,
            "num_runs": len(results),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        
        if output_path:
            output_path = Path(output_path)
            
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved aggregated results to {output_path}")
            
        return summary
        
    def compare_runs(self, run_ids: List[str],
                    metrics: Optional[List[str]] = None,
                    output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            output_path: Path to save the comparison results
            
        Returns:
            DataFrame containing the comparison results
        """
        logger.info(f"Comparing {len(run_ids)} runs")
        
        results = {}
        
        for run_id in run_ids:
            try:
                run_metadata = self.run_config_manager.get_run_metadata(run_id)
                
                if run_metadata:
                    results[run_id] = run_metadata
                    
            except Exception as e:
                logger.error(f"Error loading results for run {run_id}: {e}")
                
        comparison_data = []
        
        for run_id, run_results in results.items():
            run_data = {"run_id": run_id}
            
            if "config" in run_results:
                config = run_results["config"]
                
                for key, value in config.items():
                    if isinstance(value, (str, int, float, bool)):
                        run_data[f"config_{key}"] = value
                        
            if "metrics" in run_results:
                run_metrics = run_results["metrics"]
                
                if metrics:
                    for metric in metrics:
                        if metric in run_metrics:
                            run_data[f"metric_{metric}"] = run_metrics[metric]
                else:
                    for metric, value in run_metrics.items():
                        if isinstance(value, (int, float)):
                            run_data[f"metric_{metric}"] = value
                            
            comparison_data.append(run_data)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        if output_path:
            output_path = Path(output_path)
            
            comparison_df.to_csv(output_path, index=False)
            
            logger.info(f"Saved comparison results to {output_path}")
            
        return comparison_df


def main():
    """
    Main function for the experiment runner.
    """
    parser = argparse.ArgumentParser(description="Experiment Runner for Behavioral Clustering")
    
    parser.add_argument("--config", type=str, help="Path to the experiment configuration file")
    parser.add_argument("--batch", type=str, help="Path to the batch configuration file")
    parser.add_argument("--grid-search", type=str, help="Path to the grid search configuration file")
    parser.add_argument("--param-grid", type=str, help="Path to the parameter grid configuration file")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save experiment results")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save experiment logs")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--max-workers", type=int, help="Maximum number of parallel workers")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results from multiple experiments")
    parser.add_argument("--compare", action="store_true", help="Compare metrics across multiple runs")
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to compare")
    parser.add_argument("--run-ids", type=str, nargs="+", help="Run IDs to aggregate or compare")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    if args.config:
        config = runner.load_experiment_config(args.config)
        
        for experiment_name, experiment_config in config.items():
            runner.run_experiment(experiment_name, experiment_config)
            
    elif args.batch:
        runner.run_batch_experiments(
            args.batch,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
    elif args.grid_search and args.param_grid:
        with open(args.param_grid, "r") as f:
            param_grid = yaml.safe_load(f)
            
        runner.run_grid_search(
            args.grid_search,
            param_grid,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
    if args.aggregate and args.run_ids:
        runner.aggregate_results(
            args.run_ids,
            output_path=os.path.join(args.output_dir, "aggregated_results.json")
        )
        
    if args.compare and args.run_ids:
        runner.compare_runs(
            args.run_ids,
            metrics=args.metrics,
            output_path=os.path.join(args.output_dir, "run_comparison.csv")
        )


if __name__ == "__main__":
    main()
