"""
Benchmark Integration Module for Behavioral Clustering

This module provides utilities for integrating external benchmark datasets
into the behavioral clustering pipeline. It includes functions for loading,
converting, and processing benchmark datasets into a format compatible with
the behavioral clustering system.

Supported benchmarks:
- Anthropic Model-Written Evaluations
- TruthfulQA
- Cultural Differences Benchmark
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import requests
import zipfile
import io
import yaml

from src.behavioural_clustering.utils.dataset_loader import DatasetRegistry

logger = logging.getLogger(__name__)


class BenchmarkConverter:
    """
    Converts external benchmark datasets into a format compatible with the behavioral clustering system.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the benchmark converter.
        
        Args:
            output_dir: Directory to save converted datasets
        """
        self.output_dir = Path(output_dir) if output_dir else Path("data/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_anthropic_benchmark(self, save_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Download the Anthropic Model-Written Evaluations benchmark.
        
        Args:
            save_path: Path to save the downloaded dataset
            
        Returns:
            Path to the downloaded dataset
        """
        url = "https://github.com/anthropics/evals/raw/main/anthropic_evals/data/model_written_evals.jsonl"
        save_path = Path(save_path) if save_path else self.output_dir / "anthropic_model_written_evals.jsonl"
        
        logger.info(f"Downloading Anthropic Model-Written Evaluations from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
            
        logger.info(f"Downloaded Anthropic benchmark to {save_path}")
        
        return save_path
        
    def download_truthfulqa(self, save_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Download the TruthfulQA benchmark.
        
        Args:
            save_path: Path to save the downloaded dataset
            
        Returns:
            Path to the downloaded dataset
        """
        url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/questions.json"
        save_path = Path(save_path) if save_path else self.output_dir / "truthfulqa.json"
        
        logger.info(f"Downloading TruthfulQA from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
            
        logger.info(f"Downloaded TruthfulQA to {save_path}")
        
        return save_path
        
    def convert_anthropic_to_statements(self, input_path: Union[str, Path], 
                                       output_path: Optional[Union[str, Path]] = None,
                                       max_statements: Optional[int] = None) -> Path:
        """
        Convert Anthropic Model-Written Evaluations to statements format.
        
        Args:
            input_path: Path to the Anthropic benchmark file
            output_path: Path to save the converted dataset
            max_statements: Maximum number of statements to include
            
        Returns:
            Path to the converted dataset
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else self.output_dir / "anthropic_statements.jsonl"
        
        logger.info(f"Converting Anthropic benchmark from {input_path} to statements format")
        
        statements = []
        
        with open(input_path, "r") as f:
            for i, line in enumerate(f):
                if max_statements and i >= max_statements:
                    break
                    
                data = json.loads(line)
                
                question = data.get("question", "")
                
                statement = {
                    "id": f"anthropic_{i}",
                    "statement": question,
                    "category": data.get("category", ""),
                    "subcategory": data.get("subcategory", ""),
                    "source": "anthropic_model_written_evals",
                    "metadata": {
                        "correct_answers": data.get("correct_answers", []),
                        "incorrect_answers": data.get("incorrect_answers", [])
                    }
                }
                
                statements.append(statement)
                
        with open(output_path, "w") as f:
            for statement in statements:
                f.write(json.dumps(statement) + "\n")
                
        logger.info(f"Converted {len(statements)} Anthropic benchmark questions to statements at {output_path}")
        
        return output_path
        
    def convert_truthfulqa_to_statements(self, input_path: Union[str, Path], 
                                        output_path: Optional[Union[str, Path]] = None,
                                        max_statements: Optional[int] = None) -> Path:
        """
        Convert TruthfulQA to statements format.
        
        Args:
            input_path: Path to the TruthfulQA benchmark file
            output_path: Path to save the converted dataset
            max_statements: Maximum number of statements to include
            
        Returns:
            Path to the converted dataset
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else self.output_dir / "truthfulqa_statements.jsonl"
        
        logger.info(f"Converting TruthfulQA from {input_path} to statements format")
        
        with open(input_path, "r") as f:
            data = json.load(f)
            
        statements = []
        
        for i, item in enumerate(data):
            if max_statements and i >= max_statements:
                break
                
            statement = {
                "id": f"truthfulqa_{i}",
                "statement": item.get("question", ""),
                "category": item.get("category", ""),
                "source": "truthfulqa",
                "metadata": {
                    "correct_answers": item.get("correct_answers", []),
                    "incorrect_answers": item.get("incorrect_answers", [])
                }
            }
            
            statements.append(statement)
            
        with open(output_path, "w") as f:
            for statement in statements:
                f.write(json.dumps(statement) + "\n")
                
        logger.info(f"Converted {len(statements)} TruthfulQA questions to statements at {output_path}")
        
        return output_path
        
    def create_contrastive_hypotheses(self, statements_path: Union[str, Path],
                                     output_path: Optional[Union[str, Path]] = None,
                                     model_name: str = "claude-3-5-sonnet-20241022",
                                     model_family: str = "anthropic",
                                     batch_size: int = 10) -> Path:
        """
        Create contrastive hypotheses from statements using an LLM.
        
        Args:
            statements_path: Path to the statements file
            output_path: Path to save the contrastive hypotheses
            model_name: Name of the model to use for generating hypotheses
            model_family: Family of the model
            batch_size: Number of statements to process in each batch
            
        Returns:
            Path to the contrastive hypotheses file
        """
        from src.behavioural_clustering.models.model_factory import initialize_model
        
        statements_path = Path(statements_path)
        output_path = Path(output_path) if output_path else statements_path.parent / f"{statements_path.stem}_contrastive.jsonl"
        
        logger.info(f"Creating contrastive hypotheses from {statements_path}")
        
        statements = []
        with open(statements_path, "r") as f:
            for line in f:
                statements.append(json.loads(line))
                
        model = initialize_model({
            "model_name": model_name,
            "model_family": model_family
        }, temperature=0.7, max_tokens=500)
        
        contrastive_statements = []
        
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i+batch_size]
            
            for statement in batch:
                prompt = self._create_contrastive_prompt(statement)
                
                try:
                    response = model.generate(prompt)
                    
                    hypotheses = self._parse_contrastive_response(response)
                    
                    for j, hypothesis in enumerate(hypotheses):
                        contrastive_statement = {
                            "id": f"{statement['id']}_contrastive_{j}",
                            "statement": hypothesis,
                            "category": statement.get("category", ""),
                            "source": f"{statement.get('source', '')}_contrastive",
                            "original_statement": statement.get("statement", ""),
                            "metadata": {
                                "original_id": statement.get("id", ""),
                                "original_metadata": statement.get("metadata", {})
                            }
                        }
                        
                        contrastive_statements.append(contrastive_statement)
                        
                except Exception as e:
                    logger.error(f"Error generating contrastive hypotheses for statement {statement.get('id', '')}: {e}")
                    
        with open(output_path, "w") as f:
            for statement in contrastive_statements:
                f.write(json.dumps(statement) + "\n")
                
        logger.info(f"Created {len(contrastive_statements)} contrastive hypotheses at {output_path}")
        
        return output_path
        
    def _create_contrastive_prompt(self, statement: Dict[str, Any]) -> str:
        """
        Create a prompt for generating contrastive hypotheses.
        
        Args:
            statement: Statement dictionary
            
        Returns:
            Prompt for generating contrastive hypotheses
        """
        prompt = f"""Generate 3-5 contrastive hypotheses based on the following statement or question. 
A contrastive hypothesis should:
1. Explore different possible responses or perspectives on the statement
2. Highlight potential behavioral differences between language models
3. Be specific enough to elicit meaningful differences
4. Not be too leading or biased

Original statement: "{statement.get('statement', '')}"

Generate each contrastive hypothesis on a new line, starting with "Hypothesis: ".
"""
        
        return prompt
        
    def _parse_contrastive_response(self, response: str) -> List[str]:
        """
        Parse the response from the model to extract contrastive hypotheses.
        
        Args:
            response: Model response
            
        Returns:
            List of contrastive hypotheses
        """
        hypotheses = []
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.startswith("Hypothesis:"):
                hypothesis = line[len("Hypothesis:"):].strip()
                hypotheses.append(hypothesis)
                
        return hypotheses
        
    def register_benchmark_datasets(self, registry: DatasetRegistry) -> None:
        """
        Register benchmark datasets with the dataset registry.
        
        Args:
            registry: Dataset registry
        """
        anthropic_path = self.output_dir / "anthropic_statements.jsonl"
        truthfulqa_path = self.output_dir / "truthfulqa_statements.jsonl"
        
        if anthropic_path.exists():
            registry.register_dataset(
                "anthropic-model-written-evals",
                str(anthropic_path),
                description="Anthropic Model-Written Evaluations benchmark",
                categories=["knowledge", "reasoning", "ethics", "safety"]
            )
            
        if truthfulqa_path.exists():
            registry.register_dataset(
                "truthfulqa",
                str(truthfulqa_path),
                description="TruthfulQA benchmark for measuring truthfulness",
                categories=["knowledge", "truthfulness", "misinformation"]
            )
            
        anthropic_contrastive_path = self.output_dir / "anthropic_statements_contrastive.jsonl"
        truthfulqa_contrastive_path = self.output_dir / "truthfulqa_statements_contrastive.jsonl"
        
        if anthropic_contrastive_path.exists():
            registry.register_dataset(
                "anthropic-model-written-evals-contrastive",
                str(anthropic_contrastive_path),
                description="Contrastive hypotheses based on Anthropic Model-Written Evaluations",
                categories=["knowledge", "reasoning", "ethics", "safety", "contrastive"]
            )
            
        if truthfulqa_contrastive_path.exists():
            registry.register_dataset(
                "truthfulqa-contrastive",
                str(truthfulqa_contrastive_path),
                description="Contrastive hypotheses based on TruthfulQA",
                categories=["knowledge", "truthfulness", "misinformation", "contrastive"]
            )


def download_and_prepare_benchmarks(output_dir: Optional[Union[str, Path]] = None,
                                   registry: Optional[DatasetRegistry] = None) -> DatasetRegistry:
    """
    Download and prepare benchmark datasets for behavioral clustering.
    
    Args:
        output_dir: Directory to save benchmark datasets
        registry: Dataset registry to register the benchmarks with
        
    Returns:
        Updated dataset registry
    """
    converter = BenchmarkConverter(output_dir)
    
    if registry is None:
        from src.behavioural_clustering.utils.dataset_loader import create_default_registry
        registry = create_default_registry()
        
    try:
        anthropic_path = converter.download_anthropic_benchmark()
        converter.convert_anthropic_to_statements(anthropic_path)
        
        truthfulqa_path = converter.download_truthfulqa()
        converter.convert_truthfulqa_to_statements(truthfulqa_path)
        
        converter.register_benchmark_datasets(registry)
        
    except Exception as e:
        logger.error(f"Error downloading and preparing benchmarks: {e}")
        
    return registry


def create_benchmark_config(output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a configuration file for running benchmark evaluations.
    
    Args:
        output_path: Path to save the configuration file
        
    Returns:
        Path to the configuration file
    """
    output_path = Path(output_path) if output_path else Path("config/benchmark_config.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "anthropic_benchmark_run": {
            "name": "anthropic_benchmark_run",
            "model_settings": {
                "models": [
                    ["anthropic", "claude-3-5-haiku-20241022"],
                    ["anthropic", "claude-3-5-sonnet-20241022"],
                    ["openai", "gpt-4o"]
                ]
            },
            "data_settings": {
                "datasets": ["anthropic-model-written-evals"],
                "n_statements": 500,
                "reuse_data": ["none"],
                "new_generation": False
            },
            "embedding_settings": {
                "embedding_model": "text-embedding-3-large",
                "batch_size": 100,
                "max_retries": 3
            },
            "clustering_settings": {
                "n_clusters": 20,
                "theme_identification_model_name": "claude-3-5-sonnet-20241022",
                "theme_identification_model_family": "anthropic"
            },
            "plot_settings": {
                "hide_plots": []
            },
            "run_sections": [
                "model_comparison",
                "hierarchical_clustering"
            ]
        },
        "truthfulqa_benchmark_run": {
            "name": "truthfulqa_benchmark_run",
            "model_settings": {
                "models": [
                    ["anthropic", "claude-3-5-haiku-20241022"],
                    ["anthropic", "claude-3-5-sonnet-20241022"],
                    ["openai", "gpt-4o"]
                ]
            },
            "data_settings": {
                "datasets": ["truthfulqa"],
                "n_statements": 300,
                "reuse_data": ["none"],
                "new_generation": False
            },
            "embedding_settings": {
                "embedding_model": "text-embedding-3-large",
                "batch_size": 100,
                "max_retries": 3
            },
            "clustering_settings": {
                "n_clusters": 15,
                "theme_identification_model_name": "claude-3-5-sonnet-20241022",
                "theme_identification_model_family": "anthropic"
            },
            "plot_settings": {
                "hide_plots": []
            },
            "run_sections": [
                "model_comparison",
                "hierarchical_clustering"
            ]
        },
        "contrastive_benchmark_run": {
            "name": "contrastive_benchmark_run",
            "model_settings": {
                "models": [
                    ["anthropic", "claude-3-5-haiku-20241022"],
                    ["anthropic", "claude-3-5-sonnet-20241022"],
                    ["openai", "gpt-4o"]
                ]
            },
            "data_settings": {
                "datasets": [
                    "anthropic-model-written-evals-contrastive",
                    "truthfulqa-contrastive"
                ],
                "n_statements": 400,
                "reuse_data": ["none"],
                "new_generation": False
            },
            "embedding_settings": {
                "embedding_model": "text-embedding-3-large",
                "batch_size": 100,
                "max_retries": 3
            },
            "clustering_settings": {
                "n_clusters": 25,
                "theme_identification_model_name": "claude-3-5-sonnet-20241022",
                "theme_identification_model_family": "anthropic"
            },
            "plot_settings": {
                "hide_plots": []
            },
            "run_sections": [
                "model_comparison",
                "hierarchical_clustering",
                "iterative_evaluation"
            ],
            "iterative_settings": {
                "max_iterations": 3,
                "prompts_per_iteration": 100,
                "min_difference_threshold": 0.1,
                "responses_per_prompt": 1
            }
        }
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info(f"Created benchmark configuration at {output_path}")
    
    return output_path
