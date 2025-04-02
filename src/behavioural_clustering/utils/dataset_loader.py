import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import yaml
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: Path
    file_pattern: str = "**/*.jsonl"
    statement_field: str = "statement"
    alternative_fields: List[str] = field(default_factory=lambda: ["question", "input", "text"])
    metadata_fields: List[str] = field(default_factory=lambda: ["id", "category", "difficulty"])
    max_length: int = 500
    min_length: int = 10
    
    def __post_init__(self):
        """Convert path to Path object if it's a string."""
        if isinstance(self.path, str):
            self.path = Path(self.path)


class DatasetRegistry:
    """Registry for available datasets."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dataset registry.
        
        Args:
            config_path: Path to the dataset configuration file
        """
        self.datasets: Dict[str, DatasetConfig] = {}
        self.config_path = Path(config_path) if config_path else None
        
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_config(self):
        """Load dataset configurations from file."""
        if not self.config_path:
            logger.warning("No configuration path specified")
            return
            
        try:
            with open(str(self.config_path), 'r') as f:
                config_data = yaml.safe_load(f)
                
            if not config_data or not isinstance(config_data, dict):
                logger.warning(f"Invalid dataset configuration in {self.config_path}")
                return
                
            for name, dataset_config in config_data.items():
                if not isinstance(dataset_config, dict):
                    logger.warning(f"Invalid configuration for dataset {name}")
                    continue
                    
                config = DatasetConfig(
                    name=name,
                    path=Path(dataset_config.get('path', '')),
                    file_pattern=dataset_config.get('file_pattern', '**/*.jsonl'),
                    statement_field=dataset_config.get('statement_field', 'statement'),
                    alternative_fields=dataset_config.get('alternative_fields', ["question", "input", "text"]),
                    metadata_fields=dataset_config.get('metadata_fields', ["id", "category", "difficulty"]),
                    max_length=dataset_config.get('max_length', 500),
                    min_length=dataset_config.get('min_length', 10)
                )
                
                self.datasets[name] = config
                
        except Exception as e:
            logger.error(f"Error loading dataset configuration: {str(e)}")
    
    def save_config(self):
        """Save dataset configurations to file."""
        if not self.config_path:
            logger.warning("No configuration path specified")
            return
            
        try:
            config_data = {}
            for name, config in self.datasets.items():
                config_data[name] = {
                    'path': str(config.path),
                    'file_pattern': config.file_pattern,
                    'statement_field': config.statement_field,
                    'alternative_fields': config.alternative_fields,
                    'metadata_fields': config.metadata_fields,
                    'max_length': config.max_length,
                    'min_length': config.min_length
                }
                
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            logger.info(f"Saved dataset configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset configuration: {str(e)}")
    
    def register_dataset(self, config: DatasetConfig):
        """
        Register a new dataset.
        
        Args:
            config: Dataset configuration
        """
        self.datasets[config.name] = config
        self.save_config()
        
    def get_dataset(self, name: str) -> Optional[DatasetConfig]:
        """
        Get a dataset configuration by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dataset configuration or None if not found
        """
        return self.datasets.get(name)
        
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())


class DatasetLoader:
    """Loader for behavioral clustering datasets."""
    
    def __init__(self, registry: DatasetRegistry):
        """
        Initialize the dataset loader.
        
        Args:
            registry: Dataset registry
        """
        self.registry = registry
        
    def load_dataset(self, name: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Load a dataset by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Tuple of (list of statements with metadata, count of loaded files, count of rejected files)
        """
        config = self.registry.get_dataset(name)
        if not config:
            logger.error(f"Dataset {name} not found in registry")
            return [], 0, 0
            
        return self._load_dataset_files(config)
        
    def _load_dataset_files(self, config: DatasetConfig) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Load all files in a dataset.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (list of statements with metadata, count of loaded files, count of rejected files)
        """
        all_statements = []
        accepted_count = 0
        rejected_count = 0
        
        file_paths = glob.glob(str(config.path / config.file_pattern), recursive=True)
        logger.info(f"Found {len(file_paths)} files in dataset {config.name}")
        
        for path in file_paths:
            statements, accepted = self._load_file(Path(path), config)
            all_statements.extend(statements)
            
            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1
                
        logger.info(f"Loaded {len(all_statements)} statements from dataset {config.name}")
        logger.info(f"Accepted files: {accepted_count}, Rejected files: {rejected_count}")
        
        return all_statements, accepted_count, rejected_count
        
    def _load_file(self, file_path: Path, config: DatasetConfig) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Load statements from a file.
        
        Args:
            file_path: Path to the file
            config: Dataset configuration
            
        Returns:
            Tuple of (list of statements with metadata, whether the file was accepted)
        """
        statements = []
        accepted = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    try:
                        data = json.loads(line)
                        statement = None
                        
                        if config.statement_field in data:
                            statement = data[config.statement_field]
                        else:
                            for field in config.alternative_fields:
                                if field in data:
                                    statement = data[field]
                                    break
                        
                        if statement:
                            if config.min_length <= len(statement) <= config.max_length:
                                metadata = {
                                    'source_file': str(file_path),
                                    'dataset': config.name
                                }
                                
                                for field in config.metadata_fields:
                                    if field in data:
                                        metadata[field] = data[field]
                                
                                statements.append({
                                    'statement': statement,
                                    'metadata': metadata
                                })
                                
                                accepted = True
                                
                    except json.JSONDecodeError:
                        logger.debug(f"Invalid JSON in file {file_path}")
                        
        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {str(e)}")
            
        return statements, accepted
        
    def filter_statements(self, statements: List[Dict[str, Any]], 
                         max_length: Optional[int] = None,
                         min_length: Optional[int] = None,
                         categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Filter statements based on criteria.
        
        Args:
            statements: List of statements with metadata
            max_length: Maximum statement length
            min_length: Minimum statement length
            categories: List of categories to include
            
        Returns:
            Filtered list of statements
        """
        filtered = []
        
        for item in statements:
            statement = item['statement']
            metadata = item.get('metadata', {})
            
            if max_length and len(statement) > max_length:
                continue
                
            if min_length and len(statement) < min_length:
                continue
                
            if categories and 'category' in metadata:
                if metadata['category'] not in categories:
                    continue
                    
            filtered.append(item)
            
        logger.info(f"Filtered {len(statements)} statements to {len(filtered)}")
        return filtered


def create_default_registry(data_dir: Union[str, Path]) -> DatasetRegistry:
    """
    Create a default dataset registry with common datasets.
    
    Args:
        data_dir: Base directory for datasets
        
    Returns:
        Dataset registry with default datasets
    """
    data_dir = Path(data_dir)
    config_path = data_dir / "dataset_config.yaml"
    
    registry = DatasetRegistry(config_path)
    
    if not registry.datasets:
        registry.register_dataset(DatasetConfig(
            name="anthropic_evals",
            path=data_dir / "evals" / "anthropic_evals",
            file_pattern="**/*.jsonl",
            statement_field="statement",
            alternative_fields=["question", "input"],
            metadata_fields=["id", "category", "difficulty"]
        ))
        
        registry.register_dataset(DatasetConfig(
            name="truthful_qa",
            path=data_dir / "evals" / "truthful_qa",
            file_pattern="**/*.jsonl",
            statement_field="question",
            alternative_fields=["statement", "input"],
            metadata_fields=["id", "category", "difficulty"]
        ))
        
        registry.register_dataset(DatasetConfig(
            name="cultural_differences",
            path=data_dir / "evals" / "cultural_differences",
            file_pattern="**/*.jsonl",
            statement_field="statement",
            alternative_fields=["question", "input"],
            metadata_fields=["id", "category", "region"]
        ))
        
        registry.save_config()
        
    return registry
