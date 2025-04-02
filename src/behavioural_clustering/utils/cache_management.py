from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import json
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np

@dataclass
class ModelParams:
    model_family: str
    model_name: str
    temperature: float
    max_tokens: int
    system_message: str

    def to_dict(self) -> Dict:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_message": self.system_message
        }

@dataclass
class EmbeddingParams:
    model_name: str
    batch_size: int
    max_retries: int
    initial_sleep_time: float

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "initial_sleep_time": self.initial_sleep_time
        }

@dataclass
class DataParams:
    n_statements: int
    datasets: List[str]
    random_seed: int
    new_generation: bool
    reuse_data: List[str]

    def to_dict(self) -> Dict:
        return {
            "n_statements": self.n_statements,
            "datasets": self.datasets,
            "random_seed": self.random_seed,
            "new_generation": self.new_generation,
            "reuse_data": self.reuse_data
        }

class CacheMetadata:
    def __init__(self, 
                 data_params: DataParams,
                 model_params: List[ModelParams],
                 embedding_params: EmbeddingParams,
                 cache_type: str):
        self.creation_time = datetime.now().isoformat()
        self.data_params = data_params
        self.model_params = model_params
        self.embedding_params = embedding_params
        self.cache_type = cache_type
        self.version = "1.0.0"

    def to_dict(self) -> Dict:
        return {
            "creation_time": self.creation_time,
            "version": self.version,
            "cache_type": self.cache_type,
            "data_params": self.data_params.to_dict(),
            "model_params": [mp.to_dict() for mp in self.model_params],
            "embedding_params": self.embedding_params.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict) -> 'CacheMetadata':
        return CacheMetadata(
            data_params=DataParams(**data["data_params"]),
            model_params=[ModelParams(**mp) for mp in data["model_params"]],
            embedding_params=EmbeddingParams(**data["embedding_params"]),
            cache_type=data["cache_type"]
        )

class CacheManager:
    def __init__(self, base_cache_dir: Path):
        self.base_cache_dir = base_cache_dir
        self.metadata_dir = base_cache_dir / "metadata"
        self.data_dir = base_cache_dir / "data"
        
        # Create directories if they don't exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(self, metadata: CacheMetadata) -> str:
        """Generate a deterministic cache key based on critical parameters."""
        key_components = [
            # Data parameters
            f"data:{metadata.data_params.n_statements}",
            f"datasets:{'-'.join(sorted(metadata.data_params.datasets))}",
            
            # Model parameters (preserving order)
            "models:" + "-".join([
                f"{mp.model_family}_{mp.model_name}" 
                for mp in metadata.model_params
            ]),
            
            # System messages (must match model order)
            "messages:" + "-".join([
                hashlib.md5(mp.system_message.encode()).hexdigest()[:8]
                for mp in metadata.model_params
            ]),
            
            # Embedding parameters
            f"embed:{metadata.embedding_params.model_name}"
        ]
        
        return hashlib.md5("|".join(key_components).encode()).hexdigest()

    def save_cache(self, data: Any, metadata: CacheMetadata) -> str:
        """Save both data and metadata with the same cache key."""
        cache_key = self.generate_cache_key(metadata)
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save data
        data_path = self.data_dir / f"{cache_key}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
            
        return cache_key

    def load_cache(self, current_metadata: CacheMetadata) -> Optional[Tuple[Any, CacheMetadata]]:
        """Try to load cached data that matches the current configuration."""
        cache_key = self.generate_cache_key(current_metadata)
        
        # Check if cache exists
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        data_path = self.data_dir / f"{cache_key}.pkl"
        
        if not (metadata_path.exists() and data_path.exists()):
            return None
            
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            cached_metadata = CacheMetadata.from_dict(json.load(f))
            
        # Validate cache
        if not self._validate_cache(cached_metadata, current_metadata):
            return None
            
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        return data, cached_metadata

    def _validate_cache(self, cached: CacheMetadata, current: CacheMetadata) -> bool:
        """Validate cache based on parameter dependencies and ordering."""
        # Check data parameters
        if not self._validate_data_params(cached.data_params, current.data_params):
            return False
            
        # Check model parameters and ordering
        if not self._validate_model_params(cached.model_params, current.model_params):
            return False
            
        # Check embedding parameters
        if not self._validate_embedding_params(cached.embedding_params, current.embedding_params):
            return False
            
        return True

    def _validate_data_params(self, cached: DataParams, current: DataParams) -> bool:
        """Validate data parameters."""
        return (cached.n_statements == current.n_statements and
                sorted(cached.datasets) == sorted(current.datasets) and
                cached.random_seed == current.random_seed)

    def _validate_model_params(self, cached: List[ModelParams], current: List[ModelParams]) -> bool:
        """Validate model parameters including order."""
        if len(cached) != len(current):
            return False
            
        for c_mp, n_mp in zip(cached, current):
            if (c_mp.model_family != n_mp.model_family or
                c_mp.model_name != n_mp.model_name or
                c_mp.temperature != n_mp.temperature or
                c_mp.max_tokens != n_mp.max_tokens or
                c_mp.system_message != n_mp.system_message):
                return False
                
        return True

    def _validate_embedding_params(self, cached: EmbeddingParams, current: EmbeddingParams) -> bool:
        """Validate embedding parameters."""
        return (cached.model_name == current.model_name and
                cached.batch_size == current.batch_size) 