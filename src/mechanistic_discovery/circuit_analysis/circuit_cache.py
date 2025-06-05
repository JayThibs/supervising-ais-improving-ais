"""
Circuit Cache Module

This module provides efficient caching for computed circuits (attribution graphs).
Since circuit computation is expensive (30-60 seconds per prompt), caching is
essential for iterative analysis and experimentation.

Key Features:
    - Disk-based persistence with memory cache
    - Content-based hashing for cache keys
    - Automatic eviction of old entries
    - Thread-safe operations
    - Compression for large graphs

Technical Details:
    - Uses SHA256 hashing of (model_name, prompt, config) for cache keys
    - Stores Graph objects as compressed pickles
    - Maintains an in-memory LRU cache for fast access
    - Supports cache invalidation and selective clearing
"""

import os
import pickle
import gzip
import hashlib
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, asdict
from collections import OrderedDict
from threading import Lock
import torch

from circuit_tracer import Graph


@dataclass
class CacheEntry:
    """
    Metadata for a cached circuit.
    
    Attributes:
        key: Unique identifier for the cache entry
        model_name: Name of the model used
        prompt: Input prompt
        timestamp: When the circuit was computed
        file_path: Where the circuit is stored on disk
        access_count: Number of times accessed
        size_bytes: Size of the cached file
    """
    key: str
    model_name: str
    prompt: str
    timestamp: float
    file_path: str
    access_count: int = 0
    size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class CircuitCache:
    """
    Manages caching of computed circuits to avoid expensive recomputation.
    
    This cache uses a two-tier system:
    1. In-memory LRU cache for fast access to recent circuits
    2. Disk-based storage for persistent caching across sessions
    
    The cache is content-addressed using a hash of the model name, prompt,
    and configuration parameters. This ensures that identical computations
    always map to the same cache entry.
    
    Example:
        >>> cache = CircuitCache(cache_dir="./circuit_cache", max_memory_items=100)
        >>> 
        >>> # Check if circuit exists
        >>> if cache.has_circuit("gpt2", "Hello world", config):
        >>>     circuit = cache.get_circuit("gpt2", "Hello world", config)
        >>> else:
        >>>     circuit = compute_circuit(...)  # Expensive operation
        >>>     cache.put_circuit("gpt2", "Hello world", config, circuit)
    """
    
    def __init__(self, 
                 cache_dir: str = "./circuit_cache",
                 max_memory_items: int = 100,
                 max_disk_gb: float = 10.0):
        """
        Initialize the circuit cache.
        
        Args:
            cache_dir: Directory for storing cached circuits
            max_memory_items: Maximum number of circuits to keep in memory
            max_disk_gb: Maximum disk space to use (in GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_items = max_memory_items
        self.max_disk_bytes = int(max_disk_gb * 1024 * 1024 * 1024)
        
        # In-memory LRU cache
        self.memory_cache: OrderedDict[str, Graph] = OrderedDict()
        self.memory_lock = Lock()
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        self.metadata_lock = Lock()
        
        # Clean up old entries if needed
        self._enforce_disk_limit()
        
    def _load_metadata(self) -> Dict[str, CacheEntry]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {k: CacheEntry.from_dict(v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Failed to load cache metadata: {e}")
                return {}
        return {}
        
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with self.metadata_lock:
            data = {k: v.to_dict() for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
    def _compute_cache_key(self, 
                          model_name: str, 
                          prompt: str,
                          config: Dict[str, Any]) -> str:
        """
        Compute a unique cache key for a circuit computation.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            config: Configuration parameters that affect the circuit
            
        Returns:
            SHA256 hash as hex string
        """
        # Create a canonical representation
        key_data = {
            'model_name': model_name,
            'prompt': prompt,
            'config': {
                'max_n_logits': config.get('max_n_logits', 10),
                'desired_logit_prob': config.get('desired_logit_prob', 0.95),
                'max_feature_nodes': config.get('max_feature_nodes', None)
            }
        }
        
        # Convert to stable JSON string
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Compute hash
        return hashlib.sha256(key_str.encode()).hexdigest()
        
    def has_circuit(self, 
                   model_name: str,
                   prompt: str,
                   config: Dict[str, Any]) -> bool:
        """
        Check if a circuit is in the cache.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            config: Configuration parameters
            
        Returns:
            True if circuit is cached, False otherwise
        """
        key = self._compute_cache_key(model_name, prompt, config)
        
        # Check memory cache first
        with self.memory_lock:
            if key in self.memory_cache:
                return True
                
        # Check disk cache
        return key in self.metadata
        
    def get_circuit(self,
                   model_name: str,
                   prompt: str,
                   config: Dict[str, Any]) -> Optional[Graph]:
        """
        Retrieve a circuit from the cache.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            config: Configuration parameters
            
        Returns:
            Cached Graph object or None if not found
        """
        key = self._compute_cache_key(model_name, prompt, config)
        
        # Check memory cache first
        with self.memory_lock:
            if key in self.memory_cache:
                # Move to end (LRU)
                self.memory_cache.move_to_end(key)
                return self.memory_cache[key]
                
        # Check disk cache
        if key in self.metadata:
            entry = self.metadata[key]
            file_path = Path(entry.file_path)
            
            if file_path.exists():
                try:
                    # Load from disk
                    with gzip.open(file_path, 'rb') as f:
                        graph = pickle.load(f)
                        
                    # Update access count
                    entry.access_count += 1
                    self._save_metadata()
                    
                    # Add to memory cache
                    self._add_to_memory_cache(key, graph)
                    
                    return graph
                    
                except Exception as e:
                    print(f"Warning: Failed to load cached circuit: {e}")
                    # Remove corrupted entry
                    del self.metadata[key]
                    self._save_metadata()
                    
        return None
        
    def put_circuit(self,
                   model_name: str,
                   prompt: str,
                   config: Dict[str, Any],
                   graph: Graph) -> str:
        """
        Store a circuit in the cache.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt
            config: Configuration parameters
            graph: Computed Graph object to cache
            
        Returns:
            Cache key for the stored circuit
        """
        key = self._compute_cache_key(model_name, prompt, config)
        
        # Prepare file path
        file_name = f"{key[:8]}_{int(time.time())}.pkl.gz"
        file_path = self.cache_dir / file_name
        
        # Save to disk
        try:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(graph, f)
                
            # Get file size
            size_bytes = file_path.stat().st_size
            
            # Create metadata entry
            entry = CacheEntry(
                key=key,
                model_name=model_name,
                prompt=prompt[:100],  # Truncate long prompts
                timestamp=time.time(),
                file_path=str(file_path),
                size_bytes=size_bytes
            )
            
            # Update metadata
            with self.metadata_lock:
                self.metadata[key] = entry
                self._save_metadata()
                
            # Add to memory cache
            self._add_to_memory_cache(key, graph)
            
            # Enforce disk limit
            self._enforce_disk_limit()
            
            print(f"Cached circuit: {key[:8]}... ({size_bytes / 1024:.1f} KB)")
            return key
            
        except Exception as e:
            print(f"Error caching circuit: {e}")
            if file_path.exists():
                file_path.unlink()
            raise
            
    def _add_to_memory_cache(self, key: str, graph: Graph):
        """Add a circuit to the in-memory cache."""
        with self.memory_lock:
            # Remove oldest if at capacity
            if len(self.memory_cache) >= self.max_memory_items:
                self.memory_cache.popitem(last=False)
                
            self.memory_cache[key] = graph
            
    def _enforce_disk_limit(self):
        """Remove old cache entries if disk usage exceeds limit."""
        with self.metadata_lock:
            # Calculate total size
            total_size = sum(entry.size_bytes for entry in self.metadata.values())
            
            if total_size <= self.max_disk_bytes:
                return
                
            # Sort by access count and timestamp
            entries = list(self.metadata.items())
            entries.sort(key=lambda x: (x[1].access_count, x[1].timestamp))
            
            # Remove oldest/least accessed until under limit
            while total_size > self.max_disk_bytes and entries:
                key, entry = entries.pop(0)
                
                # Remove file
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_path.unlink()
                    
                # Update metadata
                del self.metadata[key]
                total_size -= entry.size_bytes
                
                print(f"Evicted cache entry: {key[:8]}...")
                
            self._save_metadata()
            
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            model_name: If provided, only clear entries for this model.
                       If None, clear entire cache.
        """
        with self.metadata_lock:
            if model_name is None:
                # Clear everything
                for entry in self.metadata.values():
                    file_path = Path(entry.file_path)
                    if file_path.exists():
                        file_path.unlink()
                        
                self.metadata.clear()
                
                with self.memory_lock:
                    self.memory_cache.clear()
                    
                print("Cleared entire circuit cache")
                
            else:
                # Clear only specified model
                keys_to_remove = [
                    key for key, entry in self.metadata.items()
                    if entry.model_name == model_name
                ]
                
                for key in keys_to_remove:
                    entry = self.metadata[key]
                    file_path = Path(entry.file_path)
                    if file_path.exists():
                        file_path.unlink()
                        
                    del self.metadata[key]
                    
                    with self.memory_lock:
                        self.memory_cache.pop(key, None)
                        
                print(f"Cleared {len(keys_to_remove)} entries for model: {model_name}")
                
            self._save_metadata()
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.metadata_lock:
            total_size = sum(entry.size_bytes for entry in self.metadata.values())
            model_counts = {}
            
            for entry in self.metadata.values():
                model_counts[entry.model_name] = model_counts.get(entry.model_name, 0) + 1
                
            # Find most/least accessed
            if self.metadata:
                most_accessed = max(self.metadata.values(), key=lambda x: x.access_count)
                least_accessed = min(self.metadata.values(), key=lambda x: x.access_count)
            else:
                most_accessed = least_accessed = None
                
            return {
                'total_entries': len(self.metadata),
                'total_size_mb': total_size / (1024 * 1024),
                'memory_entries': len(self.memory_cache),
                'model_counts': model_counts,
                'most_accessed': most_accessed.prompt if most_accessed else None,
                'least_accessed': least_accessed.prompt if least_accessed else None,
                'cache_hit_rate': self._calculate_hit_rate()
            }
            
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate based on access counts."""
        if not self.metadata:
            return 0.0
            
        total_accesses = sum(entry.access_count for entry in self.metadata.values())
        if total_accesses == 0:
            return 0.0
            
        # Approximate hit rate (entries with >1 access were hits)
        hits = sum(max(0, entry.access_count - 1) for entry in self.metadata.values())
        return hits / total_accesses