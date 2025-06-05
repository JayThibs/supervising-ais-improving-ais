# Duplicated Pipeline Code Analysis

## Executive Summary

After analyzing the codebase, I've identified significant code duplication across the four main modules: `behavioural_clustering`, `interventions`, `contrastive-decoding`, and `soft_prompting`. The duplication spans model management, data handling, API interactions, embedding generation, clustering algorithms, and visualization functions.

## 1. Model Loading and Initialization

### Duplicated Patterns Found:

#### behavioural_clustering/models/model_factory.py
- Uses a factory pattern with `initialize_model()` function
- Supports OpenAI, Anthropic, local models, and OpenRouter
- Simple dictionary-based configuration

#### soft_prompting/models/model_manager.py
- Implements `ModelPairManager` class with device management
- Uses YAML registry for model configuration
- Includes caching mechanism and validation
- Handles model pairs specifically

#### interventions/intervention_models/model_manager.py
- `InterventionModelManager` class with YAML config
- Focuses on intervention/original model pairs
- Includes model unloading functionality

#### contrastive-decoding/model_comparison_helpers.py
- Complex `instantiate_models()` function
- Builds custom contrastive LM classes dynamically
- Handles quantization and device mapping
- Includes model weight interpolation

### Opportunities for Consolidation:
- Create a unified `ModelManager` base class with:
  - Common device handling logic
  - Shared caching mechanism
  - Standardized YAML configuration loading
  - Base methods for model loading/unloading
- Subclasses for specific use cases (pairs, contrastive, etc.)

## 2. Data Loading and Preprocessing

### Duplicated Patterns Found:

#### behavioural_clustering/utils/data_preparation.py
- `DataPreparation` class for JSONL file loading
- `DataHandler` class with pickle-based caching
- Metadata management with YAML files
- Hash-based file identification

#### interventions/auto_finetune_eval/auto_finetuning_data.py
- Dataset generation from APIs
- TruthfulQA dataset integration
- CSV output format
- Batch text decoding

#### contrastive-decoding/model_comparison_helpers.py
- `load_jsonl_data()` with flexible key selection
- Interactive key selection mode
- Pickle-based selection caching

### Opportunities for Consolidation:
- Unified data loading interface:
  ```python
  class DataLoader:
      def load_jsonl(self, path, keys=None)
      def load_dataset(self, name, split)
      def cache_data(self, data, key)
      def load_cached(self, key)
  ```
- Shared metadata management system
- Common file discovery and validation

## 3. Embedding Generation and Caching

### Duplicated Patterns Found:

#### behavioural_clustering/utils/embedding_utils.py
- `embed_texts()` function using OpenAI API
- Batch processing with retry logic
- Exponential backoff for rate limits

#### behavioural_clustering/utils/embedding_manager.py
- `EmbeddingManager` class with JSON caching
- Key-based embedding storage
- Lazy loading of new embeddings

#### interventions/auto_finetune_eval/auto_finetuning_helpers.py
- Embedding references for t-SNE visualization
- No dedicated embedding generation (uses external)

#### contrastive-decoding/quick_cluster.py
- Local embedding model support (NV-Embed-v1)
- Recompute embeddings flag
- Integration with clustering pipeline

### Opportunities for Consolidation:
- Unified `EmbeddingService`:
  ```python
  class EmbeddingService:
      def __init__(self, cache_dir, provider='openai'):
          self.cache = EmbeddingCache(cache_dir)
          self.provider = self._init_provider(provider)
      
      def get_embeddings(self, texts, model, batch_size=32):
          # Check cache first
          # Generate missing embeddings
          # Update cache
  ```

## 4. API Handling

### Duplicated Patterns Found:

#### behavioural_clustering/models/api_models.py
- Separate classes for OpenAI, Anthropic, OpenRouter
- Retry logic with tenacity library
- Environment variable loading

#### interventions/auto_finetune_eval/auto_finetuning_helpers.py
- `make_api_request()` function supporting multiple providers
- `collect_dataset_from_api()` for batch requests
- Logging of API interactions to file

#### soft_prompting (uses external providers)
- No direct API implementation found

### Opportunities for Consolidation:
- Unified API client factory:
  ```python
  class APIClientFactory:
      @staticmethod
      def create_client(provider: str, api_key: str = None):
          # Return appropriate client
      
  class BaseAPIClient(ABC):
      def request(self, prompt, **kwargs)
      def batch_request(self, prompts, **kwargs)
  ```

## 5. Clustering Algorithms

### Duplicated Patterns Found:

#### behavioural_clustering/evaluation/clustering.py
- `Clustering` class with algorithm map
- Support for Spectral, KMeans, Agglomerative, OPTICS
- `ClusterAnalyzer` for theme identification

#### contrastive-decoding/quick_cluster.py
- Direct sklearn usage (SpectralClustering, HDBSCAN, KMeans)
- Cluster validation and matching functions
- P-value computation for clusters

#### interventions/auto_finetune_eval/auto_finetuning_interp.py
- KMeans clustering for interpretation
- Cluster matching across models
- Statistical analysis of clusters

### Opportunities for Consolidation:
- Unified clustering interface:
  ```python
  class ClusteringPipeline:
      def __init__(self, algorithm='kmeans', n_clusters=None):
          self.algorithm = self._get_algorithm(algorithm)
      
      def fit_predict(self, embeddings)
      def analyze_clusters(self, embeddings, labels)
      def match_clusters(self, labels1, labels2)
      def compute_statistics(self, clusters)
  ```

## 6. Visualization Functions

### Duplicated Patterns Found:

#### behavioural_clustering/utils/visualization.py
- `Visualization` class with plotly/matplotlib
- Directory management for plot types
- Approval prompt visualization
- t-SNE plotting

#### interventions/auto_finetune_eval/auto_finetuning_interp.py
- t-SNE visualization for model comparison
- Cluster-based coloring
- Statistical plot generation

### Opportunities for Consolidation:
- Unified visualization toolkit:
  ```python
  class VisualizationManager:
      def plot_embeddings(self, embeddings, labels, method='tsne')
      def plot_clusters(self, data, cluster_labels)
      def save_plot(self, fig, name, category)
      def create_interactive_plot(self, data)
  ```

## 7. Configuration Management

### Inconsistent Patterns:
- behavioural_clustering: Custom `RunSettings` classes
- soft_prompting: YAML-based configuration
- interventions: Mixed approach (YAML + argparse)
- contrastive-decoding: Primarily argparse

### Recommendation:
- Standardize on YAML + dataclasses/pydantic:
  ```python
  @dataclass
  class BaseConfig:
      model_name: str
      device: str = 'cuda'
      batch_size: int = 32
      
      @classmethod
      def from_yaml(cls, path: str):
          # Load and validate
  ```

## 8. Error Handling and Logging

### Inconsistent Patterns:
- Some modules use structlog, others use standard logging
- Retry logic implemented differently across modules
- API error handling varies significantly

### Recommendation:
- Standardize on structlog with consistent configuration
- Create shared retry decorators
- Unified error handling strategy

## Proposed Shared Utilities Structure

```
src/shared/
├── models/
│   ├── __init__.py
│   ├── base_manager.py      # Base model management
│   ├── api_clients.py       # Unified API clients
│   └── registry.py          # Model registry
├── data/
│   ├── __init__.py
│   ├── loaders.py          # Data loading utilities
│   ├── cache.py            # Caching mechanisms
│   └── preprocessing.py    # Common preprocessing
├── embeddings/
│   ├── __init__.py
│   ├── service.py          # Embedding generation
│   └── cache.py            # Embedding cache
├── clustering/
│   ├── __init__.py
│   ├── algorithms.py       # Clustering implementations
│   └── analysis.py         # Cluster analysis tools
├── visualization/
│   ├── __init__.py
│   ├── plots.py            # Common plotting functions
│   └── interactive.py      # Interactive visualizations
├── config/
│   ├── __init__.py
│   ├── base.py             # Base configuration classes
│   └── loader.py           # Config loading utilities
└── utils/
    ├── __init__.py
    ├── logging.py          # Logging configuration
    ├── retry.py            # Retry decorators
    └── device.py           # Device management
```

## Implementation Priority

1. **High Priority** (Most duplication, easiest wins):
   - API client consolidation
   - Embedding service unification
   - Configuration standardization

2. **Medium Priority** (Significant duplication, moderate effort):
   - Model management unification
   - Data loading consolidation
   - Clustering pipeline standardization

3. **Lower Priority** (Less critical, more complex):
   - Visualization toolkit unification
   - Complete error handling overhaul

## Next Steps

1. Create the shared utilities directory structure
2. Start with high-priority items
3. Gradually migrate existing code to use shared utilities
4. Add comprehensive tests for shared components
5. Update documentation to reflect new structure