# Files with Most Duplication

## Critical Files to Refactor

### 1. Model Management Files
- **behavioural_clustering/models/model_factory.py**
- **behavioural_clustering/models/api_models.py**
- **soft_prompting/models/model_manager.py**
- **interventions/intervention_models/model_manager.py**
- **contrastive-decoding/model_comparison_helpers.py**

**Why**: Each implements model loading, API clients, and device management differently.

### 2. Data Handling Files
- **behavioural_clustering/utils/data_preparation.py** (424 lines)
- **interventions/auto_finetune_eval/auto_finetuning_data.py** (301 lines)
- **interventions/auto_finetune_eval/auto_finetuning_helpers.py** (1859 lines!)
- **contrastive-decoding/model_comparison_helpers.py** (530 lines)

**Why**: JSONL loading, caching, and dataset preparation reimplemented multiple times.

### 3. API Request Handling
- **behavioural_clustering/models/api_models.py**
- **interventions/auto_finetune_eval/auto_finetuning_helpers.py** (make_api_request function)
- **interventions/evaluation/automated_evaluator.py**
- **contrastive-decoding/validated_analysis.py**

**Why**: Each has different retry logic, error handling, and client initialization.

### 4. Embedding Management
- **behavioural_clustering/utils/embedding_utils.py**
- **behavioural_clustering/utils/embedding_manager.py**
- Embedding logic scattered in intervention and contrastive-decoding modules

**Why**: No unified embedding service despite similar needs across modules.

### 5. Clustering Implementation
- **behavioural_clustering/evaluation/clustering.py**
- **interventions/auto_finetune_eval/auto_finetuning_interp.py**
- **contrastive-decoding/quick_cluster.py**

**Why**: Same sklearn algorithms used with different wrappers and analysis methods.

## Highest Impact Refactoring Targets

### 1. auto_finetuning_helpers.py (1859 lines)
This massive file contains:
- API request functions (duplicated)
- Data loading utilities (duplicated)
- Model generation helpers (partially duplicated)
- Evaluation metrics (some duplication)
- Visualization functions (duplicated patterns)

**Recommendation**: Break into smaller modules and extract shared functionality.

### 2. data_preparation.py (424 lines)
Contains both:
- `DataPreparation` class (JSONL loading)
- `DataHandler` class (caching and metadata)

**Recommendation**: Split into separate modules and create shared base classes.

### 3. model_comparison_helpers.py (530 lines)
Complex file mixing:
- Model instantiation
- Contrastive decoding logic
- Data loading
- Visualization helpers

**Recommendation**: Extract model management and data loading to shared utilities.

## Quick Wins

### 1. API Client Factory (1-2 days)
```python
# Extract from multiple files into shared/api/client_factory.py
class APIClientFactory:
    @staticmethod
    def create_client(provider: str, **kwargs):
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "anthropic":
            return AnthropicClient(**kwargs)
        # etc.
```

### 2. Unified File Operations (1 day)
```python
# Extract from all modules into shared/utils/io.py
class FileManager:
    @staticmethod
    def save_pickle(data, path):
        # Unified pickle saving
    
    @staticmethod
    def load_pickle(path):
        # Unified pickle loading
    
    @staticmethod
    def save_json(data, path):
        # Unified JSON operations
```

### 3. Device Management (1 day)
```python
# Extract into shared/utils/device.py
def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model_kwargs(device):
    # Unified model loading kwargs
```

### 4. Embedding Cache (2-3 days)
```python
# Create shared/embeddings/cache.py
class EmbeddingCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
    
    def get_or_create(self, texts, model, provider="openai"):
        # Unified embedding caching
```

## Metrics

### Estimated Code Reduction
- API handling: ~500 lines → ~200 lines (60% reduction)
- Data loading: ~800 lines → ~300 lines (62% reduction)
- Model management: ~700 lines → ~400 lines (43% reduction)
- Embedding handling: ~300 lines → ~150 lines (50% reduction)

### Total Potential Reduction
- Current: ~2,300 lines of duplicated code
- After refactoring: ~1,050 lines
- **Net reduction: ~1,250 lines (54%)**

## Implementation Order

1. **Week 1**: API clients and file I/O utilities
2. **Week 2**: Model management base classes
3. **Week 3**: Data loading and caching framework
4. **Week 4**: Embedding service and clustering utilities