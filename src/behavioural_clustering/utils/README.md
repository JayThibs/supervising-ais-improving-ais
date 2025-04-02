# Behavioral Clustering Utilities

This directory contains utility modules for the behavioral clustering system. These utilities provide essential functionality for dataset management, clustering algorithms, hardware detection, and visualization.

## Dataset Loader

The `dataset_loader.py` module provides a flexible system for managing and loading datasets for behavioral clustering.

### Key Features

- **Dataset Configuration**: Define dataset configurations with customizable parameters
- **Dataset Registry**: Manage multiple dataset configurations
- **Dataset Loading**: Load and filter datasets based on criteria
- **Metadata Extraction**: Extract and use metadata from datasets

### Example Usage

```python
from behavioural_clustering.utils.dataset_loader import create_default_registry, DatasetLoader

# Create a default registry with common datasets
data_dir = Path("/path/to/data")
registry = create_default_registry(data_dir)

# Create a loader
loader = DatasetLoader(registry)

# Load a dataset
statements, accepted_count, rejected_count = loader.load_dataset("anthropic_evals")

# Filter statements
filtered_statements = loader.filter_statements(
    statements, 
    max_length=500, 
    min_length=10, 
    categories=["reasoning", "ethics"]
)
```

## Clustering Algorithms

The `clustering_algorithms.py` module provides a flexible system for applying different clustering algorithms to embedding data.

### Key Features

- **Algorithm Abstraction**: Common interface for all clustering algorithms
- **Multiple Algorithms**: K-means, Spectral, Agglomerative, DBSCAN, GMM, and UMAP-based clustering
- **Factory Pattern**: Create algorithms by name
- **Evaluation Metrics**: Evaluate clustering quality using various metrics
- **Optimal Clusters**: Find the optimal number of clusters

### Example Usage

```python
from behavioural_clustering.utils.clustering_algorithms import ClusteringFactory, evaluate_clustering, find_optimal_clusters
import numpy as np

# Create embedding data
embeddings = np.array([[1, 2], [3, 4], [1, 3], [4, 2], ...])

# Create a clustering algorithm
algorithm = ClusteringFactory.create_algorithm(
    "kmeans", 
    n_clusters=5, 
    random_state=42
)

# Fit the algorithm to the data
labels = algorithm.fit(embeddings)

# Evaluate clustering quality
metrics = evaluate_clustering(embeddings, labels)
print(f"Silhouette score: {metrics['silhouette']}")

# Find the optimal number of clusters
optimal_n_clusters, optimal_labels, all_metrics = find_optimal_clusters(
    embeddings, 
    algorithm="spectral", 
    min_clusters=2, 
    max_clusters=10
)
```

## Hardware Detection

The `hardware_detection.py` module provides utilities for detecting and managing hardware resources for LLM usage.

### Key Features

- **Hardware Detection**: Detect CPU, memory, and GPU resources
- **Optimal Device Selection**: Select the optimal device for running models
- **Batch Size Calculation**: Calculate optimal batch sizes based on available memory
- **Model Configuration**: Determine optimal model configurations based on hardware
- **Parallel Execution**: Configure models for parallel execution across multiple GPUs

### Example Usage

```python
from behavioural_clustering.utils.hardware_detection import get_hardware_info, configure_models_for_hardware

# Get hardware information
hardware = get_hardware_info()
hardware.print_hardware_summary()

# Get the optimal device
device = hardware.get_optimal_device()
print(f"Optimal device: {device}")

# Configure models based on hardware
model_configs = configure_models_for_hardware()
optimal_model = model_configs['optimal_model']
print(f"Optimal model: {optimal_model['name']} on {optimal_model['device']}")

# Get parallel configurations for a model
parallel_configs = hardware.get_parallel_model_configs(
    model_configs['available_models'], 
    "llama-2-7b"
)
print(f"Can run {len(parallel_configs)} instances of llama-2-7b in parallel")
```

## Visualization

The `visualization.py` module provides utilities for visualizing clustering results and model comparisons.

### Key Features

- **Interactive Visualizations**: Create interactive visualizations using Plotly
- **Embedding Visualization**: Visualize embeddings in 2D or 3D space
- **Cluster Visualization**: Visualize clusters with different colors and markers
- **Model Comparison**: Compare responses from different models
- **Approval Visualization**: Visualize approval patterns across models

### Example Usage

```python
from behavioural_clustering.utils.visualization import plot_embedding_responses_plotly, plot_approvals_plotly

# Plot embeddings with responses
fig = plot_embedding_responses_plotly(
    embeddings_2d, 
    labels, 
    responses, 
    model_names, 
    title="Model Response Embeddings"
)
fig.show()

# Plot approval patterns
fig = plot_approvals_plotly(
    approval_matrix, 
    model_names, 
    cluster_labels, 
    title="Model Approval Patterns"
)
fig.show()
```

## Data Preparation

The `data_preparation.py` module provides utilities for preparing and managing data for behavioral clustering.

### Key Features

- **Data Loading**: Load data from various sources
- **Data Validation**: Validate data for consistency and completeness
- **Data Transformation**: Transform data into the required format
- **Data Storage**: Store and retrieve data with metadata

## Embedding Data

The `embedding_data.py` module provides utilities for managing embedding data for behavioral clustering.

### Key Features

- **Embedding Storage**: Store embeddings with associated metadata
- **Model Tracking**: Track which models generated which embeddings
- **Data Validation**: Validate embedding data for consistency and completeness
- **Matrix Conversion**: Convert embeddings to matrices for analysis
