import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Optional, Union, List
from behavioural_clustering.config.run_settings import TsneSettings

def tsne_reduction(
    combined_embeddings: Union[np.ndarray, List[np.ndarray]],
    tsne_settings: Optional[TsneSettings] = None,
    n_components: int = 2,
    perplexity: int = 50,
    n_iter: int = 2000,
    angle: float = 0.0,
    init: str = "pca",
    early_exaggeration: float = 1.0,
    learning_rate: Union[float, str] = "auto",
    random_state: int = 42,
) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction on the given embeddings.
    For small datasets (n_samples <= 30), automatically falls back to PCA.

    Args:
        combined_embeddings: Input embeddings to reduce.
        tsne_settings: Optional TsneSettings object to override default parameters.
        n_components: Number of dimensions in the embedded space.
        perplexity: Related to the number of nearest neighbors used in other manifold learning algorithms.
        n_iter: Maximum number of iterations for the optimization.
        angle: Angular size of a distant node as measured from a point in the t-SNE plot.
        init: Initialization of embedding.
        early_exaggeration: Controls how tight natural clusters in the original space are in the embedded space.
        learning_rate: The learning rate for t-SNE.
        random_state: Determines the random number generator.

    Returns:
        Reduced embeddings.
    """
    # Convert combined_embeddings to numpy array if it's a list
    if isinstance(combined_embeddings, list):
        combined_embeddings = np.array(combined_embeddings)
    
    # Ensure data is in float64 format
    combined_embeddings = combined_embeddings.astype(np.float64)
    
    n_samples = combined_embeddings.shape[0]
    
    if n_samples <= 30:
        print(f"Small dataset detected (n={n_samples}). Using PCA instead of t-SNE.")
        return pca_reduction(combined_embeddings, n_components=n_components)
    
    print("Performing t-SNE dimensionality reduction...")
    
    if tsne_settings:
        n_components = tsne_settings.dimensions
        perplexity = tsne_settings.perplexity
        n_iter = tsne_settings.n_iter
        angle = tsne_settings.angle
        init = tsne_settings.init
        early_exaggeration = tsne_settings.early_exaggeration
        learning_rate = tsne_settings.learning_rate
    
    perplexity = min(perplexity, n_samples - 1)
    
    if isinstance(init, str) and init not in ["random", "pca"]:
        init = "pca"  # Default to PCA if invalid string
    
    if isinstance(learning_rate, str) and learning_rate != "auto":
        try:
            learning_rate = float(learning_rate)
        except ValueError:
            learning_rate = "auto"  # Default to auto if invalid string

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        angle=angle,
        init=init,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    
    try:
        dim_reduce_tsne = tsne.fit_transform(X=combined_embeddings)
        check_tsne_values(dim_reduce_tsne)
        return dim_reduce_tsne
    except Exception as e:
        print(f"Error during t-SNE reduction: {str(e)}")
        print(f"Input shape: {combined_embeddings.shape}")
        print(f"Input dtype: {combined_embeddings.dtype}")
        print("Falling back to PCA...")
        return pca_reduction(combined_embeddings, n_components=n_components)

def check_tsne_values(dim_reduce_tsne: np.ndarray) -> None:
    """
    Check the validity of t-SNE reduced values.

    Args:
        dim_reduce_tsne: The t-SNE reduced embeddings to check.
    """
    if not np.isfinite(dim_reduce_tsne).all():
        print("Warning: dim_reduce_tsne contains non-finite values.")
    if np.isnan(dim_reduce_tsne).any():
        print("Warning: dim_reduce_tsne contains NaN values.")
    if np.isinf(dim_reduce_tsne).any():
        print("Warning: dim_reduce_tsne contains inf values.")
    print(f"dim_reduce_tsne shape: {dim_reduce_tsne.shape}")
    print(f"dim_reduce_tsne dtype: {dim_reduce_tsne.dtype}")

def pca_reduction(combined_embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Perform PCA dimensionality reduction on the given embeddings.

    Args:
        combined_embeddings: Input embeddings to reduce.
        n_components: Number of dimensions in the embedded space.

    Returns:
        Reduced embeddings.
    """
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=n_components)
    dim_reduce_pca = pca.fit_transform(combined_embeddings)
    return dim_reduce_pca
