import numpy as np
from sklearn.manifold import TSNE
from typing import Optional
from behavioural_clustering.config.run_settings import TsneSettings

class DimensionalityReduction:
    @staticmethod
    def tsne_reduction(
        combined_embeddings,
        tsne_settings: Optional[TsneSettings] = None,
        n_components: int = 2,
        perplexity: int = 50,
        n_iter: int = 2000,
        angle: float = 0.0,
        init: str = "pca",
        early_exaggeration: float = 1.0,
        learning_rate: str = "auto",
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Perform t-SNE dimensionality reduction on the given embeddings.

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
        print("Performing t-SNE dimensionality reduction...")
        if tsne_settings:
            n_components = tsne_settings.dimensions
            perplexity = tsne_settings.perplexity
            n_iter = tsne_settings.n_iter
            angle = tsne_settings.angle
            init = tsne_settings.init
            early_exaggeration = tsne_settings.early_exaggeration
            learning_rate = tsne_settings.learning_rate

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
        dim_reduce_tsne = tsne.fit_transform(X=combined_embeddings)
        DimensionalityReduction._check_tsne_values(dim_reduce_tsne)
        return dim_reduce_tsne

    @staticmethod
    def _check_tsne_values(dim_reduce_tsne: np.ndarray) -> None:
        """
        Check the validity of t-SNE reduced values.

        Args:
            dim_reduce_tsne: The t-SNE reduced embeddings to check.
        """
        if not np.isfinite(dim_reduce_tsne).all():
            print("Warning: dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("Warning: dim_reduce_tsne contains NaN or inf values.")
        print(f"dim_reduce_tsne dtype: {dim_reduce_tsne.dtype}")

    @staticmethod
    def pca_reduction(combined_embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Perform PCA dimensionality reduction on the given embeddings.

        Args:
            combined_embeddings: Input embeddings to reduce.
            n_components: Number of dimensions in the embedded space.

        Returns:
            Reduced embeddings.
        """
        from sklearn.decomposition import PCA

        print("Performing PCA dimensionality reduction...")
        pca = PCA(n_components=n_components)
        dim_reduce_pca = pca.fit_transform(combined_embeddings)
        return dim_reduce_pca
