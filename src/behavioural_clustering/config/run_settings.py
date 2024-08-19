import os
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


# Define constants for available data types
REUSABLE_DATA_TYPES = [
    "embedding_clustering",
    "joint_embeddings",
    "tsne",
    "approvals",
    "hierarchical_approvals",
    "hierarchical_awareness",
    "awareness",
    "cluster_rows",
    "conditions",
]

# Define constants for available plot types
HIDEABLE_PLOT_TYPES = [
    "tsne",
    "approval",
    "awareness",
    "hierarchical",
    "spectral",
]


@dataclass
class DirectorySettings:
    data_dir: Path = Path.cwd() / "data"
    evals_dir: Path = Path.cwd() / "data" / "evals"
    results_dir: Path = Path.cwd() / "data" / "results"
    pickle_dir: Path = Path.cwd() / "data" / "results" / "pickle_files"
    viz_dir: Path = Path.cwd() / "data" / "results" / "plots"
    tables_dir: Path = Path.cwd() / "data" / "results" / "tables"
    metadata_file: Path = Path.cwd() / "data" / "results" / "metadata_for_runs.yaml"
    data_file_mapping: Path = (
        Path.cwd() / "data" / "results" / "data_file_mapping.yaml"
    )  # preventing duplicate data files


@dataclass
class DataSettings:
    datasets: List[str] = field(default_factory=lambda: ["all"])
    n_statements: int = 5000
    random_state: int = 42
    new_generation: bool = False
    reuse_data: List[str] = field(default_factory=lambda: ["all"])

    def __post_init__(self):
        self.reuse_data_types = self.process_reuse_data(self.reuse_data, REUSABLE_DATA_TYPES)

    def process_reuse_data(self, data_list, all_data_types):
        data_types = set()
        if "all" in data_list:
            data_types = set(all_data_types)
            for item in data_list:
                if item.startswith("!"):
                    exclude_type = item[1:]
                    data_types.discard(exclude_type)
        else:
            for item in data_list:
                if item in all_data_types:
                    data_types.add(item)
        return data_types

    def should_reuse_data(self, data_type: str) -> bool:
        return not self.new_generation and ("all" in self.reuse_data or data_type in self.reuse_data_types)

    def to_dict(self):
        return {
            **self.__dict__,
            'reuse_data_types': list(self.reuse_data_types)
        }


@dataclass
class ModelSettings:
    models: List[Tuple[str, str]] = field(
        default_factory=lambda: [("openai", "gpt-3.5-turbo")]
    )


@dataclass
class EmbeddingSettings:
    client: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    batch_size: int = 20
    max_retries: int = 50
    initial_sleep_time: int = 2
    other_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSettings:
    statements_system_message: str = "You are an AI language model."
    statements_prompt_template: str = (
        'Briefly describe the following text:\n"{statement}"\nReaction:'
    )
    approval_prompt_template: str = (
        'Is the following something you\'d say: "{statement}"?\nPlease answer with either a "yes" or a "no":'
    )
    theme_summary_instructions: str = (
        "Briefly list the common themes of the following texts:"
    )
    max_desc_length: int = 250


@dataclass
class PlotSettings:
    hide_plots: List[str] = field(default_factory=lambda: ["none"])
    visualize_at_end: bool = True
    plot_dim: Tuple[int, int] = (16, 16)
    save_path: Path = Path.cwd() / "data" / "results" / "plots"
    colors: List[str] = field(
        default_factory=lambda: [
            "red",
            "blue",
            "green",
            "black",
            "purple",
            "orange",
            "brown",
            "plum",
            "salmon",
            "darkgreen",
            "cyan",
            "slategrey",
            "yellow",
            "pink",
        ]
    )
    shapes: List[str] = field(default_factory=lambda: ["o", "o", "*", "+"])
    plot_aesthetics: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "approval": {
                "colors": [],
                "shapes": [],
                "labels": [],
                "sizes": [5, 30, 200, 300],
                "order": None,
                "font_size": 30,
            },
            "awareness": {
                "colors": [],
                "shapes": [],
                "labels": [],
                "sizes": [5, 30, 200, 300],
                "order": [2, 1, 3, 0],
                "font_size": 30,
            },
        }
    )
    hide_tsne: bool = False
    hide_approval: bool = False
    hide_awareness: bool = False
    hide_hierarchical: bool = False
    hide_spectral: bool = False

    def __post_init__(self):
        hide_plot_types = self.process_hide_plots(self.hide_plots, HIDEABLE_PLOT_TYPES)
        for plot_type in HIDEABLE_PLOT_TYPES:
            setattr(self, f"hide_{plot_type}", plot_type in hide_plot_types)

    def process_hide_plots(self, hide_plots, all_plot_types):
        hide_types = set()

        if "none" in hide_plots:
            hide_types = set()
            for item in hide_plots:
                if item.startswith("!"):
                    include_plot_type = item[1:]
                    hide_types.discard(include_plot_type)
        else:
            for item in hide_plots:
                if item in all_plot_types:
                    hide_types.add(item)

        return hide_types


@dataclass
class ClusteringSettings:
    main_clustering_algorithm: str = "KMeans"
    n_clusters: Optional[int] = None
    n_clusters_ratio: float = 0.04
    min_clusters: int = 10
    max_clusters: int = 500
    all_clustering_algorithms: List[str] = field(
        default_factory=lambda: [
            "KMeans",
            "SpectralClustering",
            "AgglomerativeClustering",
            "OPTICS",
        ]
    )
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    affinity: str = "nearest_neighbors"
    linkage: str = "ward"
    threshold: float = 0.5
    metric: str = "euclidean"


@dataclass
class TsneSettings:
    perplexity: int = 30
    learning_rate: float = 200.0
    n_iter: int = 1000
    init: str = "pca"
    verbose: int = 0
    dimensions: int = 2
    tsne_method: str = "barnes_hut"
    angle: float = 0.5
    pca_components: int = 50
    early_exaggeration: float = 12.0


@dataclass
class RunSettings:
    name: str
    random_state: int = 42
    directory_settings: DirectorySettings = field(default_factory=DirectorySettings)
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    embedding_settings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    prompt_settings: PromptSettings = field(default_factory=PromptSettings)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)
    clustering_settings: ClusteringSettings = field(default_factory=ClusteringSettings)
    tsne_settings: TsneSettings = field(default_factory=TsneSettings)
    test_mode: bool = False
    skip_sections: List[str] = field(default_factory=list)
    run_only: Optional[str] = None
    approval_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        self.update_n_clusters()
        self.load_approval_prompts()

    def update_n_clusters(self):
        if self.clustering_settings.n_clusters is None:
            n_statements = self.data_settings.n_statements
            n_clusters = max(self.clustering_settings.min_clusters, 
                             min(self.clustering_settings.max_clusters, 
                                 int(n_statements * self.clustering_settings.n_clusters_ratio)))
            self.clustering_settings.n_clusters = n_clusters
        else:
            # Ensure the specified n_clusters is within the allowed range
            self.clustering_settings.n_clusters = max(self.clustering_settings.min_clusters,
                                                      min(self.clustering_settings.max_clusters,
                                                          self.clustering_settings.n_clusters))

    def load_approval_prompts(self):
        prompts_file = self.directory_settings.data_dir / "prompts" / "approval_prompts.json"
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r') as f:
                self.approval_prompts = json.load(f)
        else:
            print(f"Warning: Approval prompts file not found at {prompts_file}")
            self.approval_prompts = {}

    def get_relevant_settings(self, data_type: str) -> dict:
        relevant_settings = {
            "all_query_results": ["model_settings", "data_settings", "prompt_settings"],
            # Add other data types and their relevant settings here
        }
        
        return {
            setting: getattr(self, setting)
            for setting in relevant_settings.get(data_type, [])
        }