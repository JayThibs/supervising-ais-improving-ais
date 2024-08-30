import os
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict, fields
from typing import List, Tuple, Dict, Any, Optional, Union


# Define constants for available data types
REUSABLE_DATA_TYPES = [
    "embedding_clustering",
    "joint_embeddings",
    "tsne",
    "approvals",
    "hierarchical_approvals",
    "cluster_rows",
    "conditions",
]

# Define constants for available plot types
HIDEABLE_PLOT_TYPES = [
    "tsne",
    "approvals",
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
    temperature: float = 0.01
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
    max_desc_length: int = 250
    awareness_task: str = "evaluate text appropriateness"
    approval_prompt_template: str = "Given the following statement, would you approve of it? Please answer with either 'yes' or 'no'.\n\nStatement: {statement}\n\nApproval (yes / no):"


@dataclass
class PlotSettings:
    hide_plots: List[str] = field(default_factory=lambda: [])
    visualize_at_end: bool = True
    plot_dim: Tuple[int, int] = (16, 16)
    save_path: Path = field(default_factory=lambda: Path.cwd() / "data" / "results" / "plots")
    colors: List[str] = field(default_factory=lambda: [
        "#FF0000", "#0000FF", "#00FF00", "#800080", "#FFA500", "#A52A2A",
        "#FFC0CB", "#00FFFF", "#808080", "#FFFF00", "#FF00FF", "#008080",
        "#000080", "#800000", "#008000", "#808000"
    ])
    shapes: List[str] = field(default_factory=lambda: [
        "o", "s", "^", "D", "v", "p", "h", "*", "8", "+", "x", "d",
        "|", "_", "1", "2", "3", "4"
    ])
    plot_aesthetics: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "approvals": {
            "colors": [],
            "shapes": [],
            "labels": [],
            "sizes": [30, 60, 90, 120],
            "order": None,
            "font_size": 12,
            "legend_font_size": 10,
            "marker_size": 100,
            "alpha": 0.7,
        },
    })
    hide_model_comparison: bool = True
    hide_approvals: bool = True
    hide_hierarchical: bool = True
    hide_interactive_treemap: bool = True
    hide_spectral: bool = True

    def __post_init__(self):
        if "all" in self.hide_plots:
            self.hide_plots = HIDEABLE_PLOT_TYPES.copy()
        elif "none" in self.hide_plots:
            self.hide_plots = []
        else:
            self.hide_plots = [plot for plot in self.hide_plots if plot in HIDEABLE_PLOT_TYPES]
        
        self.approval_prompts = self.load_approval_prompts()
        self.hidden_approval_prompts = []
        self._update_hide_settings()
        self._update_plot_aesthetics()

    def _update_hide_settings(self):
        if "none" in self.hide_plots:
            self.hide_approvals = False
            self.hide_model_comparison = False
            self.hide_hierarchical = False
            self.hide_interactive_treemap = False
            self.hide_spectral = False
            self.hidden_approval_prompts = []
        elif "all" in self.hide_plots:
            self.hide_approvals = True
            self.hide_model_comparison = True
            self.hide_hierarchical = True
            self.hide_interactive_treemap = True
            self.hide_spectral = True
            self.hidden_approval_prompts = self.approval_prompts.copy()
        else:
            for plot_type in self.hide_plots:
                if plot_type == "approvals":
                    self.hide_approvals = True
                elif plot_type.startswith("approval_"):
                    prompt_type = plot_type.split("_", 1)[1]
                    if prompt_type in self.approval_prompts:
                        self.hidden_approval_prompts.append(prompt_type)
                elif plot_type in ["model_comparison", "hierarchical", "spectral"]:
                    setattr(self, f"hide_{plot_type}", True)

    def load_approval_prompts(self) -> List[str]:
        try:
            # Get the project root directory (two levels up from src)
            project_root = Path(__file__).resolve().parents[3]
            file_path = project_root / "data" / "prompts" / "approval_prompts.json"
            with open(file_path, "r") as f:
                return list(json.load(f).keys())
        except FileNotFoundError:
            print(f"Warning: approval_prompts.json not found at {file_path}. Using default prompt types.")
            return ["personas", "awareness"]

    def should_hide_approval_plot(self, prompt_type: str) -> bool:
        return self.hide_approvals or prompt_type in self.hidden_approval_prompts

    def _update_plot_aesthetics(self):
        # Load the approval prompts
        with open(Path(__file__).resolve().parents[3] / "data" / "prompts" / "approval_prompts.json", "r") as f:
            approval_prompts = json.load(f)
        
        for category in self.approval_prompts:
            prompts = approval_prompts[category]
            num_prompts = len(prompts)
            
            self.plot_aesthetics[f"{category}_approvals"] = {
                "colors": self.colors[:num_prompts],
                "shapes": self.shapes[:num_prompts],
                "labels": list(prompts.keys()),
                "sizes": [self.plot_aesthetics["approvals"]["marker_size"]] * num_prompts,  # Use a single size
                "order": None,
                "font_size": self.plot_aesthetics["approvals"]["font_size"],
                "legend_font_size": self.plot_aesthetics["approvals"]["legend_font_size"],
                "marker_size": self.plot_aesthetics["approvals"]["marker_size"],
                "alpha": self.plot_aesthetics["approvals"]["alpha"],
            }


@dataclass
class ClusteringSettings:
    main_clustering_algorithm: str = "KMeans"
    n_clusters_ratio: float = 0.04
    n_clusters: int = None
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
    theme_identification_model_name: str = "gpt-4o-mini"
    theme_identification_model_family: str = "openai"
    theme_identification_system_message: str = ""
    theme_identification_prompt: str = (
        "Briefly list the common themes of the following texts:"
    )


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

    @staticmethod
    def calculate_perplexity(n_statements: int) -> int:
        if n_statements < 50:
            return max(5, n_statements // 3)
        elif n_statements < 100:
            return max(10, n_statements // 5)
        elif n_statements < 500:
            return max(30, n_statements // 10)
        else:
            return min(50, n_statements // 100)


@dataclass
class RunSettings:
    name: str = "default"
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
    run_sections: List[str] = field(default_factory=list)
    approval_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    run_only: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.load_approval_prompts()
        self.update_n_clusters()
        self.update_tsne_settings()

    def update_n_clusters(self):
        if self.clustering_settings.n_clusters is None:
            n_clusters = max(self.clustering_settings.min_clusters, 
                             min(self.clustering_settings.max_clusters, 
                                 int(self.data_settings.n_statements * self.clustering_settings.n_clusters_ratio)))
            self.clustering_settings.n_clusters = n_clusters

    def update_tsne_settings(self):
        self.tsne_settings.perplexity = TsneSettings.calculate_perplexity(self.data_settings.n_statements)

    def load_approval_prompts(self):
        self.approval_prompts_file = self.directory_settings.data_dir / "prompts" / "approval_prompts.json"
        if os.path.exists(self.approval_prompts_file):
            with open(self.approval_prompts_file, 'r') as f:
                self.approval_prompts = json.load(f)
        else:
            print(f"Warning: Approval prompts file not found at {self.approval_prompts_file}")
            self.approval_prompts = {}

    def update_run_sections(self, sections: Optional[Union[str, List[str]]] = None):
        # Define available sections
        available_sections = ["model_comparison", "hierarchical_clustering"]
        available_sections.extend([f"{prompt_type}_evaluation" for prompt_type in self.approval_prompts.keys()])

        if sections is not None:
            if isinstance(sections, str):
                sections = [sections]
            if "all" in sections:
                self.run_sections = available_sections
            else:
                valid_sections = [section for section in sections if section in available_sections]
                if valid_sections:
                    self.run_sections = valid_sections
                else:
                    print(f"Warning: No valid sections provided. Run sections unchanged: {self.run_sections}")
        else:
            self.run_sections = [section for section in self.run_sections if section in available_sections]

        print(f"Updated run sections: {self.run_sections}")

    def to_dict(self) -> dict:
        def convert_paths(item):
            if isinstance(item, Path):
                return str(item)
            elif isinstance(item, dict):
                return {k: convert_paths(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_paths(i) for i in item]
            return item

        return json.loads(json.dumps(asdict(self), default=convert_paths))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunSettings':
        def convert_to_path(item):
            if isinstance(item, str) and ('dir' in item or 'path' in item):
                return Path(item)
            elif isinstance(item, dict):
                return {k: convert_to_path(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_to_path(i) for i in item]
            return item

        # Convert string paths back to Path objects
        data = convert_to_path(data)

        # Remove any unexpected keys
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        # Convert nested dictionaries to appropriate dataclass objects
        for key, value in filtered_data.items():
            if key.endswith('_settings') and isinstance(value, dict):
                setting_class = globals()[key.replace('_settings', '').capitalize() + 'Settings']
                filtered_data[key] = setting_class(**value)

        return cls(**filtered_data)