import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class ModelSettings:
    models: List[Tuple[str, str]] = field(
        default_factory=lambda: [("openai", "gpt-3.5-turbo")]
    )


@dataclass
class DataSettings:
    n_statements: int = 5000
    texts_subset: int = 10
    new_generation: bool = False
    reuse_data: List[str] = field(default_factory=lambda: ["all"])


@dataclass
class PromptSettings:
    statements_system_message: str = "You are an AI language model."
    statements_prompt_template: str = (
        'Briefly describe the following text:\n"{statement}"\nReaction:'
    )
    approval_prompt_template: str = (
        'Is the following something you\'d say: "{statement}"?\nPlease answer with either a "yes" or a "no":'
    )


@dataclass
class PlotSettings:
    hide_plots: List[str] = field(default_factory=lambda: ["all"])
    plot_dim: Tuple[int, int] = (16, 16)
    save_path: str = f"{os.getcwd()}/data/results/plots"
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


@dataclass
class ClusteringSettings:
    n_clusters: int = 200
    cluster_type: str = "spectral"
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
    tsne_init: str = "pca"
    verbose: int = 0
    random_state: int = 42
    dimensions: int = 2
    tsne_method: str = "barnes_hut"
    angle: float = 0.5
    pca_components: int = 50


@dataclass
class RunSettings:
    name: str
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    data_settings: DataSettings = field(default_factory=DataSettings)
    prompt_settings: PromptSettings = field(default_factory=PromptSettings)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)
    clustering_settings: ClusteringSettings = field(default_factory=ClusteringSettings)
    tsne_settings: TsneSettings = field(default_factory=TsneSettings)
    test_mode: bool = False
    skip_sections: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.data_settings.reuse_data == ["all"]:
            self.data_settings.reuse_data = [
                "tsne",
                "approvals",
                "hierarchical",
                "awareness",
                "rows",
                "conditions",
            ]
        if self.plot_settings.hide_plots == ["all"]:
            self.plot_settings.hide_plots = [
                "tsne",
                "approval",
                "awareness",
                "hierarchical",
            ]
