import os
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

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
    "hierarchical_approvals",
    "hierarchical_awareness",
    "spectral",
]


@dataclass
class ModelSettings:
    models: List[Tuple[str, str]] = field(
        default_factory=lambda: [("openai", "gpt-3.5-turbo")]
    )


@dataclass
class DataSettings:
    n_statements: int = 5000
    new_generation: bool = False
    reuse_data: List[str] = field(default_factory=lambda: ["all"])
    reuse_embedding_clustering: bool = False
    reuse_joint_embeddings: bool = False
    reuse_tsne: bool = False
    reuse_approvals: bool = False
    reuse_hierarchical_approvals: bool = False
    reuse_hierarchical_awareness: bool = False
    reuse_awareness: bool = False
    reuse_cluster_rows: bool = False
    reuse_conditions: bool = False

    def __post_init__(self):
        reuse_data_types = self.process_reuse_data(self.reuse_data, REUSABLE_DATA_TYPES)
        for data_type in REUSABLE_DATA_TYPES:
            setattr(self, f"reuse_{data_type}", data_type in reuse_data_types)

    def process_reuse_data(self, reuse_data, all_data_types):
        reuse_types = set()

        if "all" in reuse_data:
            reuse_types = set(all_data_types)
            for item in reuse_data:
                if item.startswith("!"):
                    exclude_type = item[1:]
                    reuse_types.discard(exclude_type)
        else:
            for item in reuse_data:
                if item in all_data_types:
                    reuse_types.add(item)

        return reuse_types


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

    def __post_init__(self):
        self.hide_plot_types = self.process_hide_plots(
            self.hide_plots, HIDEABLE_PLOT_TYPES
        )

    def process_hide_plots(self, hide_plots, all_plot_types):
        hide_types = []

        if "all" not in hide_plots:
            hide_types = all_plot_types
        else:
            for plot_type in hide_plots:
                if plot_type.startswith("!"):
                    include_plot_type = plot_type[1:]
                    if include_plot_type in all_plot_types:
                        hide_types.remove(include_plot_type)

        return hide_types


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


# @dataclass
# class ModelMetadata:
#     model_id: str
#     temperature: float
#     max_tokens: int
#     other_params: Dict[str, Any]  # Include any other model-specific parameters


# @dataclass
# class StatementResponses:
#     metadata: ModelMetadata
#     response_file: str  # Path to the .jsonl file with responses

#     def save_responses(self, statements, responses):
#         with open(self.response_file, "w") as f:
#             for statement, response in zip(statements, responses):
#                 json.dump({"statement": statement, "response": response}, f)
#                 f.write("\n")

#     @staticmethod
#     def load_responses(filepath):
#         statements, responses = [], []
#         with open(filepath, "r") as f:
#             for line in f:
#                 data = json.loads(line)
#                 statements.append(data["statement"])
#                 responses.append(data["response"])
#         return statements, responses


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
