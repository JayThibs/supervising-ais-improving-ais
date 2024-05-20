# These scripts can be used to run the clustering algorithm on the data.
# As part of the scripts directory, the focus is to make it easy to
# rerun parts of the code, e.g. clustering, visualization, CD, etc.
# run_clustering.py

import json
from dataclasses import asdict, dataclass
from behavioural_clustering.clustering import (
    ClusteringPipeline,
    visualize_clusters,
)


@dataclass
class PipelineState:
    combined_embeddings: np.ndarray
    chosen_clustering: ClusteringModel


def main():
    # Load previous run pipeline state
    state_file = "previous_state.json"
    with open(state_file) as f:
        state_dict = json.load(f)
    state = PipelineState(**state_dict)

    # Rerun clustering using the embeddings
    clusterer = ClusteringAlgorithm()
    new_clustering = clusterer.fit(state.combined_embeddings)

    # Visualize clusters
    visualize_clusters(new_clustering)

    # Save new state
    new_state = PipelineState(
        combined_embeddings=state.combined_embeddings, chosen_clustering=new_clustering
    )
    new_state_dict = asdict(new_state)
    with open(state_file, "w") as f:
        json.dump(new_state_dict, f)


if __name__ == "__main__":
    main()


# ---------------


@dataclass
class ClusteringPipelineState:
    text_subset: list
    query_results_per_model: dict
    model_info_list: dict
    joint_embeddings: np.ndarray
    combined_embeddings: np.ndarray
    chosen_clustering: Any  # Actual cluster object
    approvals_statements_and_embeddings: list
    statement_embeddings: np.ndarray


def save_pipeline_state(state: ClusteringPipelineState, filepath: str):
    # Convert dataclass to dict to serialize
    state_dict = asdict(state)

    # Save to JSON file
    with open(filepath, "w") as f:
        json.dump(state_dict, f)


def load_pipeline_state(filepath: str) -> ClusteringPipelineState:
    with open(filepath) as f:
        state_dict = json.load(f)

    return ClusteringPipelineState(**state_dict)


if __name__ == "__main__":
    # Run pipeline
    state = run_pipeline()

    # Save state
    save_pipeline_state(state, "pipeline_state.json")

    # Reload later
    reloaded_state = load_pipeline_state("pipeline_state.json")

    # Redo clustering, analysis, etc.
    redo_clustering(reloaded_state)
