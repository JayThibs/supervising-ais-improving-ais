# These scripts can be used to run the clustering algorithm on the data.
# As part of the scripts directory, the focus is to make it easy to
# rerun parts of the code, e.g. clustering, visualization, CD, etc.
# run_clustering.py

import json
from dataclasses import asdict, dataclass
from unsupervised_llm_behavioural_clustering.clustering import (
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
