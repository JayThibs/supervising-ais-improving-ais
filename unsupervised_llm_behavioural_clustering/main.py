from data_preparation import load_api_key, clone_repo, load_evaluation_data
from utils import query_model  # Add other functions from utils that are used
from model_evaluation import (
    generate_responses,
    embed_responses,
    perform_clustering,
    analyze_clusters,
)
from visualization import plot_dimension_reduction, visualize_hierarchical_clustering


# Include function arguments for variables that are not defined within main()
def main(api_key_file, repo_url, file_paths, texts_subset, llm, prompt):
    # Prepare the data
    api_key = load_api_key(api_key_file)
    clone_repo(repo_url, "evals")
    all_texts = load_evaluation_data(file_paths)

    # Evaluate the model
    generation_results = generate_responses(texts_subset, llm, prompt)
    joint_embeddings_all_llms = embed_responses(generation_results)
    clustering = perform_clustering(joint_embeddings_all_llms)
    rows = analyze_clusters(joint_embeddings_all_llms, clustering)

    # Visualize the results
    plot_dimension_reduction(dim_reduce_tsne, joint_embeddings_all_llms)
    visualize_hierarchical_clustering(
        clustering, approvals_statements_and_embeddings, rows, colors
    )


# When calling main(), pass in the appropriate arguments
if __name__ == "__main__":
    main(api_key_file, repo_url, file_paths, texts_subset, llm, prompt)
