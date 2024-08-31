## Clustering and Analysis

These files are used for clustering and analyzing the outputs of contrastive decoding experiments.

### quick_cluster.py

This file contains the main functionality for clustering and analyzing text data generated from contrastive decoding. The core operational loop is as follows:

1. Load two collections of texts: a 'starting' collection (more indicative of the starting model's behavior) and a 'comparison' collection (more indicative of the comparison model's behavior).
2. Embed both sets of texts using the model specified by `local_embedding_model_str`. By default, these embeddings will be saved to pickle files (and loaded from those files if they're already present unless `recompute_embeddings` is set).
3. Cluster both sets of texts independently in their embedding spaces.
4. Use an assistant model (local or via OpenAI API) to:
   a. Assign labels to both sets of clusters based on sampled texts from each cluster.
   b. Validate these labels by checking if the assistant can distinguish between held-out texts in the cluster and other texts from the collection.
5. Perform pairwise matching between clusters in the starting and comparison sets.
6. For each cluster pair, generate a 'contrastive' text label describing the difference between the two sides.
7. Validate contrastive labels by testing if the assistant can use them to distinguish between texts from the 'starting' and 'comparison' sides.
8. Further test contrastive labels by having the assistant generate texts for each model and evaluating their relative probabilities.

Key features and components:

- Text Embedding: Uses pre-trained language models to generate embeddings for input texts.
- Clustering: Implements various clustering algorithms (e.g., K-means, HDBSCAN) on the embeddings.
- Label Generation and Validation: Uses language models to generate descriptive labels for clusters and contrastive labels for cluster pairs, validating them using AUC scores and p-values.
- Comparative Analysis: Allows comparison between different datasets or models.
- Generative Validation: Tests contrastive labels by generating new texts and evaluating their model probabilities.
- AUC Scoring and P-value Computation: Evaluates the effectiveness of generated labels and computes statistical significance.


Important functions:

- `contrastive_label_double_cluster()`: Generates contrastive labels for a pair of clusters.
- `label_single_cluster()`: Generates labels for a single cluster using a language model.
- `validate_cluster_label_discrimination_power()`: Validates cluster labels using AUC scores.
- `validate_cluster_label_comparative_discrimination_power()`: Validates contrastive labels for cluster pairs.
- `match_clusterings()`: Matches clusters between two sets of clusterings.
- `assistant_generative_compare()`: Uses an LLM to generate texts based on cluster differences and evaluates them.

The script supports various command-line arguments for customizing the clustering and analysis process, including:

- `--path`: Path to the input file containing the starting text data.
- `--compare_to_path`: Path to the comparison dataset for comparative analysis.
- `--n_clusters`: Number of clusters to generate.
- `--cluster_method`: Clustering method to use (e.g., "kmeans", "hdbscan").
- `--color_by_divergence`: Flag to enable color-coding of divergences.
- `--original_tokenizer`: Tokenizer to use for the original model.
- `--local_embedding_model_str`: Model to use for text embedding.
- `--local_labelings_model`: Model to use for generating cluster labels.
- `--starting_model_str`: HuggingFace model path for the starting model.
- `--comparison_model_str`: HuggingFace model path for the comparison model.
- `--sampled_texts_per_cluster`: Number of texts to sample for label generation.
- `--sampled_comparison_texts_per_cluster`: Number of texts to sample for label validation.

#### AUC Scoring and P-value Computation

The script uses Area Under the Curve (AUC) scoring and p-value computations to evaluate the effectiveness of generated labels and their statistical significance:

- AUC Scoring: The AUC score represents how effectively a label-derived score acts as a predictor for some ground truth latent variable for the texts in question. For example, when we give the assistant model a label and ask it whether a given text is consistent with this label, we measure how well the assistant model's answer logits predict whether or not the text actually belongs to the cluster that label is supposed to describe. AUC scores range from 0 to 1, with 0.5 representing random chance and 1 representing perfect prediction.

- P-value Computation: The p-value is derived by comparing the observed AUC scores to a null distribution over AUCs. This null distribution is computed by re-running the AUC computation process, but permuting the relationship between texts and labels. The p-value represents the probability of obtaining an AUC score as extreme as the observed one under the null hypothesis (i.e., that there is no real relationship between the labels and the texts). Each p-value is computed independently for each label, without any multiple comparison corrections for the fact that we're evaluating many different labels.

These metrics are used in various parts of the analysis:

1. Validating cluster labels: AUC scores and p-values are computed to assess how well the generated labels describe their respective clusters.
2. Validating contrastive labels: Similar metrics are used to evaluate how well the contrastive labels distinguish between texts from the 'starting' and 'comparison' sides of cluster pairs.
3. Evaluating generated texts: AUC scores and p-values are computed to assess how well the contrastive labels allow the assistant model to generate texts that are more similar to either the starting or comparison model.

The `validate_cluster_label_discrimination_power()`, `validate_cluster_label_comparative_discrimination_power()`, and `validated_assistant_generative_compare()` functions are key components in computing these metrics.

Important parameters related to AUC scoring and p-value computation:

- `--skip_p_value_computation`: Flag to skip p-value computation if only AUC scores are needed.
- `--use_normal_distribution_for_p_values`: Flag to use a normal distribution approximation for p-value computation instead of empirical permutation testing.
- `--permutations_for_null`: Number of permutations to use when computing the null distribution for p-values.


### quick_cluster_commands.sh

This shell script contains example commands for running `quick_cluster.py` with various configurations. It serves as a reference for users to understand how to execute clustering and analysis tasks with different parameters. Key features include:

- Multiple Configuration Examples: Demonstrates various ways to run `quick_cluster.py` with different input files, clustering parameters, and analysis options.
- GPU Assignment: Shows how to assign specific GPUs to different clustering tasks using `CUDA_VISIBLE_DEVICES`.
- Output Redirection: Illustrates how to save the output of clustering runs to log files.
- Parallel Execution: Some commands are set up to run in parallel using background processes.

Example usage pattern:

CUDA_VISIBLE_DEVICES=0 python quick_cluster.py --path llama3-instruct-ai_prefix_fix_interp_KL_log_1.txt --compare_to_path llama3-ai_prefix_fix_interp_KL_log_1.txt --n_strs_show 5 --n_clusters 150 --n_clusters_compare 150 --color_by_divergence --original_tokenizer NousResearch/Meta-Llama-3-8B --skip_every_n_decodings 0 --skip_every_n_decodings_compare 0 --recompute_embeddings --sampled_comparison_texts_per_cluster 10 --non_cluster_comparison_texts 10 --generated_labels_per_cluster 1 --clustering_instructions "Summarize the following texts" &> clustering_results/llama_instruct__COMPARE__base-ai_prefix_fix_interp_NV-Embed-v1_summarize_instruct_KL_log_1.txt