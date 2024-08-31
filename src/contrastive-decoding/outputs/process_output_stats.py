import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from itertools import combinations
import scipy.stats
from terminaltables import AsciiTable


parser = argparse.ArgumentParser()
parser.add_argument('--stats_file', type=str, default='new_stats_summary.pkl')
parser.add_argument('--tokenizer_str', type=str, default="mistralai/Mistral-7B-v0.1") # Alternatively, "NousResearch/Meta-Llama-3-8B-Instruct"
parser.add_argument('--run_substrs', type=str, default="mistral,openhermes") # Alternatively, "llama3"
parser.add_argument('--an_toks', action='store_true')
parser.add_argument('--plot_markers_list', type=str, default="o")
parser.add_argument('--skip_frac', type=float, default=0.0)
parser.add_argument('--max_points', type=int, default=5000)
parser.add_argument('--runs_groups', type=str, default="0")

args = parser.parse_args()

# Has structure:
# {
#     'run_1_name': {
#         'input_tokens': List[List[str]],                 # shape: (n_generations, num_tokens)
#         'forward_divergences': List[float],              # shape: (n_generations,)
#         'starting_perplexities': List[float],            # shape: (n_generations,)
#         'comparison_perplexities': List[float],          # shape: (n_generations,)
#         'forward_token_divergences': List[List[float]],  # shape: (n_generations, num_tokens)
#         'backward_divergences': List[float],             # shape: (n_generations,)
#         'backward_token_divergences': List[List[float]], # shape: (n_generations, num_tokens)
#         'embeddings': List[List[float]]                  # shape: (n_generations, embedding_dim)
#     }
#     'run_2_name': {
#         ...
#     }
#     ...
# }
stats_dict = pickle.load(open(args.stats_file, 'rb'))

runs_names_substrs = args.run_substrs.split(',')
run_keys = [k for k in stats_dict.keys() if any(substr in k for substr in runs_names_substrs) and not "no_top_p_limit" in k]
run_names = [k.replace("-no_prefix_KL_log", "") for k in run_keys]

print(f"stats_dict.keys(): {stats_dict.keys()}")
print(f"runs_names_substrs: {runs_names_substrs}")
print(f"Run keys: {run_keys}")
print(f"Run names: {run_names}")

if args.skip_frac > 0.0:
    for run_key in run_keys:
        run_dict_keys = list(stats_dict[run_key].keys())
        run_skip_records_indices = np.random.choice(range(len(stats_dict[run_key][run_dict_keys[0]])), int(len(stats_dict[run_key][run_dict_keys[0]]) * args.skip_frac), replace=False)
        for dict_key in run_dict_keys:
            stats_dict[run_key][dict_key] = [x for i, x in enumerate(stats_dict[run_key][dict_key]) if i not in run_skip_records_indices]

plot_markers = args.plot_markers_list.split(',')
if len(plot_markers) > 1:
    assert len(plot_markers) == len(run_keys)
else:
    plot_markers = plot_markers * len(run_keys)
print(f"Plot markers: {plot_markers}")

runs_groups = [int(i) for i in args.runs_groups.split(',')]
if len(runs_groups) > 1:
    assert len(runs_groups) == len(run_keys)
else:
    runs_groups = runs_groups * len(run_keys)

# Get a colormap to assign different colors for each run
#colors = plt.cm.get_cmap('tab10', len(stats_dict))
colors = ['blue', 'green', 'red', 'black', 'purple', 'orange', 'brown', 'pink', 'gray']

if args.an_toks:
    # Initialize dictionaries to store token divergences
    token_forward_divergences = defaultdict(list)
    token_backward_divergences = defaultdict(list)

    # Collect all token divergences from each run
    for run_key in run_keys:
        run_data = stats_dict[run_key]
        for generation_idx in range(len(run_data['forward_divergences'])):
            tokens = run_data['input_tokens'][generation_idx]
            forward_divs = run_data['forward_token_divergences'][generation_idx]
            backward_divs = run_data['backward_token_divergences'][generation_idx]
            for token, fwd_div, bwd_div in zip(tokens, forward_divs, backward_divs):
                token_forward_divergences[token].append(fwd_div)
                token_backward_divergences[token].append(bwd_div)

    # Function to calculate average divergences and filter tokens
    def calculate_average_divergences(token_divergences, min_occurrences=3):
        average_divergences = {}
        for token, divergences in token_divergences.items():
            if len(divergences) >= min_occurrences:
                average_divergences[token] = np.mean(divergences)
        return average_divergences

    # Calculate average divergences for forward and backward
    avg_forward_divergences = calculate_average_divergences(token_forward_divergences)
    avg_backward_divergences = calculate_average_divergences(token_backward_divergences)

    # Function to sort tokens by divergence and print top and bottom tokens
    def print_sorted_tokens(divergences, metric_name):
        sorted_tokens = sorted(divergences.items(), key=lambda x: x[1])
        print(f"Top 10 tokens by average {metric_name}:")
        for token, value in sorted_tokens[-10:]:
            print(f"{token}: {value:.5f}")
        print(f"Bottom 10 tokens by average {metric_name}:")
        for token, value in sorted_tokens[:10]:
            print(f"{token}: {value:.5f}")

    # Print sorted tokens for both forward and backward divergences
    print_sorted_tokens(avg_forward_divergences, 'forward_token_divergences')
    print_sorted_tokens(avg_backward_divergences, 'backward_token_divergences')


    # Function to plot histograms of average token divergences
    def plot_histograms_of_divergences(divergences, title):
        plt.figure(figsize=(10, 6))
        plt.hist(divergences.values(), bins=30, alpha=0.75, color='blue')
        plt.title(f'Histogram of Average {title}')
        plt.xlabel('Divergence Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'plots/{title}_histogram.pdf', bbox_inches='tight')
        plt.close()

    # Plot histograms for forward and backward token divergences
    plot_histograms_of_divergences(avg_forward_divergences, 'Forward Token Divergences')
    plot_histograms_of_divergences(avg_backward_divergences, 'Backward Token Divergences')



# For each metric forward_divergences, backward_divergences, starting_perplexities, comparison_perplexities, plot the distribution of the metric in question for each run, as well as for all runs combined.

metrics = ['forward_divergences', 'backward_divergences', 'starting_perplexities', 'comparison_perplexities']
all_metrics_data = {metric: [] for metric in metrics}

# Collect data for each metric from specified runs in run_keys
for run_key in run_keys:
    file_data = stats_dict[run_key]
    for metric in metrics:
        all_metrics_data[metric].extend(file_data[metric])

def filter_outliers_unpaired(metric_data, sd_threshold=None, frac_contains=None):
    #print(f"Filtering outliers from {metric_data}.")
    if sd_threshold is None and frac_contains is None:
        raise ValueError("Either sd_threshold or frac_contains must be provided.")
    mask = np.ones(len(metric_data), dtype=bool)
    if sd_threshold is not None:
        # Filter outliers from metric_data using the mean and standard deviation of the data
        mean, std = np.mean(metric_data), np.std(metric_data)
        #print(f"Mean: {mean}, std: {std}, allowed range: {mean - sd_threshold * std} to {mean + sd_threshold * std}")
        mask = (metric_data > mean - sd_threshold * std) & (metric_data < mean + sd_threshold * std)
    if frac_contains is not None:
        # Filter outliers to keep the frac_contains fraction of the data closest to the median
        num_to_keep = int(frac_contains * len(metric_data))
        median = np.median(metric_data)
        absdiff = np.abs(metric_data - median)
        keep_indices = np.argsort(absdiff)[:num_to_keep]
        mask_frac_contains = np.zeros(len(metric_data), dtype=bool)
        mask_frac_contains[keep_indices] = True
        mask = mask & mask_frac_contains
    return metric_data[mask]

def subsample_data(data_sets_tuple, max_points=5000):
    if type(data_sets_tuple) is list:
        data_sets_tuple = tuple(data_sets_tuple)
    if type(data_sets_tuple) is not tuple:
        data_sets_tuple = (data_sets_tuple,)
    if len(data_sets_tuple[0]) > max_points:
        frac_to_keep = max_points / len(data_sets_tuple[0])
        indices_to_keep = np.random.choice(len(data_sets_tuple[0]), int(frac_to_keep * len(data_sets_tuple[0])), replace=False)
        keep_mask = np.zeros(len(data_sets_tuple[0]), dtype=bool)
        keep_mask[indices_to_keep] = True
        return tuple(data_set[keep_mask] for data_set in data_sets_tuple)
    else:
        return data_sets_tuple

# Plot distribution for each metric for each run
for metric in metrics:
    print(f"Plotting distributions of {metric} for each run.")
    plt.figure(figsize=(10, 6))
    for run_key, run_name, run_index in zip(run_keys, run_names, range(len(run_keys))):
        file_data = stats_dict[run_key]
        filtered_metric_data = filter_outliers_unpaired(np.array(file_data[metric]), sd_threshold=4, frac_contains=0.99)
        #print(f"Maximum value in filtered_metric_data: {np.max(filtered_metric_data)}")

        filtered_metric_data = subsample_data(filtered_metric_data, max_points=2 * args.max_points)
        sns.histplot(filtered_metric_data, kde=True, label=f'Run: {run_name}', color=colors[run_index])
    plt.title(f'Distribution of {metric} for each run')
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{metric}_runs_distributions.pdf', bbox_inches='tight')
    plt.close()

# Plot combined distribution for each metric
for metric in metrics:
    print(f"Plotting combined distribution of {metric} for all runs.")
    plt.figure(figsize=(10, 6))
    filtered_all_metrics_data = filter_outliers_unpaired(np.array(all_metrics_data[metric]), sd_threshold=4, frac_contains=0.99)
    filtered_all_metrics_data = subsample_data(filtered_all_metrics_data, max_points=2 * args.max_points)
    #sns.distplot(all_metrics_data[metric], color='red', label=f'All Runs Combined', hist=False, kde=True)
    sns.kdeplot(filtered_all_metrics_data, color='red', label=f'All Runs Combined')
    plt.title(f'Combined Distribution of {metric}')
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{metric}_combined_distribution.pdf', bbox_inches='tight')
    plt.close()


# For each pair of metrics forward_divergences, backward_divergences, starting_perplexities, comparison_perplexities, plot the correlation between the two metrics in question for each run, as well as for all runs combined (the latter by itself in a second plot).

# Define pairs of metrics to analyze
metric_pairs = list(combinations(metrics, 2))

# Function to calculate linear regression and annotate plot
def plot_linear_fit(x, y, ax, color='orange'):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    line = slope * np.array(x) + intercept
    ax.plot(x, line, color=color, label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}')

# Define a function to filter outliers more than 10 SDs from the mean for paired data
def filter_outliers_paired(metric1_data, metric2_data, sd_threshold=None, frac_contains=None):
    if sd_threshold is None and frac_contains is None:
        raise ValueError("Either sd_threshold or frac_contains must be provided.")
    combined_mask = np.ones(len(metric1_data), dtype=bool)
    if sd_threshold is not None:
        mean1, std1 = np.mean(metric1_data), np.std(metric1_data)
        mean2, std2 = np.mean(metric2_data), np.std(metric2_data)
    
        # Create masks for values within 10 SDs of the mean for both metrics
        mask1 = (metric1_data > mean1 - sd_threshold * std1) & (metric1_data < mean1 + sd_threshold * std1)
        mask2 = (metric2_data > mean2 - sd_threshold * std2) & (metric2_data < mean2 + sd_threshold * std2)
        
        # Combine masks to ensure points are valid in both metrics
        combined_mask = mask1 & mask2
    if frac_contains is not None:
        # Filter outliers to keep the frac_contains fraction of the data closest to the median
        num_to_keep = int(frac_contains * len(metric1_data))
        median1 = np.median(metric1_data)
        median2 = np.median(metric2_data)
        absdiff1 = np.abs(metric1_data - median1)
        absdiff2 = np.abs(metric2_data - median2)
        keep_indices1 = np.argsort(absdiff1)[:num_to_keep]
        keep_indices2 = np.argsort(absdiff2)[:num_to_keep]
        mask1 = np.zeros(len(metric1_data), dtype=bool)
        mask2 = np.zeros(len(metric2_data), dtype=bool)
        mask1[keep_indices1] = True
        mask2[keep_indices2] = True
        combined_mask = mask1 & mask2 & combined_mask
    # Filter both datasets with the combined mask
    return metric1_data[combined_mask], metric2_data[combined_mask]

# Plot correlation for each pair of metrics for each run, excluding outliers more than 10 SDs from the mean
for metric1, metric2 in metric_pairs:
    print(f"Plotting correlation between {metric1} and {metric2} for each run.")
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for run_i, run_key in enumerate(run_keys):
        file_data = stats_dict[run_key]
        metric1_data = np.array(file_data[metric1])
        metric2_data = np.array(file_data[metric2])
        
        # Apply the outlier filter to both metrics simultaneously
        filtered_metric1_data, filtered_metric2_data = filter_outliers_paired(metric1_data, metric2_data, sd_threshold=4, frac_contains=0.99)
        
        # Plot data without outliers
        filtered_metric1_data, filtered_metric2_data = subsample_data((filtered_metric1_data, filtered_metric2_data), max_points=2 * args.max_points)
        plt.scatter(filtered_metric1_data, filtered_metric2_data, label=f'Run: {run_names[run_i]}', color=colors[run_i])
        plot_linear_fit(filtered_metric1_data, filtered_metric2_data, ax, color=colors[run_i])
        
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.title(f'Correlation between {metric1} and {metric2} for each run')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{metric1}_vs_{metric2}_correlation_per_run.pdf', bbox_inches='tight')
    plt.close()

# Plot combined correlation for each pair of metrics
for metric1, metric2 in metric_pairs:
    print(f"Plotting combined correlation between {metric1} and {metric2} for all runs.")
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    combined_metric1 = []
    combined_metric2 = []
    for run_key in run_keys:
        file_data = stats_dict[run_key]
        combined_metric1.extend(file_data[metric1])
        combined_metric2.extend(file_data[metric2])
    
    # Apply the outlier filter to both metrics simultaneously
    combined_metric1, combined_metric2 = filter_outliers_paired(np.array(combined_metric1), np.array(combined_metric2), sd_threshold=4, frac_contains=0.99)
    combined_metric1, combined_metric2 = subsample_data((combined_metric1, combined_metric2), max_points=2 * args.max_points)
    
    plt.scatter(combined_metric1, combined_metric2, color='red', label='All Runs Combined')
    plot_linear_fit(combined_metric1, combined_metric2, ax)
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.title(f'Combined Correlation between {metric1} and {metric2}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/{metric1}_vs_{metric2}_combined_correlation.pdf', bbox_inches='tight')
    plt.close()




# For each of forward_token_divergences and backward_token_divergences, plot the association between the token index and the metric in question for each run
metrics = ['forward_token_divergences', 'backward_token_divergences']

for metric in metrics:
    print(f"Plotting {metric} vs token index for each run grouped by run_group.")
    unique_groups = set(runs_groups)
    num_groups = len(unique_groups)
    fig, axs = plt.subplots(1, num_groups, figsize=(10 * num_groups, 6), sharey=True)

    for group_id in unique_groups:
        group_indices = [i for i, x in enumerate(runs_groups) if x == group_id]
        for idx in group_indices:
            run_key = run_keys[idx]
            run_name = run_names[idx]
            plot_marker = plot_markers[idx]
            run_data = stats_dict[run_key]
            all_token_divergences = []
            all_token_indices = []
            for generation_idx in range(len(run_data[metric])):
                token_indices = list(range(len(run_data['forward_token_divergences'][generation_idx])))
                token_divergences = run_data[metric][generation_idx]
                all_token_divergences.extend(token_divergences)
                all_token_indices.extend(token_indices)
            all_token_divergences, all_token_indices = filter_outliers_paired(np.array(all_token_divergences), np.array(all_token_indices), sd_threshold=4, frac_contains=0.99)

            # Potentially subsample the data
            all_token_divergences, all_token_indices = subsample_data((all_token_divergences, all_token_indices), max_points=args.max_points)

            axs[group_id].scatter(all_token_indices, all_token_divergences, label=f"{run_name}", marker=plot_marker)
            
            # Calculate and plot the best fit line for all points
            slope, intercept, r_value, _, _ = scipy.stats.linregress(all_token_indices, all_token_divergences)
            line = slope * np.array(all_token_indices) + intercept
            axs[group_id].plot(all_token_indices, line, label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}', color='black')

        axs[group_id].set_title(f'Group {group_id}: Token Index vs {metric}')
        axs[group_id].set_xlabel('Token Index')
        axs[group_id].set_ylabel(metric)
        axs[group_id].legend()
        axs[group_id].grid(True)

    plt.savefig(f'plots/{metric}_vs_token_index_grouped.pdf', bbox_inches='tight')
    plt.close()

# For each metric, generate a different plot for each token location, from 0 to 40, grouped by run_group.
for metric in metrics:
    print(f"Plotting distribution of {metric} values at each token index for each run, grouped by run_group.")
    unique_groups = set(runs_groups)
    for token_index in range(0, 40, 1):  # Assuming token indices range from 0 to 40
        fig, axs = plt.subplots(1, len(unique_groups), figsize=(10 * len(unique_groups), 6), sharey=True)
        for group_id in unique_groups:
            group_indices = [i for i, x in enumerate(runs_groups) if x == group_id]
            for run_i in group_indices:
                run_key = run_keys[run_i]
                run_data = stats_dict[run_key]
                
                metric_values_at_index = [generation[token_index] for generation in run_data[metric] if len(generation) > token_index]
                metric_values_at_index = filter_outliers_unpaired(np.array(metric_values_at_index), sd_threshold=4, frac_contains=0.99)
                sns.histplot(metric_values_at_index, kde=False, label=f"{run_names[run_i]} Distribution at Token {token_index}", color=colors[run_i], alpha=0.2, bins=50, ax=axs[group_id])
            
            axs[group_id].set_title(f'Group {group_id}: Distribution of {metric} Values at Token Index {token_index}')
            axs[group_id].set_xlabel(f'{metric} Value')
            axs[group_id].set_ylabel('Frequency')
            axs[group_id].set_yscale('log')
            axs[group_id].set_ylim(1, 2000)
            axs[group_id].legend()
            axs[group_id].grid(True)

        plt.savefig(f'plots/{metric}_value_distribution_at_token_{token_index}_grouped.pdf', bbox_inches='tight')
        plt.close()

# Generate divergence histograms for all tokens across all positions, broken up by group IDs, with each run shown independently
print("Generating divergence histograms for all tokens across all positions, broken up by group IDs, with each run shown independently.")

# Assuming 'stats_dict' contains all the necessary run data and 'runs_groups' maps run_keys to group_ids
unique_groups = set(runs_groups)
for metric in metrics:
    # Create a figure for each metric
    fig, axs = plt.subplots(1, len(unique_groups), figsize=(12 * len(unique_groups), 8), sharey=True)
    
    for group_id in unique_groups:
        group_indices = [i for i, x in enumerate(runs_groups) if x == group_id]
        
        # Plot histogram for each run in the group independently
        for run_i in group_indices:
            run_key = run_keys[run_i]
            run_data = stats_dict[run_key]
            all_divergences = []

            # Collect all divergence data for the metric across all tokens in the run
            for generation in run_data[metric]:
                all_divergences.extend(generation)

            # Filter outliers from the collected divergences
            all_divergences = np.array(all_divergences)
            filtered_divergences = filter_outliers_unpaired(all_divergences, sd_threshold=4, frac_contains=0.99)

            # Plot histogram for the run
            axs[group_id].hist(filtered_divergences, bins=100, alpha=0.7, label=f"Run {run_keys[run_i]}")

        axs[group_id].set_title(f'Group {group_id}: Histogram of {metric} Divergences')
        axs[group_id].set_xlabel('Divergence')
        axs[group_id].set_ylabel('Frequency')
        axs[group_id].legend()
        axs[group_id].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/{metric}_divergence_histogram_by_group_and_run.pdf')
    plt.close()
    print(f"Histogram for {metric} by group and run saved.")




# For each metric, plot the correlation between the metric's value at token i and its value at token i+j for each run, j in range(1, 5)
for j in range(1, 5):
    print(f"Plotting correlation of {metric} at Token i and i+{j} for each run.")
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for run_i, run_key, run_name, plot_marker in zip(range(len(run_keys)), run_keys, run_names, plot_markers):
            run_data = stats_dict[run_key]
            token_divergences = []
            next_token_divergences = []
            for generation_idx in range(len(run_data[metric])):
                if len(run_data[metric][generation_idx]) > j:  # Ensure there is a next token
                    current_divergences = run_data[metric][generation_idx]
                    token_divergences.extend(current_divergences[:-j])  # Current token i
                    next_token_divergences.extend(current_divergences[j:])  # Next token i+j

            # Plot the correlation between current token divergence and next token divergence
            token_divergences, next_token_divergences = filter_outliers_paired(np.array(token_divergences), np.array(next_token_divergences), sd_threshold=4, frac_contains=0.99)
            token_divergences, next_token_divergences = subsample_data((token_divergences, next_token_divergences), max_points=args.max_points)

            plt.scatter(token_divergences, next_token_divergences, label=f"{run_name}", marker=plot_marker, color=colors[run_i])
            # Calculate and plot the best fit line for the points
            slope, intercept, r_value, _, _ = scipy.stats.linregress(token_divergences, next_token_divergences)
            line = slope * np.array(token_divergences) + intercept
            plt.plot(token_divergences, line, label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r_value**2:.2f}', color=colors[run_i])

        plt.title(f'Correlation of {metric} at Token i and i+{j}')
        plt.xlabel(f'{metric} at Token i')
        plt.ylabel(f'{metric} at Token i+{j}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/{metric}_correlation_token_i_i+{j}.pdf', bbox_inches='tight')
        plt.close()



# Use TSNE to reduce the dimensionality of the embeddings and plot the results, using different colors for each run


# Assuming `stats_dict` is a dictionary where keys are run identifiers and each run contains 'embeddings' key with a list of embeddings
# Example structure: stats_dict = {'run1': {'embeddings': [[0.1, 0.2, ...], [0.2, 0.3, ...]]}, 'run2': {'embeddings': [[0.3, 0.4, ...], [0.4, 0.5, ...]]}}

print("Plotting TSNE of all embeddings by run.")

# Collect all embeddings and labels for coloring
all_embeddings = []
labels = []

for run_key in run_keys:
    run_data = stats_dict[run_key]
    emb_list = run_data['embeddings']
    all_embeddings.extend(emb_list)
    labels.extend([run_key] * len(emb_list))  # Extend labels list with the run_key for each embedding

# Convert list of embeddings to numpy array
all_embeddings_array = np.array(all_embeddings)
perplexity = 30
# Initialize TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, early_exaggeration=12)  # Using 2 components for 2D visualization
transformed_embeddings = tsne.fit_transform(all_embeddings_array)  # Apply TSNE transformation to all embeddings

plt.figure(figsize=(25, 18))

# Plot each run's embeddings in the transformed space
for idx, (run_key, run_name, plot_marker) in enumerate(zip(run_keys, run_names, plot_markers)):
    # Select indices for current run's embeddings
    indices = [i for i, label in enumerate(labels) if label == run_key]
    plt.scatter(transformed_embeddings[indices, 0], transformed_embeddings[indices, 1], color=colors[idx], label=f'Run: {run_name}', s=1, marker=".")

plt.title('TSNE of All Embeddings by Run')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.legend(markerscale=5)  # Increase the size of the markers in the legend
plt.grid(True)
plt.savefig(f'plots/tsne_all_embeddings_by_run_perp_{perplexity}.pdf', bbox_inches='tight')
plt.close()

# Compute the average pairwise divergence between the points in each run, then output a n_runs x n_runs matrix of the average divergences between each run's points.

# Initialize matrix to store average divergences
n_runs = len(run_keys)
average_divergences_matrix = np.zeros((n_runs, n_runs))

# Compute pairwise divergences for each pair of runs
for i, run_key_i in enumerate(run_keys):
    indices_i = [idx for idx, label in enumerate(labels) if label == run_key_i]
    embeddings_i = transformed_embeddings[indices_i]
    
    for j, run_key_j in enumerate(run_keys):
        if i <= j:  # Compute divergence only once for each pair (matrix is symmetric)
            indices_j = [idx for idx, label in enumerate(labels) if label == run_key_j]
            embeddings_j = transformed_embeddings[indices_j]
            
            # Calculate pairwise distances between all points in run i and run j, using the L1/2 norm
            pairwise_distances = scipy.spatial.distance.cdist(embeddings_i, embeddings_j, 'minkowski', p=0.5)
            
            # Compute average of these distances
            average_distance = round(np.mean(pairwise_distances), 2)
            average_divergences_matrix[i, j] = average_distance
            average_divergences_matrix[j, i] = average_distance  # Symmetric assignment

# Output the matrix of average divergences using terminaltables
table_data = [['Run'] + [k for k in run_names]]
for i, run_name in enumerate(run_names):
    row = [run_name] + list(average_divergences_matrix[i])
    table_data.append(row)

table = AsciiTable(table_data)
print("Average pairwise divergences between runs:")
print(table.table)




