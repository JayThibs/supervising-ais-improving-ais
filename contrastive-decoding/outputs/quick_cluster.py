
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--api_key_path", default="../../../key.txt")
parser.add_argument("--n_clusters", default=30)
args = parser.parse_args()
path = args.path
api_key_path = args.api_key_path
n_clusters = int(args.n_clusters)


# Read OpenAI auth key from a file
with open(api_key_path, 'r') as file:
    openai_auth_key = file.read().strip()

# Authenticate with the OpenAI API
client = OpenAI(api_key=openai_auth_key)

df = pd.read_csv(path)

# First, load past results into arrays:
# decoded_strs is an array of strings
# divergence_values stores a single float for each entry in decoded_strs
divergence_values = df['divergence'].values
loaded_strs = df['decoding'].values
n_datapoints = len(loaded_strs)
#decoded_strs = [s.split("|")[1] for s in loaded_strs]
decoded_strs = [s.replace("|", "") for s in loaded_strs]
for i in range(len(decoded_strs)):
    if decoded_strs[i][0] == ' ':
        decoded_strs[i] = decoded_strs[i][1:]
#print(decoded_strs)

# Generate embeddings for the past results.
# embeddings_list is a n_datapoints x embedding_dim list of floats
embeddings_list = []
batch_size = 100
for i in tqdm(range(0, len(decoded_strs), batch_size)):
    batch = decoded_strs[i:i+batch_size]
    embeddings = client.embeddings.create(input = batch, model = "text-embedding-ada-002").data
    embeddings_list.extend([e.embedding for e in embeddings])

# Now, cluster the entries of embeddings_list
clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(embeddings_list)
labels = clustering.labels_

# Calculate the average divergence value of all the texts in each cluster.
average_divergence_values = [0] * len(set(labels))
n_texts_in_cluster = [0] * len(set(labels))
for i in range(n_datapoints):
    cluster = labels[i]
    average_divergence_values[cluster] += divergence_values[i]
    n_texts_in_cluster[cluster] += 1
for i in range(len(average_divergence_values)):
    average_divergence_values[i] /= n_texts_in_cluster[i]

# For each cluster, we print out the average divergence as well as the 5 top divergence texts and the 5 bottom divergence texts assigned to that cluster.
for cluster in set(labels):
    print("===================================================================")
    print(f"Cluster {cluster}:")
    print(f"Average divergence: {average_divergence_values[cluster]}")
    
    cluster_indices = [i for i, x in enumerate(labels) if x == cluster]
    cluster_divergences = [divergence_values[i] for i in cluster_indices]
    cluster_texts = [decoded_strs[i] for i in cluster_indices]
    
    sorted_indices = sorted(range(len(cluster_divergences)), key=lambda k: cluster_divergences[k])

    top_5_texts = [cluster_texts[i] for i in sorted_indices[:5]]
    bottom_5_texts = [cluster_texts[i] for i in sorted_indices[-5:]]

    # Use OpenAI chat completions to generate a cluster label based on the top 5 texts
    top_5_texts_label = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at labeling clusters. You produce a single susinct label immediately in response to every query."},
            {"role": "user", "content": "Texts in current cluster: " + ', '.join(top_5_texts)}
        ],
        max_tokens=30)
    bottom_5_texts_label = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at labeling clusters. You produce a single susinct label immediately in response to every query."},
            {"role": "user", "content": "Texts in current cluster: " + ', '.join(bottom_5_texts)}
        ],
        max_tokens=30)
    
    print("Top 5 divergence texts:")
    for i in sorted_indices[:5]:
        print(f"Divergence: {round(cluster_divergences[i], 4)} Text: {cluster_texts[i]}")
    print("Cluster label (top 5):")
    print(top_5_texts_label.choices[0].message.content)
    print("\n")
    
    print("Bottom 5 divergence texts:")
    for i in sorted_indices[-5:]:
        print(f"Divergence: {round(cluster_divergences[i], 4)} Text: {cluster_texts[i]}")
    print("Cluster label (bottom 5):")
    print(bottom_5_texts_label.choices[0].message.content)
    print("\n\n")
