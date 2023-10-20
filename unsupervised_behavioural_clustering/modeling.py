import openai
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def embed_texts(texts, model="text-embedding-ada-002"):
    ...


def joint_embedding(inputs, responses):
    ...


def cluster_embeddings(embeddings, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans


def reduce_dimensions(embeddings, perplexity=30, n_iter=1000):
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter)
    return tsne.fit_transform(embeddings)
