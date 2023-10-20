from sklearn.cluster import AgglomerativeClustering


def hierarchical_clustering(embeddings):
    model = AgglomerativeClustering()
    clusters = model.fit_predict(embeddings)
    return clusters
