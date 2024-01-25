# These scripts can be used to run the clustering algorithm on the data.
# As part of the scripts directory, the focus is to make it easy to
# rerun parts of the code, e.g. clustering, visualization, CD, etc.
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, SpectralClustering
from sklearn.cluster.hierarchical import dendrogram
