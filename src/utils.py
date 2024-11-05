import numpy as np
from scipy.stats import entropy

def minmax(x, axis = 0):
    return (x - x.min(axis=axis)) / (x.max(axis=axis) - x.min(axis=axis))

def calculate_homogeneity(data) -> np.array:
    cluster_array = np.asanyarray(data)
    normalized_cluster_array = cluster_array / np.sum(cluster_array, axis=1,keepdims=True)
    hom = 1 - entropy(normalized_cluster_array, axis=1) / np.log(normalized_cluster_array.shape[1])
    return hom