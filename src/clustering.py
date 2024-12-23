from sklearn.neighbors import NearestCentroid
from typing import Callable, Literal, Any, Union, Iterable
from gap_statistic import OptimalK
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

from visualization.dendogram import *

import warnings

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False
    warnings.warn("matplotlib not installed; results plotting is disabled.")
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None
    warnings.warn(
        "joblib not installed, will be unavailable as a backend for parallel processing."
    )


class AgglomerativeClusteringWrapper(AgglomerativeClustering):
    def __init__(self, n_init: None | int = 1, n_clusters: None | int = 2, *, affinity: str | Callable[..., Any] = "deprecated", metric: None | str | Callable[..., Any] = 'euclidean', memory: None | Any | str = None, connectivity: None | Any | Callable[..., Any] = None, compute_full_tree: bool | Literal['auto'] = "auto", linkage: Literal['ward'] | Literal['complete'] | Literal['average'] | Literal['single'] = "ward", distance_threshold: None | float | Any = None, compute_distances: bool = False) -> None:
        self.affinity = 'precomputed'#affinity
        self.n_init = n_init
        super().__init__(n_clusters, metric=metric, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, distance_threshold=distance_threshold, compute_distances=compute_distances)
    
    def predict(self, X):
        return self.fit_predict(X, None)
    

class OptimalK_Wrapper(OptimalK):
    def __init__(self,
                 clusterer: Callable = AgglomerativeClustering,
                 clusterer_kwargs: dict = {
                     'linkage': 'ward'
                },
                parallel_backend = 'joblib'
        ):
        super().__init__(
            clusterer=clusterer,
            clusterer_kwargs=clusterer_kwargs,
            parallel_backend=parallel_backend
        )

    def find_optimal_K(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_refs: int = 3,
        cluster_array: Iterable[int] = ()
        ):

        """
        Calculates optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        :param X - pandas dataframe or numpy array of data points of shape (n_samples, n_features)
        :param n_refs - int: Number of random reference data sets used as inertia reference to actual data.
        :param cluster_array - 1d iterable of integers; each representing n_clusters to try on the data.
        """

        self.X = X

        # Convert the 1d array of n_clusters to try into an array
        # Raise error if values are less than 1 or larger than the unique sample in the set.
        cluster_array = np.array([x for x in cluster_array]).astype(int)
        if np.where(cluster_array < 1)[0].shape[0]:
            raise ValueError(
                "cluster_array contains values less than 1: {}".format(
                    cluster_array[np.where(cluster_array < 1)[0]]
                )
            )
        if cluster_array.shape[0] > X.shape[0]:
            raise ValueError(
                "The number of suggested clusters to try ({}) is larger than samples in dataset. ({})".format(
                    cluster_array.shape[0], X.shape[0]
                )
            )
        if not cluster_array.shape[0]:
            raise ValueError("The supplied cluster_array has no values.")

        # Array of resulting gaps.
        gap_df = pd.DataFrame({"n_clusters": [], "gap_value": []})

        # Define the compute engine; all methods take identical args and are generators.
        if self.parallel_backend == "joblib":
            engine = self._process_with_joblib
        elif self.parallel_backend == "multiprocessing":
            engine = self._process_with_multiprocessing
        elif self.parallel_backend == "rust":
            engine = self._process_with_rust
        else:
            engine = self._process_non_parallel

        # Calculate the gaps for each cluster count.
        my_data = []
        for gap_calc_result in engine(X, n_refs, cluster_array):

            my_data.append({
                    "n_clusters": gap_calc_result.n_clusters,
                    "gap_value": gap_calc_result.gap_value,
                    "ref_dispersion_std": gap_calc_result.ref_dispersion_std,
                    "sdk": gap_calc_result.sdk,
                    "sk": gap_calc_result.sk,
                    "gap*": gap_calc_result.gap_star,
                    "sk*": gap_calc_result.sk_star,
            })

            # Assign this loop's gap statistic to gaps
            gap_df = pd.DataFrame(my_data)
            
            gap_df["gap_k+1"] = gap_df["gap_value"].shift(-1)
            gap_df["gap*_k+1"] = gap_df["gap*"].shift(-1)
            gap_df["sk+1"] = gap_df["sk"].shift(-1)
            gap_df["sk*+1"] = gap_df["sk*"].shift(-1)
            gap_df["diff"] = gap_df["gap_value"] - gap_df["gap_k+1"] + gap_df["sk+1"]
            gap_df["diff*"] = gap_df["gap*"] - gap_df["gap*_k+1"] + gap_df["sk*+1"]

        # drop auxilariy columns
        gap_df.drop(
            labels=["sdk", "gap_k+1", "gap*_k+1", "sk+1", "sk*+1"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        self.gap_df = gap_df.sort_values(by="n_clusters", ascending=True).reset_index(
            drop=True
        )
        self.n_clusters = int(
            self.gap_df.loc[np.argmax(self.gap_df.gap_value.values)].n_clusters
        )
    
        return self.n_clusters
    
    def plot_gaps(self, model, knee, elbow, size = (30, 10)):
        """
        Plots the results of the last run optimal K search procedure.
        Four plots are printed:
        (1) A plot of the Gap value - as defined in the original Tibshirani et
        al paper - versus n, the number of clusters.
        (2) A plot of diff versus n, the number of clusters, where diff =
        Gap(k) - Gap(k+1) + s_{k+1}. The original Tibshirani et al paper
        recommends choosing the smallest k such that this measure is positive.
        (3) A plot of the Gap* value - a variant of the Gap statistic suggested
        in "A comparison of Gap statistic definitions with and with-out
        logarithm function" [https://core.ac.uk/download/pdf/12172514.pdf],
        which simply removes the logarithm operation from the Gap calculation -
        versus n, the number of clusters.
        (4) A plot of the diff* value versus n, the number of clusters. diff*
        corresponds to the aforementioned diff value for the case of Gap*.
        """
        if not MATPLOTLIB_FOUND:
            print("matplotlib not installed; results plotting is disabled.")
            return
        if not hasattr(self, "gap_df") or self.gap_df is None:
            print("No results to print. OptimalK not called yet.")
            return
        
        fig = plt.figure(figsize=size)

        # Dendrogram plot
        plt.subplot(1, 2, 1)    
        model = model(n_clusters=self.n_clusters, linkage='ward', compute_distances=True)
        model.fit(self.X)

        #cm = plt.get_cmap("gist_rainbow")
        #color_dict = {i: cm(1.0 * i / self.n_clusters) for i in range(self.n_clusters)}
        link_color_func = lambda k: f'C{int(k)}'
        plot_dendrogram(model, dendogram_function=fancy_dendrogram, truncate_mode = 'lastp', p = self.n_clusters, link_color_func = link_color_func)

        # Gap values plot
        plt.subplot(1, 2, 2)
        plt.plot(self.gap_df.n_clusters, self.gap_df.gap_value, linewidth=2, label = 'data')
        plt.scatter(
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].n_clusters,
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].gap_value,
            s=95,
            c="r",
            label = 'optimal-K'
        )
        plt.vlines(
            knee, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="knee",
            linewidth = 3, color = 'orange'
        )
        plt.vlines(
            elbow, plt.ylim()[0], plt.ylim()[1], linestyles="--", label="elbow",
            linewidth = 3, color = 'teal'
        )
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Gap Value")
        plt.legend(loc="best")
        plt.title("Gap Values by Cluster Count")

        


        # diff plot
        if False:
            plt.plot(self.gap_df.n_clusters, self.gap_df["diff"], linewidth=3)
            plt.grid(True)
            plt.xlabel("Cluster Count")
            plt.ylabel("Diff Value")
            plt.title("Diff Values by Cluster Count")
            plt.show()

        # diff plot
        if False:
            # Gap* plot
            plt.subplot(1, 3, 3)
            max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
            plt.plot(self.gap_df.n_clusters, self.gap_df["gap*"], linewidth=3)
            plt.scatter(
                self.gap_df.loc[max_ix]["n_clusters"],
                self.gap_df.loc[max_ix]["gap*"],
                s=250,
                c="r",
            )
            plt.grid(True)
            plt.xlabel("Cluster Count")
            plt.ylabel("Gap* Value")
            plt.title("Gap* Values by Cluster Count")

        # diff* plot
        if False:
            plt.plot(self.gap_df.n_clusters, self.gap_df["diff*"], linewidth=3)
            plt.grid(True)
            plt.xlabel("Cluster Count")
            plt.ylabel("Diff* Value")
            plt.title("Diff* Values by Cluster Count")
            plt.show()

        return fig
    

def agglomerative_clustering_function(X, k, **kwargs):
    """ 
    These user defined functions *must* take the X and a k 
    and can take an arbitrary number of other kwargs, which can
    be pass with `clusterer_kwargs` when initializing OptimalK
    """
    
    # Here you can do whatever clustering algorithm you heart desires,
    # but we'll do a simple wrap of the MeanShift model in sklearn.
    
    model = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=False)
    y = model.fit_predict(X)
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(X, y)

    cluster_centers_ = nearest_centroid.centroids_
    
    # Return the location of each cluster center,
    # and the labels for each point.
    return cluster_centers_, y

def make_agglomerative_clustering_function(linkage='ward'):
    def agglomerative_clustering_function(X, k, **kwargs):
        """ 
        These user defined functions *must* take the X and a k 
        and can take an arbitrary number of other kwargs, which can
        be pass with `clusterer_kwargs` when initializing OptimalK
        """
        
        # Here you can do whatever clustering algorithm you heart desires,
        # but we'll do a simple wrap of the MeanShift model in sklearn.
        
        model = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=False)
        y = model.fit_predict(X)
        nearest_centroid = NearestCentroid()
        nearest_centroid.fit(X, y)

        cluster_centers_ = nearest_centroid.centroids_
        
        # Return the location of each cluster center,
        # and the labels for each point.
        return cluster_centers_, y
    return agglomerative_clustering_function

