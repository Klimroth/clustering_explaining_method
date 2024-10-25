import os.path
from itertools import chain, combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.cluster import AgglomerativeClustering
#from clustering import AgglomerativeClusteringWrapper as AgglomerativeClustering
from clustering import OptimalK_Wrapper, agglomerative_clustering_function
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.distance import jensenshannon, correlation, euclidean
from tqdm.contrib.concurrent import thread_map

from visualize_result import ResultVisualizer


import config


"""
draw_gap_statistic_plot(): outputs gap statistic evaluation on the observable 
    to determine the number of observable patterns
calculate_observable_patterns(): using config.NUMBER_OBSERVABLE_PATTERNS, 
    it conducts the clustering on observables and plots a dendrogram
calculate_explainable_distances(): depending on feature selection mode it outputs pairwise 
    distances based on explainable features and plots a dendrogram
"""


class ClusteringApplier:

    @staticmethod
    def _read_observable_data(
        read_only_feature_col: bool = True,
    ) -> pd.DataFrame | None:
        required_file: str = (
            f"{config.OUTPUT_FOLDER_BASE}base_data/{config.DATASET_NAME}_observable_dataset_scaled.xlsx"
        )
        try:
            df: pd.DataFrame = pd.read_excel(required_file)
            if read_only_feature_col:
                df = df[list(config.OBSERVABLE_FEATURE_NAMES.keys())]
        except:
            print(f"Error. Invalid file {required_file}.")
            return None

        return df

    @staticmethod
    def read_explaining_features() -> pd.DataFrame | None:
        try:
            df = pd.read_excel(
                f"{config.OUTPUT_FOLDER_BASE}base_data/{config.DATASET_NAME}_explainable_dataset_scaled.xlsx",
                index_col=0
            )
        except:
            print(f"File not valid {config.INPUT_FILE_EXPLAINING_FEATURES}")
            return None
        return df

    @staticmethod
    def ___draw_gap_statistic_plot() -> None:

        df: pd.DataFrame = ClusteringApplier._read_observable_data()

        gs = GapStatistics(
            algorithm=AgglomerativeClustering,
            distance_metric="minkowski",
            return_params=True,
        )
        num_colors: int = config.GAP_STATISTIC_CLUSTER_RANGE
        cm = plt.get_cmap("gist_rainbow")
        color_dict = {i: cm(1.0 * i / num_colors) for i in range(num_colors)}
        gs.fit_predict(K=config.GAP_STATISTIC_CLUSTER_RANGE, X=df.to_numpy())
        gs.plot(original_labels=df.columns, colors=color_dict)

        ouput_folder: str = f"{config.OUTPUT_FOLDER_BASE}gapstat/"
        if not os.path.exists(ouput_folder):
            os.makedirs(ouput_folder)
        plt.savefig(
            f"{ouput_folder}{config.DATASET_NAME}_gap-statistic-plot.pdf",
            bbox_inches="tight",
        )

    @staticmethod
    def draw_gap_statistic_plot() -> None:

        df: pd.DataFrame = ClusteringApplier._read_observable_data()

        optimal_K = OptimalK_Wrapper(clusterer=agglomerative_clustering_function)

        X = df.to_numpy()
        cluster_range = np.arange(2, config.GAP_STATISTIC_CLUSTER_RANGE)
        n_clusters = optimal_K.find_optimal_K(X, cluster_array=cluster_range)

        fig = optimal_K.plot_gaps(AgglomerativeClustering, size = (30, 7))

        ouput_folder: str = f"{config.OUTPUT_FOLDER_BASE}gapstat/"
        if not os.path.exists(ouput_folder):
            os.makedirs(ouput_folder)
        fig.savefig(
            f"{ouput_folder}{config.DATASET_NAME}_gap-statistic-plot.pdf",
            bbox_inches="tight",
        )

    @staticmethod
    def _plot_dendrogram_by_distance_matrix(
        mat: np.array, labels: List[str], x_label: str, y_label: str, title: str
    ):

        plt.cla()
        plt.clf()

        dists = squareform(mat)
        linkage_matrix = linkage(dists, "single")
        with plt.rc_context({"lines.linewidth": 0.25}):
            dendrogram(
                linkage_matrix,
                labels=np.array(labels),
                color_threshold=0,
                above_threshold_color="k",
                leaf_rotation=90.0,
            )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        output_path = f"{config.OUTPUT_FOLDER_BASE}dendrogram/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(
            f"{output_path}{config.DATASET_NAME}_dendrogram-{x_label}-{y_label}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _plot_dendrogram(
        df: pd.DataFrame,
        x_label: str,
        title: str,
        y_label: str = "Ward's distance measure",
        no_labels: bool = True,
    ) -> None:

        plt.cla()
        plt.clf()
        Z = sch.linkage(df.to_numpy(), method="ward")
        with plt.rc_context({"lines.linewidth": 0.25}):
            sch.dendrogram(
                Z,
                labels=df.index,
                no_labels=no_labels,
                color_threshold=0,
                above_threshold_color="k",
                leaf_rotation=90.0,
            )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        output_path = f"{config.OUTPUT_FOLDER_BASE}dendrogram/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.savefig(
            f"{output_path}{config.DATASET_NAME}_dendrogram-{x_label}-{y_label}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _calculate_fingerprints(df: pd.DataFrame) -> pd.DataFrame:

        group_names: List[str] = list(set(df[config.GROUP_NAME].to_list()))
        num_clusters: int = max(df["pattern_type"].to_list()) + 1
        group_stat: Dict[str, Dict[int, int]] = {
            grp_name: {j: 0 for j in range(1, num_clusters + 1)}
            for grp_name in group_names
        }

        ret = {j: [] for j in range(1, num_clusters + 1)}
        ret[config.GROUP_NAME] = []

        for row in df.index:
            group_stat[df.loc[row, config.GROUP_NAME]][
                int(df.loc[row, "pattern_type"]) + 1
            ] += 1

        for grp_name in group_stat.keys():
            quantities = group_stat[grp_name]
            N: int = np.sum(list(quantities.values()))
            for j in quantities.keys():
                ret[j].append(quantities[j] / N)
            ret[config.GROUP_NAME].append(grp_name)

        return pd.DataFrame(ret).set_index(config.GROUP_NAME)
    
    @staticmethod
    def calculate_pairwise_fingerprint_distances(
            df: pd.DataFrame, distance: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if distance not in [
            "jensenshannon",
            "euclidean",
            "correlation",
        ]:
            raise Exception(
                "Invalid distance measure used to measure similarity."
            )
        
        distance_matrix: np.array = np.zeros((df.shape[0], df.shape[0]))
        arr = df.to_numpy()

        for grp_i in range(df.shape[0]):
            for grp_j in range(df.shape[0]):     
                fingerprint_1 = arr[grp_i]
                fingerprint_2 = arr[grp_j]
                if distance == "jensenshannon":
                    dist: float = 1.0 * jensenshannon(fingerprint_1, fingerprint_2)
                elif distance == "correlation":
                    dist = 1.0 * correlation(fingerprint_1, fingerprint_2)
                else:
                    dist = 1.0 * euclidean(fingerprint_1, fingerprint_2)
                distance_matrix[grp_i, grp_j] = dist

        normalised_distance_matrix = distance_matrix / np.sum(distance_matrix)

        return pd.DataFrame(
            distance_matrix, columns=df.index, index=df.index
        ), pd.DataFrame(
            normalised_distance_matrix, columns=df.index, index=df.index, 
        )
    
    @staticmethod
    def calculate_pairwise_distances(
            df: pd.DataFrame, feature_names: List[str], distance: str,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if distance not in [
            "jensenshannon",
            "euclidean",
            "correlation",
        ]:
            raise Exception(
                "Invalid distance measure used to measure similarity."
            )

        distance_matrix: np.array = np.zeros((df.shape[0], df.shape[0]))
        arr = df.loc[:, feature_names].to_numpy()

        for grp_i in range(df.shape[0]):
            for grp_j in range(df.shape[0]):      
                fingerprint_1 = arr[grp_i]
                fingerprint_2 = arr[grp_j]
                if distance == "jensenshannon":
                    dist: float = 1.0 * jensenshannon(fingerprint_1, fingerprint_2)
                elif distance == "correlation":
                    dist = 1.0 * correlation(fingerprint_1, fingerprint_2)
                else:
                    dist = 1.0 * euclidean(fingerprint_1, fingerprint_2)
                distance_matrix[grp_i, grp_j] = dist

        normalised_distance_matrix = distance_matrix / np.sum(distance_matrix)

        return pd.DataFrame(
            distance_matrix, columns=df.index, index=df.index
        ), pd.DataFrame(
            normalised_distance_matrix, columns=df.index, index=df.index, 
        )

    @staticmethod
    def calculate_observable_patterns() -> None:
        # load data
        df: pd.DataFrame = ClusteringApplier._read_observable_data(
            read_only_feature_col=False
        )
        try:
            clustering_data = df.loc[:, list(config.OBSERVABLE_FEATURE_NAMES.keys())]
        except:
            print(f"Error. Invalid input.")
            return None

        # dendrogram observables
        ClusteringApplier._plot_dendrogram(
            df=clustering_data, x_label=config.OBSERVABLE_NAME, title=""
        )

        # clustering
        clusterer = AgglomerativeClustering(
            n_clusters=config.NUMBER_OBSERVABLE_PATTERNS, linkage="ward", compute_distances=True
        )

        clusterer.fit_predict(clustering_data)
        cluster_labels = clusterer.labels_

        df["pattern_type"] = cluster_labels

        df_cluster_median: pd.DataFrame = (
            df.copy()[list(config.OBSERVABLE_FEATURE_NAMES.keys()) + ["pattern_type"]]
            .groupby("pattern_type")
            .median()
        )

        # fingerprints
        try:
            # We keep both variants
            # Note that this might be a compatibility issue
            # For different versions of Panda
            df_observable_data = df[df["oversampled"] == False]
        except:
            df_observable_data = df[df["oversampled"] is False]

        df_fingerprint = ClusteringApplier._calculate_fingerprints(df_observable_data)

        pw_dist, pw_norm_dist = ClusteringApplier.calculate_pairwise_fingerprint_distances(
            df_fingerprint, config.DISTANCE_MEASURE_FINGERPRINT
        )

        ClusteringApplier._plot_dendrogram_by_distance_matrix(
            pw_norm_dist.to_numpy(),
            labels=pw_norm_dist.columns,
            x_label=config.GROUP_NAME,
            y_label="Distance of fingerprints",
            title="",
        )

        # write output
        output_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        df_cluster_median.to_excel(
            f"{output_path}{config.DATASET_NAME}-observable-patterns-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index_label="pattern_type",
            index=True
        )
        df_observable_data.to_excel(
            f"{output_path}{config.DATASET_NAME}-cluster_assignment-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=True,
        )
        df_fingerprint.to_excel(
            f"{output_path}{config.DATASET_NAME}-fingerprint-observables-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=True,
        )
        pw_dist.to_excel(
            f"{output_path}{config.DATASET_NAME}-distance-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=True
        )
        pw_norm_dist.to_excel(
            f"{output_path}{config.DATASET_NAME}-distance-normalized-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=True
        )

        ResultVisualizer.plot_simple_radar_chart(
            observable_patterns=[
                df_cluster_median.loc[row, :].to_list()
                for row in sorted(df_cluster_median.index)
            ],
            observable_labels=list(config.OBSERVABLE_FEATURE_NAMES.keys()),
        )

    @staticmethod
    def _get_correlation_coefficient(args) -> float:
        features: List[str] = args[0]
        df_explainable: pd.DataFrame = args[1]
        df_observable_distances: pd.DataFrame = args[2]

        _, df_explainable_distances = ClusteringApplier.calculate_pairwise_distances(
            df_explainable, features, config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES,
        )

        x = df_explainable_distances.to_numpy().flatten()
        y = df_observable_distances.to_numpy().flatten()

        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _feature_selection_exhaustive(
        df_explainable: pd.DataFrame,
        df_observable_distances: pd.DataFrame,
        features: List[str],
    ) -> Tuple[List[str], float]:

        powerset_features = chain.from_iterable(
            combinations(features, r) for r in range(2, len(features) + 1)
        )
        powerset_method_input = [
            (list(feature_set), df_explainable, df_observable_distances)
            for feature_set in powerset_features
        ]
        correlation_coefficients: List[float] = sorted(
            thread_map(
                ClusteringApplier._get_correlation_coefficient,
                powerset_method_input,
                desc="",
                max_workers=config.MAX_NUM_THREADS,
            )
        )

        maximum_correlation: float = max(correlation_coefficients)
        optimal_feature_set: List[str] = features
        current_feature_size = len(features)

        for index in range(len(correlation_coefficients)):
            if correlation_coefficients[index] == maximum_correlation:
                current_feature_set = powerset_method_input[index][0]
                if len(current_feature_set) < current_feature_size:
                    current_feature_size = len(current_feature_set)
                    optimal_feature_set = current_feature_set

        return optimal_feature_set, maximum_correlation

    @staticmethod
    def _feature_selection_greedy(
        df_explainable: pd.DataFrame,
        df_observable_distances: pd.DataFrame,
        features: List[str],
    ) -> Tuple[List[str], float]:
        currently_used_features: List[str] = []
        remaining_features: List[str] = features
        current_correlation_coefficient: float = -2.0

        while True:
            best_feature: str = ""
            for feature in remaining_features:
                feature_coeff: float = ClusteringApplier._get_correlation_coefficient(
                    [
                        currently_used_features + [feature],
                        df_explainable,
                        df_observable_distances,
                    ]
                )
                if feature_coeff > current_correlation_coefficient:
                    best_feature = feature
                    current_correlation_coefficient = feature_coeff
            if best_feature == "":
                break
            currently_used_features.append(best_feature)
            remaining_features.remove(best_feature)

        return currently_used_features, current_correlation_coefficient

    @staticmethod
    def calculate_explainable_distances():
        df_explainable: pd.DataFrame = ClusteringApplier.read_explaining_features()
        df_observable_distances: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_BASE}observables/{config.DATASET_NAME}-distance-normalized-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index_col=0
        )
        features: List[str] = list(config.EXPLAINING_FEATURE_NAMES.keys())

        # Ensure that both datasets contain the same indices
        valid_indices = np.intersect1d(df_explainable.index, df_observable_distances.index)
        df_explainable = df_explainable.loc[valid_indices]
        df_observable_distances = df_observable_distances.loc[valid_indices]

        if df_explainable is None or df_observable_distances is None:
            return

        if config.INFERENCE_MODE_EXPLAINING_FEATURES == "exact":
            optimal_feature_set, maximum_correlation = (
                ClusteringApplier._feature_selection_exhaustive(
                    df_explainable, df_observable_distances, features
                )
            )
        else:
            optimal_feature_set, maximum_correlation = (
                ClusteringApplier._feature_selection_greedy(
                    df_explainable, df_observable_distances, features
                )
            )

        overview_dict: Dict[str, List[str | float]] = {
            "correlation": [maximum_correlation]
        }
        for feature in features:
            overview_dict[feature] = [1 if feature in optimal_feature_set else 0]

        output_file: str = (
            f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-optimal_explainable_features-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
        )
        pd.DataFrame(overview_dict).to_excel(output_file, index=False)

        # draw dendrogram of those explaining set
        _, df_explainable_distances = ClusteringApplier.calculate_pairwise_distances(
            df_explainable, features, config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES
        )

        ClusteringApplier._plot_dendrogram_by_distance_matrix(
            mat=df_explainable_distances.to_numpy(),
            labels=features,
            x_label=config.GROUP_NAME,
            y_label="Distance based on explainable features",
            title="Similarity based on the optimal set of explainable features",
        )