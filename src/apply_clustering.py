import os.path
from itertools import chain, combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.cluster import AgglomerativeClustering
from clustering import OptimalK_Wrapper, agglomerative_clustering_function
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.distance import jensenshannon, correlation, euclidean
from tqdm.contrib.concurrent import thread_map

from visualize_result import ResultVisualizer
from kneed import KneeLocator


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
            df: pd.DataFrame = pd.read_excel(required_file).set_index(config.GROUP_NAME)
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
    def draw_gap_statistic_plot(
        use_config:bool=True, df_observable:pd.DataFrame=None,
        gap_statistic_cluster_range:int=10, observed_features=List[str]
        ) -> int:

        if use_config:
            df: pd.DataFrame = ClusteringApplier._read_observable_data()
            gap_statistic_cluster_range = config.GAP_STATISTIC_CLUSTER_RANGE
        else:
            df = df_observable.loc[:, observed_features]

        optimal_K = OptimalK_Wrapper(clusterer=agglomerative_clustering_function)

        X = df.to_numpy()
        cluster_range = np.arange(2, gap_statistic_cluster_range)
        n_clusters = optimal_K.find_optimal_K(X, cluster_array=cluster_range) 
        # n_clusters will be stored in optimal_K.n_clusters
        # and will be accessed this way in the next step.

        kn = KneeLocator(optimal_K.gap_df.n_clusters, optimal_K.gap_df.gap_value, curve='concave', direction='increasing')
        elblow_or_knee = kn.knee
        
        fig = optimal_K.plot_gaps(AgglomerativeClustering, knee=elblow_or_knee, size = (30, 7))

        if use_config:
            ouput_folder: str = f"{config.OUTPUT_FOLDER_BASE}gapstat/"
            if not os.path.exists(ouput_folder):
                os.makedirs(ouput_folder)
            fig.savefig(
                f"{ouput_folder}{config.DATASET_NAME}_gap-statistic-plot.pdf",
                bbox_inches="tight",
            )

        return min(n_clusters, elblow_or_knee)



    @staticmethod
    def _plot_dendrogram_by_distance_matrix(
        mat: np.array, labels: List[str], x_label: str, y_label: str, title: str,
        use_config:bool=True
    )->None|plt.Figure:
        
        fig = plt.figure(figsize=(20, 10))
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

        if use_config:
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
            return None
        else:
            return fig

    @staticmethod
    def _plot_dendrogram(
        df: pd.DataFrame,
        x_label: str,
        title: str,
        y_label: str = "Ward's distance measure",
        no_labels: bool = True,
        use_config:bool = True
    ) -> None:
        
        plt.figure(figsize=(20, 10))

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

        if use_config:
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
    def _calculate_fingerprints(df: pd.DataFrame, use_config:bool=True, group_name='') -> pd.DataFrame:

        if use_config:
            group_name = config.GROUP_NAME
        
        group_names: List[str] = list(set(df[group_name].to_list()))
        num_clusters: int = max(df["pattern_type"].to_list()) + 1
        group_stat: Dict[str, Dict[int, int]] = {
            grp_name: {j: 0 for j in range(num_clusters)}
            for grp_name in group_names
        }

        ret = {j: [] for j in range(num_clusters)}
        ret[group_name] = []

        for row in df.index:
            group_stat[df.loc[row, group_name]][
                int(df.loc[row, "pattern_type"])
            ] += 1

        for grp_name in group_stat.keys():
            quantities = group_stat[grp_name]
            N: int = np.sum(list(quantities.values()))
            for j in quantities.keys():
                ret[j].append(quantities[j] / N)
            ret[group_name].append(grp_name)

        return pd.DataFrame(ret).set_index(group_name).sort_index()
    
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
                    dist = 1.0 * correlation(fingerprint_1, fingerprint_2, centered=False)
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
                    dist = 1.0 * correlation(fingerprint_1, fingerprint_2, centered=False)
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
    def calculate_observable_patterns(
        df_observable_data:pd.DataFrame=None,
        _n_clusters:int = 2,
        use_config:bool=True,
        observable_name:str = '',
        observable_feature_names:List[str]|None=None,
        number_observable_patterns:str = 'auto',
        distance_measure_fingerprint:str = 'jensenshannon',
        group_name:str = '',
        observable_pattern_name:str = 'Name',
        spiderplot_scaling:str = 'none',
        plot_title:str = 'Title',
        scale_adjustment:bool = True
        ) -> dict:
        # load data

        if use_config:
            observable_name = config.OBSERVABLE_NAME
            observable_feature_names = config.OBSERVABLE_FEATURE_NAMES
            number_observable_patterns = config.NUMBER_OBSERVABLE_PATTERNS
            distance_measure_fingerprint = config.DISTANCE_MEASURE_FINGERPRINT
            group_name=config.GROUP_NAME
            plot_title = config.OBSERVABLE_PATTERN_NAME_PLURAL
            df: pd.DataFrame = ClusteringApplier._read_observable_data(
                read_only_feature_col=False
            )
            selected_features = list(observable_feature_names.keys()) 
        else:
            df = df_observable_data
            selected_features = observable_feature_names
        
        try:
            clustering_data = df.loc[:, selected_features]
        except:
            print(f"Error. Invalid input.")
            return None

        # dendrogram observables
        ClusteringApplier._plot_dendrogram(
            df=clustering_data, x_label=observable_name, title="", use_config=use_config
        )

        # clustering
        if str(number_observable_patterns).lower() == 'auto':
            n_clusters = _n_clusters
        else:
            n_clusters = number_observable_patterns
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward", compute_distances=True
        )

        clusterer.fit_predict(clustering_data)
        cluster_labels = clusterer.labels_

        df["pattern_type"] = cluster_labels

        df_cluster_median: pd.DataFrame = (
            df.copy()[selected_features + ["pattern_type"]]
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

        df_fingerprint = ClusteringApplier._calculate_fingerprints(
            df_observable_data.reset_index(),
            use_config=use_config,
            group_name=group_name
        )

        pw_dist, pw_norm_dist = ClusteringApplier.calculate_pairwise_fingerprint_distances(
            df_fingerprint, distance_measure_fingerprint
        )

        assert pw_dist.index.to_list() == pw_dist.columns.to_list(), 'Something went wrong: Expected pw_dist index to be equal to its columns.'
        assert pw_norm_dist.index.to_list() == pw_norm_dist.columns.to_list(), 'Something went wrong: Expected pw_norm_dist index to be equal to its columns.'

        dendrogram = ClusteringApplier._plot_dendrogram_by_distance_matrix(
            pw_norm_dist.to_numpy(),
            labels=pw_norm_dist.columns,
            use_config=use_config,
            x_label=group_name,
            y_label="Distance of fingerprints",
            title="",
        )
        
        if use_config:

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
            
        fig = ResultVisualizer.plot_simple_radar_chart(
            observable_patterns= np.array([
                df_cluster_median.loc[row, :].to_list()
                for row in sorted(df_cluster_median.index)
            ]),
            observable_labels=selected_features,
            use_config=use_config,
            observable_pattern_name=observable_pattern_name,
            spiderplot_scaling=spiderplot_scaling,
            observable_pattern_name_plural=plot_title,
            scale_adjustment=scale_adjustment  
        )

        output = {
                'df_cluster_median': df_cluster_median,
                'df_observable_data': df_observable_data,
                'df_fingerprint': df_fingerprint,
                'pw_dist': pw_dist,
                'pw_norm_dist': pw_norm_dist,
                'spider_plots': fig,
                'fingerprint_distance_plots': dendrogram
            }

        return output

    @staticmethod
    def _get_correlation_coefficient(args) -> float:
        features: List[str] = args[0]
        df_explainable: pd.DataFrame = args[1]
        df_observable_distances: pd.DataFrame = args[2]
        penalty_size: float = args[3]
        distance_measure = args[4]

        _, df_explainable_distances = ClusteringApplier.calculate_pairwise_distances(
            df_explainable, features, distance_measure,
        )

        x = df_explainable_distances.to_numpy().flatten()
        y = df_observable_distances.to_numpy().flatten()

        k = len(features)
        r = float(np.corrcoef(x, y)[0, 1])
        r_tilde = r - penalty_size*k
        
        return r_tilde

    @staticmethod
    def _feature_selection_exhaustive(
        df_explainable: pd.DataFrame,
        df_observable_distances: pd.DataFrame,
        features: List[str],
        use_config:bool=True,
        max_num_threads:int=4,
        sparsity_parameter:float=0.,
        distance_measure='correlation'


    ) -> Tuple[List[str], float]:
        
        if use_config:
            max_num_threads = config.MAX_NUM_THREADS
            sparsity_parameter = config.SPARSITY
            distance_measure = config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES

        powerset_features = chain.from_iterable(
            combinations(features, r) for r in range(2, len(features) + 1)
        )
        powerset_method_input = [
            [
                list(feature_set), df_explainable, df_observable_distances, sparsity_parameter, distance_measure]
            for feature_set in powerset_features
        ]
        correlation_coefficients: List[float] = thread_map(
            ClusteringApplier._get_correlation_coefficient,
            powerset_method_input,
            max_workers=max_num_threads,
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
        penalty_size: float,
        distance_measure: str
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
                        penalty_size,
                        distance_measure
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
    def calculate_explainable_distances(
        use_config:bool=True,
        df_explainable:pd.DataFrame|None=None,
        df_observable_distances:pd.DataFrame|None=None,
        explaining_features: List[str]|None=None,
        method:str = 'exact',
        distance_measure:str = 'correlation',
        max_num_threads:int=6,
        sparsity_parameter:float=0.,
        group_name:str='Index'
    ) -> dict:
        if use_config:
            df_explainable: pd.DataFrame = ClusteringApplier.read_explaining_features()
            df_observable_distances: pd.DataFrame = pd.read_excel(
                f"{config.OUTPUT_FOLDER_BASE}observables/{config.DATASET_NAME}-distance-normalized-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
                index_col=0
            )
            features: List[str] = list(config.EXPLAINING_FEATURE_NAMES.keys())
            method = config.INFERENCE_MODE_EXPLAINING_FEATURES
            distance_measure = config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES
            max_num_threads=config.MAX_NUM_THREADS
            sparsity_parameter=config.SPARSITY
        else:
            features = explaining_features
            if group_name in df_explainable.columns:
                df_explainable.set_index(group_name, inplace=True)

        # Ensure that both datasets contain the same indices
        valid_indices = np.intersect1d(df_explainable.index, df_observable_distances.index)
        valid_indices.sort()
        df_explainable = df_explainable.loc[valid_indices]
        df_observable_distances = df_observable_distances.loc[valid_indices]

        assert df_observable_distances.index.to_list() == df_observable_distances.columns.to_list(), 'Something went wrong: Expected df_observable_distances index to be equal to its columns.'

        if df_explainable is None or df_observable_distances is None:
            return
        
        if method == "exact":
            optimal_feature_set, maximum_correlation = (
                ClusteringApplier._feature_selection_exhaustive(
                    df_explainable=df_explainable,
                    df_observable_distances=df_observable_distances,
                    features=features,
                    use_config=use_config,
                    distance_measure=distance_measure,
                    max_num_threads=max_num_threads,
                    sparsity_parameter=sparsity_parameter, 
                )
            )
        else:
            optimal_feature_set, maximum_correlation = (
                ClusteringApplier._feature_selection_greedy(
                    df_explainable=df_explainable,
                    df_observable_distances=df_observable_distances,
                    features=features,
                    penalty_size=sparsity_parameter,
                    distance_measure=distance_measure
                )
            )

        overview_dict: Dict[str, List[str | float]] = {
            "correlation": [maximum_correlation]
        }
        for feature in features:
            overview_dict[feature] = [1 if feature in optimal_feature_set else 0]

        overview_df = pd.DataFrame(overview_dict)

        if use_config:
            output_path = f"{config.OUTPUT_FOLDER_BASE}explaining_features/"
            output_file: str = (
                f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-optimal_explainable_features-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            overview_df.to_excel(output_file, index=False)

        # draw dendrogram of those explaining set
        _, df_explainable_distances = ClusteringApplier.calculate_pairwise_distances(
            df_explainable, features, distance_measure
        )

        dendrogram = ClusteringApplier._plot_dendrogram_by_distance_matrix(
            mat=df_explainable_distances.to_numpy(),
            labels=list(df_explainable_distances.index),
            use_config=use_config,
            x_label=group_name,
            y_label="Distance based on explainable features",
            title="Similarity based on the optimal set of explainable features",
        )

        output = {
            'overview_df': overview_df,
            'df_explainable_distances': df_explainable_distances,
            'dendrogram': dendrogram,  
            'optimal_feature_set': optimal_feature_set      
        }

        return output

        

        