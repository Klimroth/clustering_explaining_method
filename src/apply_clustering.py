import os.path
from itertools import chain, combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.cluster import AgglomerativeClustering
from clustering import OptimalK_Wrapper, agglomerative_clustering_function, make_agglomerative_clustering_function
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.distance import jensenshannon, correlation, euclidean
from tqdm.contrib.concurrent import thread_map
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from numpy.random import permutation

from visualize_result import ResultVisualizer
from kneed import KneeLocator

from collections import defaultdict
from utils import calculate_homogeneity


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
        gap_statistic_cluster_range:int=10, observed_features=List[str],
        linkage='ward',
        plot:bool=True
        ) -> int:

        if use_config:
            df: pd.DataFrame = ClusteringApplier._read_observable_data()
            gap_statistic_cluster_range = config.GAP_STATISTIC_CLUSTER_RANGE
        else:
            df = df_observable.loc[:, observed_features]

        my_agglomerative_clustering_function = make_agglomerative_clustering_function(linkage=linkage)
        optimal_K = OptimalK_Wrapper(clusterer=my_agglomerative_clustering_function, clusterer_kwargs={
                     'linkage': linkage
        })

        X = df.to_numpy()
        cluster_range = np.arange(2, gap_statistic_cluster_range)
        n_clusters = optimal_K.find_optimal_K(X, cluster_array=cluster_range) 
        # n_clusters will be stored in optimal_K.n_clusters
        # and will be accessed this way in the next step.

        kn = KneeLocator(optimal_K.gap_df.n_clusters, optimal_K.gap_df.gap_value, curve='concave', direction='increasing')
        elb = KneeLocator(optimal_K.gap_df.n_clusters, optimal_K.gap_df.gap_value, curve='convex', direction='decreasing')

        knee = kn.knee
        elbow = elb.elbow
        
        if plot:
            fig = optimal_K.plot_gaps(AgglomerativeClustering, knee=knee, elbow=elbow, size = (30, 7))

        if use_config:
            ouput_folder: str = f"{config.OUTPUT_FOLDER_BASE}gapstat/"
            if not os.path.exists(ouput_folder):
                os.makedirs(ouput_folder)
            fig.savefig(
                f"{ouput_folder}{config.DATASET_NAME}_gap-statistic-plot.pdf",
                bbox_inches="tight",
            )

        return {
            'n_clusters': n_clusters,
            'knee': knee,
            'elbow': elbow,
            'clusterer': optimal_K
        }

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
    def rowwise_calculate_pairwise_distances(
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
    def numpy_matrix_correlation(A:np.array) -> np.array:

        """
        Compute the pairwise correlation distance between elements in a 2-D matrix.

        The correlation distance between two `u` and `v` in `A`, is
        defined as

        .. math::

            1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                    {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}

        where :math:`\\bar{u}` is the mean of the elements of `u`
        and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

        Parameters
        ----------
        A : (N,M) array_like
            Input array.
        w : (N,) array_like, optional
            The weights for each value in `u` and `v`. Default is None,
            which gives each value a weight of 1.0
        centered : bool, optional
            If True, `u` and `v` will be centered. Default is False.

        Returns
        -------
        correlation : double
            The correlation distances in 2-D array `A`.
        """
        
        AA = np.matmul(A, A.T)
        diag = np.diag(AA)
        dist = 1.0 - AA / np.sqrt(diag.reshape(-1, 1) * diag.reshape(1, -1))
        # Clip the result to avoid rounding error
        return np.clip(dist, 0.0, 2.0)

    @staticmethod
    def numpy_relative_entropy(x:np.array, y:np.array) -> np.array:
        return (x * np.log(x/y))

    @staticmethod
    def numpy_matrix_jensenshannon(A:np.array, epsilon:float=1e-20) -> np.array:

        ########################################################################################
        ### Conversion of the jensenshannon distance from scipy.spatial.distance to pytorch. ###
        ########################################################################################

        """
        Compute the pairwise Jensen-Shannon distance (metric) between
        all entries in a matrix. This is the square root
        of the Jensen-Shannon divergence.

        The Jensen-Shannon distance between two probability
        vectors `p` and `q` is defined as,

        .. math::

        \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}

        where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
        and :math:`D` is the Kullback-Leibler divergence.

        This routine will normalize `p` and `q` if they don't sum to 1.0.

        Parameters
        ----------
        A : (N,M) array_like
            matrix of probability vectors
        
        Returns
        -------
        js : double or ndarray
            The Jensen-Shannon distances along the `axis`.

        """
        
        clamped_A = A.clip(min = epsilon)
        normalized_A = clamped_A / np.expand_dims(np.sum(clamped_A, axis=1), -1)
        m_A = np.permute_dims(np.expand_dims(normalized_A, -1) + normalized_A.T, (2,0,1)) / 2.0
        relative_entropies = ClusteringApplier.numpy_relative_entropy(normalized_A, m_A) + epsilon
        js = (relative_entropies.sum(axis = 2) + relative_entropies.sum(axis = 2).T).clip(min = epsilon)
        return np.sqrt(js / 2.0)

    @staticmethod
    def calculate_pairwise_distances(
            df: pd.DataFrame, feature_names: List[str], distance: str,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if distance not in [
            "jensenshannon",
            "correlation",
        ]:
            raise Exception(
                "Invalid distance measure used to measure similarity."
            )
        
        fingerprint_df = df.loc[:, feature_names]
        fingerprint_array = fingerprint_df.to_numpy()

        if distance == "jensenshannon":
            distance_matrix = ClusteringApplier.numpy_matrix_jensenshannon(fingerprint_array)
        elif distance == "correlation":
            distance_matrix = ClusteringApplier.numpy_matrix_correlation(fingerprint_array)
        else:
            raise Exception(f'Not Implemented: {distance}')


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
        plot_title:str = 'Title',
        linkage:str= 'ward',
        plot:bool=True
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
        if plot:
            ClusteringApplier._plot_dendrogram(
                df=clustering_data, x_label=observable_name, title="", use_config=use_config
            )

        # clustering
        if str(number_observable_patterns).lower() == 'auto':
            n_clusters = _n_clusters
        else:
            n_clusters = number_observable_patterns
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, compute_distances=True,
        )

        clusterer.fit_predict(clustering_data.copy())
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

        if plot:
            dendrogram = ClusteringApplier._plot_dendrogram_by_distance_matrix(
                pw_norm_dist.to_numpy(),
                labels=pw_norm_dist.columns,
                use_config=use_config,
                x_label=group_name,
                y_label="Distance of fingerprints",
                title="",
            )
        else:
            dendrogram = None
        
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
        
        if plot:
            fig = ResultVisualizer.plot_simple_radar_chart(
                observable_patterns= np.array([
                    df_cluster_median.loc[row, :].to_list()
                    for row in sorted(df_cluster_median.index)
                ]),
                observable_labels=selected_features,
                use_config=use_config,
                observable_pattern_name=observable_pattern_name,
                observable_pattern_name_plural=plot_title,
            )
        else:
            fig = None

        hom = calculate_homogeneity(df_fingerprint)
        hom_df = pd.DataFrame(hom, index = df_fingerprint.index).T

        if use_config:
            if plot:
                ResultVisualizer.plot_homogeneity(hom_df=hom_df)
            df_hom = df_fingerprint.copy()
            df_hom.loc[:, 'Homogeneity'] = hom_df.T
            df_hom.to_excel(
                f"{output_path}{config.DATASET_NAME}-homogeneity-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
                index=True
            )

        output = {
                'df_cluster_median': df_cluster_median,
                'df_observable_data': df_observable_data,
                'df_fingerprint': df_fingerprint,
                'pw_dist': pw_dist,
                'pw_norm_dist': pw_norm_dist,
                'spider_plots': fig,
                'fingerprint_distance_plots': dendrogram,
                'clusterer': clusterer,
                'clustering_data': clustering_data,
                'hom_df': hom_df
            }

        return output

    @staticmethod
    def _get_correlation_coefficient(args) -> Tuple:
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
        r = abs(float(np.corrcoef(x, y)[0, 1]))
        r_tilde = r - penalty_size*k
        
        return r_tilde, r

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
            combinations(features, r) for r in range(1, len(features) + 1)
        )
        powerset_method_input = [
            [
                list(feature_set), df_explainable, df_observable_distances, sparsity_parameter, distance_measure]
            for feature_set in powerset_features
        ]
        correlation_coefficient_tuples: List[Tuple] = thread_map(
            ClusteringApplier._get_correlation_coefficient,
            powerset_method_input,
            max_workers=max_num_threads,
        )

        correlation_coefficient_tuples = np.array(correlation_coefficient_tuples)
        correlation_coefficients_with_penalty = correlation_coefficient_tuples[:, 0]
        correlation_coefficients_without_penalty = correlation_coefficient_tuples[:, 1]
        
        maximum_correlation_with_penalty: float = max(correlation_coefficients_with_penalty)
        maximum_correlation_without_penalty = -np.inf
        optimal_feature_set: List[str] = features

        for index in range(len(correlation_coefficients_with_penalty)):
            if correlation_coefficients_with_penalty[index] == maximum_correlation_with_penalty:
                current_feature_set = powerset_method_input[index][0]
                current_correlation_without_penalty = correlation_coefficients_without_penalty[index]
                if current_correlation_without_penalty > maximum_correlation_without_penalty:
                    maximum_correlation_without_penalty = current_correlation_without_penalty
                    optimal_feature_set = current_feature_set

        return optimal_feature_set, maximum_correlation_without_penalty

    @staticmethod
    def __feature_selection_greedy(
        df_explainable: pd.DataFrame,
        df_observable_distances: pd.DataFrame,
        features: List[str],
        penalty_size: float,
        distance_measure: str
    ) -> Tuple[List[str], float]:

        currently_used_features: List[str] = []
        remaining_features: List[str] = features
        current_correlation_coefficient_with_penalty: float = -np.inf
        current_correlation_coefficient_without_penalty: float = -np.inf

        while True:
            best_feature: str = ""
            for feature in remaining_features:
                feature_coeff_with_penalty, feature_coeff_without_penalty = ClusteringApplier._get_correlation_coefficient(
                    [
                        currently_used_features + [feature],
                        df_explainable,
                        df_observable_distances,
                        penalty_size,
                        distance_measure
                    ]
                )
                if feature_coeff_with_penalty > current_correlation_coefficient_with_penalty:
                    best_feature = feature
                    current_correlation_coefficient_with_penalty = feature_coeff_with_penalty
                    current_correlation_coefficient_without = feature_coeff_without_penalty
            if best_feature == "":
                break
            currently_used_features.append(best_feature)
            remaining_features.remove(best_feature)

        return currently_used_features, current_correlation_coefficient_without
    
    @staticmethod
    def _feature_selection_greedy(
        df_explainable: pd.DataFrame,
        df_observable_distances: pd.DataFrame,
        features: List[str],
        use_config:bool=True,
        sparsity_parameter: float=0.,
        distance_measure: str='correlation',
        N: int = 2
    ) -> Tuple[List[str], float]:
        
        if use_config:
            sparsity_parameter = config.SPARSITY
            distance_measure = config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES

        subset_powerset_features = chain.from_iterable(
            combinations(features, r) for r in range(1, N+1)
        )
        subset_powerset_method_input = [
            [
                list(feature_set), df_explainable, df_observable_distances, sparsity_parameter, distance_measure]
            for feature_set in subset_powerset_features
        ]
        subset_correlation_coefficient_tuples: List[Tuple] = thread_map(
            ClusteringApplier._get_correlation_coefficient,
            subset_powerset_method_input,
            max_workers=config.MAX_NUM_THREADS,
        )

        subset_correlation_coefficient_tuples = np.array(subset_correlation_coefficient_tuples)
        subset_correlation_coefficients_with_penalty = subset_correlation_coefficient_tuples[:, 0]
        subset_correlation_coefficients_without_penalty = subset_correlation_coefficient_tuples[:, 1]

        feature_importances = defaultdict(lambda: 0.)
        part_to_sum_feature_importances = defaultdict(lambda: 0.)

        for j in range(len(features), len(subset_powerset_method_input)):
            feature_subset = subset_powerset_method_input[j][0]
            importance = subset_correlation_coefficients_with_penalty[j]
            feature_importances[j] = importance

        cross_importance_df = pd.DataFrame(index = features, columns= features)

        for j in range(len(features), len(subset_powerset_method_input)):
            feature_subset = subset_powerset_method_input[j][0]
            importance = subset_correlation_coefficients_with_penalty[j]
            cross_importance_df.loc[feature_subset[0], feature_subset[1]] = importance
            cross_importance_df.loc[feature_subset[1], feature_subset[0]] = importance

        starting_index = np.argmax(list(feature_importances.values()))
        starting_pair = list(feature_importances.keys())[starting_index]
        starting_features = subset_powerset_method_input[starting_pair][0]

        selected_features = starting_features
        print(f'Started with {starting_features}')

        _current_score = ClusteringApplier._get_correlation_coefficient([
            selected_features,
            df_explainable,
            df_observable_distances,
            sparsity_parameter,
            distance_measure
        ])
        current_score_with_penalty = _current_score[0]
        current_score_without_penalty = _current_score[1]

        base_feature_importances = defaultdict(lambda: [])
        for i in range(len(features)):
            feature = subset_powerset_method_input[i][0][0]
            base_value = subset_correlation_coefficients_with_penalty[i]
            base_feature_importances[feature].append(base_value)

        base_df = pd.DataFrame(base_feature_importances)
        best_base = base_df.T.iloc[base_df.sum(axis=0).argmax()]
        best_base_feature = best_base.name
        best_base_score = float(best_base.values[0])

        while True:
            
            remaining_indices = cross_importance_df.loc[:, selected_features].drop(selected_features).sum(axis=1)
            if len(remaining_indices) == 0:
                break

            score_dict = {}
            score_without_penalty_dict = {}
            for _feature in cross_importance_df.drop(selected_features).index:
                _score = ClusteringApplier._get_correlation_coefficient([
                    selected_features + [_feature],
                    df_explainable,
                    df_observable_distances,
                    sparsity_parameter,
                    distance_measure
                ])
                score_without_penalty_dict[str(selected_features + [_feature])] = _score[1]
                score_dict[_score[0]] = _feature
        
            next_score = max(score_dict.keys())
            next_feature = score_dict[next_score]

            if next_score > current_score_with_penalty:
                print(f"Added '{next_feature}'")
                selected_features = selected_features + [next_feature]
                current_score_with_penalty = next_score
                current_score_without_penalty = score_without_penalty_dict[str(selected_features)]
            else:
                print(f"Refuted '{next_feature}'")
                break
        
        if current_score_with_penalty <= best_base_score:
            selected_features = [best_base_feature]
            print(f'Override: Base feature performed best: {best_base_feature}')

        best_score_with_penalty = ClusteringApplier._get_correlation_coefficient([
            selected_features,
            df_explainable,
            df_observable_distances,
            sparsity_parameter,
            distance_measure
        ])[1]

        return selected_features, best_score_with_penalty

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
        heuristics_N:int=2,
        higher_order_importance_k:int=100,
        group_name:str='Index',
        debug:bool=False,
        plot:bool=True
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
            heuristics_N=config.HEURISTIC_N
            higher_order_importance_k = config.HIGHER_ORDER_IMPORTANCE_K
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
                    use_config=use_config,
                    sparsity_parameter=sparsity_parameter,
                    distance_measure=distance_measure,
                    N = heuristics_N
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

        if debug:
            _, rdf_explainable_distances = ClusteringApplier.rowwise_calculate_pairwise_distances(
                df_explainable, features, distance_measure
            )
            assert np.isclose(df_explainable_distances.to_numpy(), rdf_explainable_distances.to_numpy()).all().all(), 'Debug Failed'

        if plot:
            dendrogram = ClusteringApplier._plot_dendrogram_by_distance_matrix(
                mat=df_explainable_distances.to_numpy(),
                labels=list(df_explainable_distances.index),
                use_config=use_config,
                x_label=group_name,
                y_label="Distance based on explainable features",
                title="Similarity based on the optimal set of explainable features",
            )
        else:
            dendrogram = None

        optimal_feature_importances_df = ClusteringApplier._higher_order_feature_importance(
            feature_set = optimal_feature_set,
            df_explainable = df_explainable,
            df_observable_distances = df_observable_distances,
            use_config=use_config,
            K = higher_order_importance_k
        )

        if use_config:
            output_path = f"{config.OUTPUT_FOLDER_BASE}explaining_features/"
            output_file: str = (
                f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-optimal_explainable_feature_importances-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            optimal_feature_importances_df.to_excel(output_file, index=True)
            exp_file: str = (
                f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-pairwise_explainable_distances-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
            )
            df_explainable_distances.to_excel(exp_file, index=True)

        output = {
            'overview_df': overview_df,
            'df_explainable_distances': df_explainable_distances,
            'dendrogram': dendrogram,  
            'optimal_feature_set': optimal_feature_set,
            'optimal_feature_importances': optimal_feature_importances_df
        }

        return output
    

    @staticmethod
    def _higher_order_feature_importance(
        feature_set:List,
        df_explainable:pd.DataFrame,
        df_observable_distances:pd.DataFrame,
        use_config:bool,
        K:int=100,
    ):

        # Def. 4: Higher-order permutation-based feature importance
        my_feature_set = np.array(feature_set)

        X = df_explainable.copy().loc[:, my_feature_set].to_numpy()
        y = df_observable_distances.copy().to_numpy()

        feature_combinations = []
        ([[feature_combinations.append(np.array(c)) for c in combinations(np.arange(len(my_feature_set)), i)] for i in range(1, len(my_feature_set)+1)])

        model = LinearRegression()
        model.fit(X,y)
        y_pred = model.predict(X)
        R2 = explained_variance_score(y_true=y, y_pred=y_pred)
        feature_importances = defaultdict(lambda: 0.)

        i = 0
        for cols in feature_combinations:
            for k in range(K):
                permuted_X = X.copy()
                for col_i in cols:
                    permuted_X[:, col_i] = permutation(permuted_X[:, col_i])
                y_pred = model.predict(permuted_X)
                feature_importances[i] += explained_variance_score(y_true=y, y_pred=y_pred)
            i += 1

        i = 0
        for cols in feature_combinations:
            feature_importances[i] = max(0, R2 - (feature_importances[i]/K))
            i += 1
            
        adj_sum = sum(feature_importances.values())
        for key, val in feature_importances.items():
            feature_importances[key] = val / adj_sum

        assert np.isclose(sum(feature_importances.values()), 1)

        def human_readable(_dict: dict) -> dict:
            return {str([str(a) for a in my_feature_set[feature_combinations[k]]]):[v] for k, v in _dict.items()}

        human_readable_dict = human_readable(feature_importances)
        
        if use_config:
            ResultVisualizer.plot_higher_order_feature_importances(
                human_readable_dict=human_readable_dict
            )
        
        output_df = pd.DataFrame(human_readable_dict).T.rename(columns={0:'Importance Score'})
        return output_df
    
    @staticmethod
    def visualize_fingerprints(
        df_fingerprint:pd.DataFrame|None=None,
        df_explainable:pd.DataFrame|None=None,
        optimal_features: List[str]|None=None,
        use_config:bool=True
    ):
        
        if use_config:
            input_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
            df_explainable: pd.DataFrame = ClusteringApplier.read_explaining_features()
            df_fingerprint = pd.read_excel(
                f"{input_path}{config.DATASET_NAME}-fingerprint-observables-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
                index_col=0
            )
            input_file: str = (
                f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-optimal_explainable_features-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
            )
            optimal_df = pd.read_excel(input_file, index_col=0)
            optimal_features = list(optimal_df.T[optimal_df.T > 0].dropna().T.columns)

        # Ensure that both datasets contain the same indices
        valid_indices = np.intersect1d(df_explainable.index, df_fingerprint.index)
        valid_indices.sort()
        df_explainable = df_explainable.loc[valid_indices]
        df_fingerprint = df_fingerprint.loc[valid_indices]

        ResultVisualizer.plot_result_radar_chart(
            simplex_coordinates_fingerprint=df_fingerprint,
            categories_fingerprint=[f'Pattern_{i}' for i in df_fingerprint.columns],
            simplex_coordinates_explainable=df_explainable,
            categories_explainable=df_explainable.columns,
            use_config=use_config,
            title='fingerprints_all_features'
        )

        ResultVisualizer.plot_result_radar_chart(
            simplex_coordinates_fingerprint=df_fingerprint,
            categories_fingerprint=[f'Pattern_{i}' for i in df_fingerprint.columns],
            simplex_coordinates_explainable=df_explainable.loc[:, optimal_features],
            categories_explainable=df_explainable.loc[:, optimal_features].columns,
            use_config=use_config,
            title='fingerprints_optimal_features'
        )

    @staticmethod
    def visualize_regression(
        df_explainable_distances:pd.DataFrame|None=None,
        distance_observable_distances:pd.DataFrame|None=None,
        use_config:bool=True
    ):
        
        if use_config:
            exp_file: str = (
                f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-pairwise_explainable_distances-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
            )
            df_explainable_distances = pd.read_excel(exp_file, index_col=0)
            obs_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
            distance_observable_distances = pd.read_excel(
                f"{obs_path}{config.DATASET_NAME}-distance-normalized-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
                index_col=0
            )

        return ResultVisualizer.make_regression_plot(
            distance_explainable=df_explainable_distances,
            distance_observable=distance_observable_distances,
            use_config=use_config
        )
        





        

        