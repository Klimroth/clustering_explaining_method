import os.path
from typing import Dict, List

import numpy as np
import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

import config


# calculate fingerprints per group, use oversampled=False dataframe
# draw dendrogram and save the pairwise distances
## output -> excel with fingerprints, response types, dendrogram, pairwise distances


class ClusteringApplier:

    @staticmethod
    def read_observable_data(read_only_feature_col: bool = True) -> pd.DataFrame | None:
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
    def draw_gap_statistic_plot() -> None:

        df: pd.DataFrame = ClusteringApplier.read_observable_data()

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
            f"{output_path}{config.DATASET_NAME}_{title}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def _calculate_fingerprints(df: pd.DataFrame) -> pd.DataFrame:

        group_names: List[str] = list(set(df[config.GROUP_NAME].to_list()))
        num_clusters: int = max(df["pattern_type"].to_list()) + 1
        group_stat: Dict[str, Dict[int, int]] = {grp_name: {j: 0 for j in range(1, num_clusters + 1)} for grp_name in group_names}

        ret = {j: [] for j in range(1, num_clusters + 1)}
        ret[config.GROUP_NAME] = []

        for row in df.index:
            group_stat[df.loc[row, config.GROUP_NAME]][int(df.loc[row, 'pattern_type'])] += 1

        for grp_name in group_stat.keys():
            quantities = group_stat[grp_name]
            N: int = np.sum(list(quantities.values()))
            for j in quantities.keys():
                ret[j].append(quantities[j] / N)
            ret[config.GROUP_NAME].append(grp_name)

        return pd.DataFrame(ret)


    @staticmethod
    def calculate_observable_patterns() -> None:
        # load data
        df: pd.DataFrame = ClusteringApplier.read_observable_data(
            read_only_feature_col=False
        )
        try:
            clustering_data = df[
                list(config.OBSERVABLE_FEATURE_NAMES.keys())
            ].to_numpy()
        except:
            print(f"Error. Invalid input.")
            return None

        # dendrogram observables
        ClusteringApplier._plot_dendrogram(
            df=df, x_label=config.OBSERVABLE_NAME, title=""
        )

        # clustering
        clusterer = AgglomerativeClustering(
            n_clusters=config.NUMBER_OBSERVABLE_PATTERNS, linkage="ward"
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
        df_observable_data = df[df["oversampled"] is False]
        df_fingerprint = ClusteringApplier._calculate_fingerprints(df_observable_data)

        ClusteringApplier._plot_dendrogram(
            df=df_fingerprint, x_label=config.GROUP_NAME, title=""
        )

        # TODO: output pairwise distances between groups based on fignerprints



        output_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        df_cluster_median.to_excel(
            f"{output_path}{config.DATASET_NAME}-observable-patterns-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index_label="pattern_type",
        )
        df_observable_data.to_excel(
            f"{output_path}{config.DATASET_NAME}-cluster_assignment-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=False,
        )
        df_fingerprint.to_excel(
            f"{output_path}{config.DATASET_NAME}-fingerprint-observables-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx",
            index=False,
        )
