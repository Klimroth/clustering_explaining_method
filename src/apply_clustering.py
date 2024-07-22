### get the fingerprints
# read prepared explaining features data sheet
# draw gap statistic plot (based on config)
# based on config, apply clustering
# save the response types (observe that this are still scaled variables!)
import os.path

import pandas as pd
from gapstatistics.gapstatistics import GapStatistics
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

import config


# calculate fingerprints per group, use oversampled=False dataframe
# draw dendrogram and save the pairwise distances
## output -> excel with fingerprints, response types, dendrogram, pairwise distances


class ClusteringApplier:

    @staticmethod
    def read_observable_data() -> pd.DataFrame | None:
        required_file: str = (
            f"{config.OUTPUT_FOLDER_BASE}base_data/{config.DATASET_NAME}_observable_dataset_scaled.xlsx"
        )
        try:
            df: pd.DataFrame = pd.read_excel(required_file)
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
        plt.savefig(
            f"{config.OUTPUT_FOLDER_BASE}gapstat/{config.DATASET_NAME}_gap-statistic-plot.pdf",
            bbox_inches="tight",
        )
