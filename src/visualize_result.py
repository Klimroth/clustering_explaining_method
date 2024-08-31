import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import config
from apply_clustering import ClusteringApplier


class ResultVisualizer:

    @staticmethod
    def visualize_results():
        df_explainable: pd.DataFrame = ClusteringApplier.read_explaining_features()

        df_optimal_explainable: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_BASE}explaining_features/{config.DATASET_NAME}-optimal_explainable_features-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
        )
        explainable_features: List[str] = []
        for feature in config.EXPLAINING_FEATURE_NAMES.keys():
            if df_optimal_explainable.loc[0, feature] == 1:
                explainable_features.append(feature)

        df_observable_distances: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_BASE}observables/{config.DATASET_NAME}-distance-normalized-matrix-{config.DISTANCE_MEASURE_FINGERPRINT}-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
        )
        _, df_explainable_distances = ClusteringApplier.calculate_pairwise_distances(
            df_explainable, explainable_features
        )

        ResultVisualizer.make_regression_plot(df_explainable_distances, df_observable_distances)

        # TODO
        # read observable patterns and plot radar chart
        # for each of the patterns make one radar plot (scaled to [0,1])

        # read fingerprints per group with respect to observable patterns and plot radar chart
        # read explainable patterns per group and plot radar chart
        # --> as a subplot per group: fingerprint | explainable pattern
        # --> call plot_radar_chart(..., title="groupname") for each group




    @staticmethod
    def plot_radar_chart(
        simplex_coordinates_fingerprint: List[float],
        categories_fingerprint: List[str],
        simplex_coordinates_explainable: List[float],
        categories_explainable: List[str],
        title: str,
    ):

        # normalize to length one
        simplex_coordinates_fingerprint = np.array(
            simplex_coordinates_fingerprint
        ) / np.sum(np.array(simplex_coordinates_fingerprint))
        simplex_coordinates_explainable = np.array(
            simplex_coordinates_explainable
        ) / np.sum(np.array(simplex_coordinates_explainable))

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("observable patterns", "explainable features"),
        )

        fig.add_trace(
            go.Scatterpolar(
                r=simplex_coordinates_fingerprint,
                theta=categories_fingerprint,
                fill="toself",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatterpolar(
                r=simplex_coordinates_explainable,
                theta=categories_explainable,
                fill="toself",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            title=title,
        )

        fig.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
            }
        )

        output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.write_image(
            f"{output_path}{config.DATASET_NAME}-{title}-radar_plot-{config.NUMBER_OBSERVABLE_PATTERNS}.png",
            dpi=300,
        )

    @staticmethod
    def make_regression_plot(
        distance_explainable: pd.DataFrame, distance_observable: pd.DataFrame
    ):

        if set(distance_explainable.index.to_list()) != set(
            distance_observable.index.to_list
        ):
            print(
                "ERROR: Explainables and observables have a different index. Cannot make distance regression."
            )
            return

        distances: Dict[str, List[str | float]] = {
            "distance explainable features": [],
            "distance observable features": [],
            "groupname_1": [],
            "groupname_2": [],
        }

        group_names: List[str] = distance_explainable.index.to_list()
        for j in range(len(group_names)):
            for i in range(j + 1, len(group_names)):
                distances["distance explainable features"].append(
                    distance_explainable.loc[group_names[j], group_names[i]]
                )
                distances["distance observable features"].append(
                    distance_observable.loc[group_names[j], group_names[i]]
                )
                distances["groupname_1"].append(group_names[j])
                distances["groupname_2"].append(group_names[i])

        df = pd.DataFrame(distances)
        fig = px.scatter(
            df,
            x="distance explainable features",
            y="distance observable features",
            trendline="ols",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )

        output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.write_image(
            f"{output_path}{config.DATASET_NAME}-regression_plot-{config.NUMBER_OBSERVABLE_PATTERNS}.png",
            dpi=300,
        )


# make radar plots:
# - fingerprint per group
# - scaled variants of observable patterns and of explainable features (as radar plot)
# - linear regression (x = distance observable, y = distance explainable)
