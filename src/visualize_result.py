from typing import List

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

        # TODO
        # read observable patterns and plot radar chart

        # read fingerprints per group with respect to observable patterns and plot radar chart
        # read explainable patterns per group and plot radar chart
        # --> as a subplot per group: fingerprint | explainable pattern

        # TODO 2
        # read pairwise distances and make linear regression plot

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

        fig.write_image("...")

    @staticmethod
    def make_regression_plot(
        distance_explainable: pd.DataFrame, distance_observable: pd.DataFrame
    ):

        # TODO: join the dataframes such that we get "pair" "explainable" "observable",

        fig = px.scatter(df, x="total_bill", y="tip", trendline="ols")
        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        fig.write_image("...")


# make radar plots:
# - fingerprint per group
# - scaled variants of observable patterns and of explainable features (as radar plot)
# - linear regression (x = distance observable, y = distance explainable)
