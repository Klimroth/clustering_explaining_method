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
import warnings

from utils import minmax, calculate_homogeneity

try:
    import kaleido
except:
    warnings.warn('kaleido not found, try pip install --upgrade "kaleido==0.1.*"')

if kaleido.__version__ != '0.1.0.post1':
    warnings.warn(f'kaleido version {kaleido.__version__} may not be able to save the resulting plots, if you encounter problems try using version 0.1.0.post1 instead')



class ResultVisualizer:

    @staticmethod
    def visualize_results():
        pass
        '''
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
            df_explainable, explainable_features, config.DISTANCE_MEASURE_EXPLAINABLE_FEATURES
        )

        ResultVisualizer.make_regression_plot(
            df_explainable_distances, df_observable_distances
        )

        df_fingerprint: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_BASE}observables/{config.DATASET_NAME}-fingerprint-observables-{config.NUMBER_OBSERVABLE_PATTERNS}.xlsx"
        )
        column_headings: List[int] = [
            j + 1 for j in range(config.NUMBER_OBSERVABLE_PATTERNS)
        ]

        categories_fingerprints: List[str] = [
            f"{config.OBSERVABLE_PATTERN_NAME} {j}" for j in column_headings
        ]
        group_names: List[str] = sorted(df_fingerprint[config.GROUP_NAME].to_list())
        for group_name in group_names:
            ResultVisualizer.plot_result_radar_chart(
                simplex_coordinates_fingerprint=df_fingerprint[
                    df_fingerprint[config.GROUP_NAME] == group_name
                ][column_headings].to_list(),
                categories_fingerprint=categories_fingerprints,
                simplex_coordinates_explainable=df_explainable[
                    df_explainable[config.GROUP_NAME] == group_name
                ][explainable_features].to_list(),
                categories_explainable=explainable_features,
                title=group_name,
            )'''

    @staticmethod
    def plot_simple_radar_chart(
        observable_patterns: List[List[float]], observable_labels: List[str],
        max_fingerprints_per_col: int = 2
    ):
        subplot_titles: List[str] = [
            f"{config.OBSERVABLE_PATTERN_NAME} {j+1}"
            for j in range(len(observable_patterns))
        ]

        if config.SPIDERPLOT_SCALING == 'minmax':
            scale = minmax
        elif config.SPIDERPLOT_SCALING == 'none':
            scale = lambda x: x
        else:
            warnings.warn(f'Unknown SPIDERPLOT_SCALING {config.SPIDERPLOT_SCALING}. No scaling will be used.')
            scale = lambda x: x
 
        scaled_observable_patterns = scale(observable_patterns)

        if np.min(scaled_observable_patterns) < 0:
            plot_range = [-1, 1]
        else:
            plot_range = [0, 1]

        num_rows = int(np.ceil(len(scaled_observable_patterns)/max_fingerprints_per_col))
        num_cols = min(len(scaled_observable_patterns), max_fingerprints_per_col)

        fig = make_subplots(
            rows=num_rows, cols=num_cols, specs=[[{'type': 'polar'}]*num_cols]*num_rows,
            horizontal_spacing=0.3, vertical_spacing=0.05,
            subplot_titles=[f'Fingerprint {j+1} (homogeneity: {calculate_homogeneity(observable_patterns)[j]:.2f})' for j in range(len(scaled_observable_patterns))]
        )

        for j in range(len(scaled_observable_patterns)):
            row = j // max_fingerprints_per_col + 1
            col = j % num_cols + 1
            adj_scale = np.sum(np.array(observable_patterns[j])) if config.ADJ_SCALE else 1
            current_pattern: np.array = np.array(scaled_observable_patterns[j]) / adj_scale
            fig.add_scatterpolar(r=current_pattern, theta=observable_labels, fill="toself", row=row, col=col)

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            title=config.OBSERVABLE_PATTERN_NAME_PLURAL,
        )

        fig.update_polars(dict(radialaxis=dict(visible=True, range=plot_range, showticklabels=False)))
        fig.update_layout(height=500*num_rows, width=1000)

        output_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        fig.write_image(
            f"{output_path}{config.DATASET_NAME}-{config.OBSERVABLE_PATTERN_NAME_PLURAL}-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
        )

    @staticmethod
    def plot_result_radar_chart(
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
            subplot_titles=["observable patterns", "explainable features"],
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
