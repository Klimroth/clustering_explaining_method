import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import seaborn as sns

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
    def plot_simple_radar_chart(
        observable_patterns: List[List[float]], observable_labels: List[str],
        max_fingerprints_per_col: int = 2,
        use_config:bool = True,
        observable_pattern_name:str = 'Name',
        observable_pattern_name_plural:str = 'Names',
        spacing = {
            'horizontal_spacing': 0.3,
            'vertical_spacing': 0.05,
            'height': 500,
            'width': 1000,
            'row_heights': None
        }
    ):
        
        if use_config:
            observable_pattern_name = config.OBSERVABLE_PATTERN_NAME
            observable_pattern_name_plural = config.OBSERVABLE_PATTERN_NAME_PLURAL        

        subplot_titles: List[str] = [
            f"{observable_pattern_name} {j+1}"
            for j in range(len(observable_patterns))
        ]
 
        scaled_observable_patterns = observable_patterns

        if np.min(scaled_observable_patterns) < 0:
            plot_range = [-1, 1]
        else:
            plot_range = [0, 1]

        num_rows = int(np.ceil(len(scaled_observable_patterns)/max_fingerprints_per_col))
        num_cols = min(len(scaled_observable_patterns), max_fingerprints_per_col)

        fig = make_subplots(
            rows=num_rows, cols=num_cols, specs=[[{'type': 'polar'}]*num_cols]*num_rows,
            horizontal_spacing=spacing['horizontal_spacing'], vertical_spacing=spacing['vertical_spacing'], row_heights=spacing['row_heights'],
            subplot_titles=subplot_titles #[f'Cluster {j+1}' for j in range(len(scaled_observable_patterns))]
        )

        for j in range(len(scaled_observable_patterns)):
            row = j // max_fingerprints_per_col + 1
            col = j % num_cols + 1
            current_pattern: np.array = np.array(scaled_observable_patterns[j])
            fig.add_scatterpolar(r=current_pattern, theta=observable_labels, fill="toself", row=row, col=col)

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            title=observable_pattern_name_plural,
        )

        fig.update_polars(dict(radialaxis=dict(visible=True, range=plot_range, showticklabels=False)))
        fig.update_layout(height=spacing['height']*num_rows, width=spacing['width'])

        if use_config:
            output_path = f"{config.OUTPUT_FOLDER_BASE}observables/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            fig.write_image(
                f"{output_path}{config.DATASET_NAME}-{config.OBSERVABLE_PATTERN_NAME_PLURAL}-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
            )
        else:
            return fig

    @staticmethod
    def plot_result_radar_chart(
        simplex_coordinates_fingerprint: pd.DataFrame,
        categories_fingerprint: List[str],
        simplex_coordinates_explainable: pd.DataFrame,
        categories_explainable: List[str],
        title: str,
        use_config: bool = False
    ):

        assert simplex_coordinates_fingerprint.shape[0] == simplex_coordinates_explainable.shape[0], \
        'Error: Expected the same number of rows in simplex_coordinates_fingerprint and simplex_coordinates_explainable'   

        num_rows=simplex_coordinates_fingerprint.shape[0]
        num_cols=2
        plot_range = [0,1]

        row_titles = list(simplex_coordinates_explainable.index)

        fig = make_subplots(
                rows=num_rows,
                cols=num_cols,
                specs=[[{'type': 'polar', 'l':0.1}]*(num_cols)]*num_rows,
                horizontal_spacing=0.3, vertical_spacing=(0.05 / (num_rows - 1)),
                column_titles=["observable patterns", "explanatory features"],
            )
        
        for i in range(num_rows):

            fig.add_scatterpolar(
                r=simplex_coordinates_fingerprint.iloc[i],
                theta=categories_fingerprint,
                fill="toself",
                row=i+1,
                col=1,    
            )

            fig.add_annotation(
                dict(
                x= 0.5,
                y = 1 - ((i+0.9) / num_rows),
                text=row_titles[i],
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(size=14, color="black"),
                align="center"
            ))

            fig.add_scatterpolar(
                r=simplex_coordinates_explainable.iloc[i],
                theta=categories_explainable,
                fill="toself",
                row=i+1,
                col=2,
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            title=dict(
                text=title.replace('_', ' '),
                automargin=False,
                xref='paper'
            ),
            margin=dict(
                t=50,  # Top margin
                b=50,  # Bottom margin
                l=50,  # Left margin
                r=50   # Right margin
            ),
            height=300*num_rows,
            width=600
        )

        fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    range=plot_range,
                    showticklabels=False)
                )
        )

        if use_config:        
            output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            fig.write_image(
                f"{output_path}{config.DATASET_NAME}-{title}-radar_plot-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
            )
        else:
            return fig

    @staticmethod
    def make_regression_plot(
        distance_explainable: pd.DataFrame,
        distance_observable: pd.DataFrame,
        use_config: bool = False
    ):

        if set(distance_explainable.index.to_list()) != set(
            distance_observable.index.to_list()
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
            trendline_color_override='orange',
        )
        fig.update_layout(
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            width = 800,
            height = 800,
        )

        if use_config:
            output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            fig.write_image(
                f"{output_path}{config.DATASET_NAME}-regression_plot-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
            )
        else:
            return fig

    @staticmethod
    def plot_higher_order_feature_importances(
        human_readable_dict:dict
    ):

        fig = plt.figure(figsize=(20, 10))
        ax = sns.barplot(human_readable_dict)
        ax.axes.xaxis.set_tick_params(rotation=90)

        output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.tight_layout()

        fig.savefig(
            f"{output_path}{config.DATASET_NAME}-higher_order_feature_importances-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
        )

    @staticmethod
    def plot_homogeneity(hom_df:pd.DataFrame):
        fig = plt.figure(figsize=(10, 5))
        dark_black_palette = sns.dark_palette("#000000", n_colors=hom_df.shape[1], reverse=True)
        ax = sns.barplot(hom_df, palette=dark_black_palette)
        ax.axes.set_ylabel('Homogeneity')
        ax.axes.xaxis.set_tick_params(rotation=90)
        output_path = f"{config.OUTPUT_FOLDER_BASE}result_visualization/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plt.tight_layout()

        fig.savefig(
            f"{output_path}{config.DATASET_NAME}-homogeneity-{config.NUMBER_OBSERVABLE_PATTERNS}.pdf",
        )


# make radar plots:
# - fingerprint per group
# - scaled variants of observable patterns and of explainable features (as radar plot)
# - linear regression (x = distance observable, y = distance explainable)
