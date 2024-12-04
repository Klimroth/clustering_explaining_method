from data_preparation import DataPreparator
from apply_clustering import ClusteringApplier
import config

CONDUCT_DATA_PREPARATION = True
CONDUCT_GAP_STAT_ANALYSIS = True
CONDUCT_OBSERVABLE_CLUSTERING = True
CONDUCT_EXPLAINABLE_DISTANCES = True

optimal_number_of_clusters = config.NUMBER_OBSERVABLE_PATTERNS

if CONDUCT_DATA_PREPARATION:
    DataPreparator.prepare_data()

if CONDUCT_GAP_STAT_ANALYSIS:
    gap_analysis_result = ClusteringApplier.draw_gap_statistic_plot()

if CONDUCT_OBSERVABLE_CLUSTERING:
    ClusteringApplier.calculate_observable_patterns(_n_clusters=gap_analysis_result['n_clusters'])

if CONDUCT_EXPLAINABLE_DISTANCES:
    ClusteringApplier.calculate_explainable_distances()
